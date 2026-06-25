# src/models_transformers.py
# ===========================================================================
# CORRECTIONS PAR RAPPORT À L'ANCIENNE VERSION
# ---------------------------------------------------------------------------
#  1. FORMAT ABSA    : input = "[aspect] : [texte]"  → guide le modèle
#  2. DONNÉES        : lit corpus_absa_clean.csv (aspects mappés) en priorité
#  3. SPLIT 70/15/15 : conforme au §2.4.5 du mémoire (fini le 5-fold CV Transformer)
#  4. fp16 + grad_checkpointing + batch=8  → compatible T4 16 Go
#  5. max_length=96  → 25 % plus rapide, couvre 95 % des avis courts
#  6. WeightedTrainer → Cross-Entropy pondérée pour neg/pos/neutral (40/40/20)
#  7. Évaluation par ASPECT → cellules de la matrice ABSA
#  8. Surveillance mémoire GPU → lisible dans Colab
#  9. Résultats JSON → compatible matrix_absa.py
# 10. Gestion propre des cas "données insuffisantes"
# 11. Bug corrigé : target_names dans classification_report (ordre incohérent
#     dans l'ancienne version)
# ===========================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PROCESSED, RESULTS_DIR, MODELS_DIR, LOGS_DIR,
    TRANSFORMERS_PARAMS, RANDOM_SEED,
    POLARITY_MAPPING, POLARITY_NAMES, POLARITY_CLASSES,
)
from utils import setup_logger, set_seed, save_json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")  # pas d'interface graphique (Colab/serveur)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# ASPECTS : définis ici pour éviter de modifier config.py (à y ajouter plus tard)
# ---------------------------------------------------------------------------
ASPECTS = ["qualité", "service", "prix", "livraison", "interface"]


# ===========================================================================
# 0. UTILITAIRE — SURVEILLANCE MÉMOIRE GPU
# ===========================================================================

def log_gpu_memory(tag: str = "") -> None:
    """Affiche la mémoire GPU réservée — utile dans Colab."""
    if torch.cuda.is_available():
        used  = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️  GPU [{tag}] {used:.1f} / {total:.1f} Go réservés")
    else:
        logger.info("⚠️  Aucun GPU détecté — entraînement sur CPU (très lent).")


# ===========================================================================
# 1. DATASET PYTORCH
# ===========================================================================

class ABSADataset(Dataset):
    """
    Dataset PyTorch pour la classification de polarité aspectuelle.

    Format d'entrée : "[aspect] : [texte]"
    Exemple : "qualité : La batterie tient très bien la charge."

    Ce format dit explicitement au modèle sur quel aspect se concentrer
    (standard en ABSA avec Transformers — cf. BERT-PT, APC).
    Le padding dynamique est géré par DataCollatorWithPadding.
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 96):
        self.texts     = [str(t) for t in texts]
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,          # DataCollatorWithPadding s'en charge
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ===========================================================================
# 2. TRAINER PONDÉRÉ (CORRECTION DU DÉSÉQUILIBRE 40/40/20)
# ===========================================================================

class WeightedTrainer(Trainer):
    """
    Trainer HuggingFace avec Cross-Entropy pondérée.

    Pourquoi : la classe "neutre" représente seulement 20 % du corpus
    alors que neg et pos en représentent chacun 40 %.
    Sans pondération, le modèle ignore souvent la classe minoritaire
    et produit un Macro-F1 trop optimiste.

    La pondération est calculée via compute_class_weight('balanced')
    de scikit-learn et passée à la CrossEntropyLoss.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ===========================================================================
# 3. CLASSE PRINCIPALE TransformerModel
# ===========================================================================

class TransformerModel:
    """
    Pipeline complet pour un modèle Transformer ABSA multilingue.

    Utilisation rapide dans Colab :
        model = TransformerModel("xlm-roberta-base", lang="all")
        model.load_data()
        model.run_pipeline()
        model.save_results()
    """

    # Fichiers de données cherchés dans l'ordre (priorité décroissante)
    _DATA_PRIORITY = [
        "corpus_absa_clean.csv",     # produit par build_clean_corpus.py ← PRÉFÉRÉ
        "all_datasets_tf_ready.csv", # ancien fichier (fallback si clean absent)
    ]

    # ------------------------------------------------------------------
    # 3.1  CONSTRUCTEUR
    # ------------------------------------------------------------------

    def __init__(self, model_name: str, lang: str = "all", num_labels: int = 3):
        """
        Args:
            model_name  : identifiant HuggingFace  ex. "xlm-roberta-base"
            lang        : "fr", "en", "ru" ou "all" (pas de filtre)
            num_labels  : nombre de classes de polarité (3 par défaut)
        """
        self.model_name  = model_name
        self.lang        = lang
        self.num_labels  = num_labels
        self.tokenizer   = None
        self.model       = None
        self.trainer     = None
        self.results: dict = {}
        self.has_enough_data: bool = False
        self.df: pd.DataFrame = pd.DataFrame()
        self._test_df: pd.DataFrame = pd.DataFrame()   # ← test set isolé

        set_seed(RANDOM_SEED)

        # Paramètres depuis config.py (valeurs Colab-safe par défaut)
        self.params = TRANSFORMERS_PARAMS.get(model_name, {})
        if not self.params:
            logger.warning(f"⚠️  {model_name} absent de TRANSFORMERS_PARAMS — valeurs par défaut.")
            self.params = {
                "model_name":    model_name,
                "num_labels":    num_labels,
                "learning_rate": 2e-5,
                "batch_size":    8,
                "epochs":        3,
                "max_length":    96,
                "warmup_ratio":  0.1,
                "weight_decay":  0.01,
            }

        logger.info(f"🔧 TransformerModel : {model_name} | lang={lang}")

    # ------------------------------------------------------------------
    # 3.2  CHARGEMENT DES DONNÉES
    # ------------------------------------------------------------------

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charge et prépare les données pour l'entraînement ABSA.

        Transformations appliquées :
          - Filtre par langue si lang != "all"
          - Formate l'entrée : "[aspect] : [texte brut]"
          - Encode la polarité en entier via POLARITY_MAPPING
          - Supprime les lignes avec polarité inconnue
        """
        if filepath is None:
            filepath = self._find_data_file()

        logger.info(f"📂 Chargement : {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)

        # ── Filtre langue
        if self.lang != "all":
            df = df[df["lang"] == self.lang].copy()
            logger.info(f"   Filtre lang='{self.lang}' → {len(df)} lignes")

        # ── Vérifications minimales
        for col in ("text", "polarity", "lang"):
            if col not in df.columns:
                raise ValueError(f"Colonne obligatoire absente : '{col}' dans {filepath}")

        # ── Contrôle du nombre de classes
        unique_pol = df["polarity"].dropna().unique()
        if len(unique_pol) < 2:
            logger.warning(f"⚠️  {self.lang} : une seule classe ({unique_pol}) — pipeline ignoré.")
            self.has_enough_data = False
            self.df = df
            return df

        if len(df) < 50:
            logger.warning(f"⚠️  Seulement {len(df)} lignes pour lang='{self.lang}' — pipeline ignoré.")
            self.has_enough_data = False
            self.df = df
            return df

        # ── Format ABSA : "[aspect] : [texte]"
        if "aspect" in df.columns:
            df["absa_input"] = (
                df["aspect"].fillna("général").str.lower().str.strip()
                + " : "
                + df["text"].fillna("")
            )
        else:
            logger.info("   ℹ️  Colonne 'aspect' absente — texte brut utilisé.")
            df["absa_input"] = df["text"].fillna("")

        # ── Encodage polarité
        df["label"] = df["polarity"].map(POLARITY_MAPPING)
        n_unknown = df["label"].isna().sum()
        if n_unknown > 0:
            logger.warning(f"   ⚠️  {n_unknown} lignes avec polarité inconnue → supprimées.")
            df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        self.df = df.reset_index(drop=True)
        self.has_enough_data = True

        logger.info(f"   ✅ {len(df)} lignes prêtes")
        logger.info(f"   Polarités  : {dict(df['polarity'].value_counts())}")
        if "aspect" in df.columns:
            logger.info(f"   Aspects    : {dict(df['aspect'].value_counts())}")

        return self.df

    def _find_data_file(self) -> str:
        """Retourne le premier fichier de données disponible."""
        for fname in self._DATA_PRIORITY:
            path = os.path.join(DATA_PROCESSED, fname)
            if os.path.exists(path):
                logger.info(f"   📁 Fichier sélectionné : {fname}")
                return path
        raise FileNotFoundError(
            f"Aucun fichier de données trouvé dans {DATA_PROCESSED}.\n"
            f"Lancer build_clean_corpus.py d'abord (voir src/build_clean_corpus.py)."
        )

    # ------------------------------------------------------------------
    # 3.3  SPLIT STRATIFIÉ 70 / 15 / 15
    # ------------------------------------------------------------------

    def _split_data(self):
        """
        Découpe stratifiée 70 / 15 / 15 (train / val / test).

        Conforme au protocole §2.4.5 du mémoire :
        « Le corpus est divisé selon une répartition stratifiée 70/15/15. »
        Le jeu de test est isolé et évalué UNE SEULE FOIS après
        la sélection définitive des hyperparamètres.

        Retourne : X_train, X_val, X_test, y_train, y_val, y_test,
                   df_test (pour l'évaluation par aspect)
        """
        X = self.df["absa_input"].tolist()
        y = self.df["label"].tolist()

        # Split 1 : train+val (85 %) / test (15 %)
        X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
            X, y, range(len(X)),
            test_size=0.15,
            stratify=y,
            random_state=RANDOM_SEED,
        )

        # Split 2 : train (≈70 % total) / val (≈15 % total)
        val_ratio = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=val_ratio,
            stratify=y_tv,
            random_state=RANDOM_SEED,
        )

        # Stocker le DataFrame test pour l'évaluation par aspect
        self._test_df = self.df.iloc[list(idx_test)].reset_index(drop=True)

        logger.info(
            f"   ✂️  Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # 3.4  TOKENIZER & MODÈLE
    # ------------------------------------------------------------------

    def _load_tokenizer_and_model(self) -> None:
        """Charge le tokenizer et le modèle depuis HuggingFace Hub."""
        logger.info(f"📥 Chargement du modèle : {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        # gradient_checkpointing : réduit la VRAM de ~30 % (vitesse -20 % seulement)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("   ✅ gradient_checkpointing activé")
        log_gpu_memory("après chargement modèle")

    def _reset_model(self) -> None:
        """Recharge le modèle depuis zéro et vide le cache GPU."""
        self.tokenizer = None
        self.model     = None
        self.trainer   = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._load_tokenizer_and_model()

    # ------------------------------------------------------------------
    # 3.5  MÉTRIQUES
    # ------------------------------------------------------------------

    def _compute_metrics(self, eval_pred) -> dict:
        """Métriques calculées à chaque évaluation pendant l'entraînement."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "f1_macro":        f1_score(labels, preds, average="macro", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro":    recall_score(labels, preds, average="macro", zero_division=0),
            "accuracy":        accuracy_score(labels, preds),
        }

    # ------------------------------------------------------------------
    # 3.6  CONSTRUCTION DES TrainingArguments
    # ------------------------------------------------------------------

    def _build_training_args(self, output_dir: str, has_eval: bool) -> TrainingArguments:
        """
        TrainingArguments optimisés pour Colab T4 (16 Go VRAM).

        Paramètre           Valeur   Raison
        ------------------  -------  -----------------------------------------
        fp16                True     Demi-précision → -40 % VRAM, -30 % temps
        batch_size          8        Évite les OOM avec XLM-RoBERTa (270 M params)
        gradient_accum      2        Simule batch effectif de 16 sans OOM
        max_length          96       Couvre 95 % des avis courts (< 30 mots)
        save_total_limit    1        Économise l'espace disque Colab (15 Go)
        eval_steps          50       Évaluation fréquente → early stopping réactif
        """
        use_fp16   = torch.cuda.is_available()
        batch_size = self.params.get("batch_size", 8)

        return TrainingArguments(
            output_dir=output_dir,
            # ── Évaluation & arrêt précoce
            eval_strategy="steps"  if has_eval else "no",
            eval_steps=50          if has_eval else None,
            load_best_model_at_end=has_eval,
            metric_for_best_model="f1_macro" if has_eval else None,
            greater_is_better=True,
            # ── Sauvegarde (minimal pour Colab)
            save_strategy="steps"  if has_eval else "no",
            save_steps=100,
            save_total_limit=1,
            # ── Hyperparamètres
            learning_rate=self.params.get("learning_rate", 2e-5),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=self.params.get("epochs", 3),
            weight_decay=self.params.get("weight_decay", 0.01),
            warmup_ratio=self.params.get("warmup_ratio", 0.1),
            # ── Optimisations Colab T4
            fp16=use_fp16,
            gradient_accumulation_steps=2,   # batch effectif = 8 × 2 = 16
            dataloader_num_workers=2,
            # ── Logging
            logging_dir=os.path.join(LOGS_DIR, "transformers"),
            logging_steps=25,
            report_to="none",
            seed=RANDOM_SEED,
            disable_tqdm=False,
        )

    # ------------------------------------------------------------------
    # 3.7  ENTRAÎNEMENT
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: list, y_train: list,
        X_val:   list = None, y_val: list = None,
    ):
        """
        Entraîne le modèle avec :
          - Cross-Entropy pondérée (WeightedTrainer)
          - Early stopping (patience=3) si un jeu de validation est fourni
          - DataCollatorWithPadding (padding dynamique → moins de calcul inutile)

        La pondération des classes compense le déséquilibre 40/40/20 :
        la classe neutre, minoritaire, reçoit automatiquement un poids plus élevé.
        """
        if not self.has_enough_data:
            logger.warning("⚠️  Pas assez de données — entraînement annulé.")
            return None

        if self.tokenizer is None or self.model is None:
            self._load_tokenizer_and_model()

        max_len  = self.params.get("max_length", 96)
        collator = DataCollatorWithPadding(self.tokenizer)

        train_dataset = ABSADataset(X_train, y_train, self.tokenizer, max_len)
        eval_dataset  = (
            ABSADataset(X_val, y_val, self.tokenizer, max_len) if X_val else None
        )

        # ── Pondération des classes
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        w_tensor = torch.tensor(weights, dtype=torch.float)

        # Afficher les poids pour le rapport
        pol_labels = [POLARITY_NAMES.get(c, str(c)) for c in classes]
        logger.info(f"   ⚖️  Poids classes : { {p: round(w, 3) for p, w in zip(pol_labels, weights)} }")

        output_dir = os.path.join(MODELS_DIR, self.model_name.replace("/", "_"))
        training_args = self._build_training_args(output_dir, has_eval=(eval_dataset is not None))

        callbacks = []
        if eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        self.trainer = WeightedTrainer(
            class_weights=w_tensor,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )

        logger.info(f"🚀 Entraînement sur {len(X_train)} exemples (lang={self.lang})...")
        log_gpu_memory("avant entraînement")
        self.trainer.train()
        log_gpu_memory("après entraînement")
        logger.info("   ✅ Entraînement terminé")
        return self.trainer

    # ------------------------------------------------------------------
    # 3.8  PRÉDICTION
    # ------------------------------------------------------------------

    def predict(self, texts: list) -> np.ndarray:
        """Prédit les polarités sur une liste de textes ABSA formatés."""
        if self.trainer is None:
            raise RuntimeError("Appeler train() avant predict().")
        max_len = self.params.get("max_length", 96)
        dataset = ABSADataset(texts, [0] * len(texts), self.tokenizer, max_len)
        output  = self.trainer.predict(dataset)
        return np.argmax(output.predictions, axis=1)

    # ------------------------------------------------------------------
    # 3.9  ÉVALUATION GLOBALE
    # ------------------------------------------------------------------

    def evaluate_on_test(self, X_test: list, y_test: list) -> dict:
        """
        Évalue sur le jeu de test isolé (une seule fois).
        Retourne les métriques globales pour la matrice ABSA.
        """
        logger.info(f"🔍 Évaluation sur {len(X_test)} exemples de test...")
        y_pred = self.predict(X_test)

        # BUG CORRIGÉ : target_names dans l'ordre du POLARITY_MAPPING
        # neg→0, neutral→1, pos→2  ↔  POLARITY_CLASSES = ["Négatif","Neutre","Positif"]
        report = classification_report(
            y_test, y_pred,
            target_names=POLARITY_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "f1_macro":                 round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "precision_macro":          round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall_macro":             round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "accuracy":                 round(accuracy_score(y_test, y_pred), 4),
            "f1_per_class": {
                POLARITY_CLASSES[i]: round(f1_score(y_test, y_pred, average=None, zero_division=0)[i], 4)
                for i in range(self.num_labels)
            },
            "classification_report":    report,
            "confusion_matrix":         cm.tolist(),
            "n_test":                   len(X_test),
        }

        logger.info(f"   Macro-F1  : {metrics['f1_macro']}")
        logger.info(f"   Précision : {metrics['precision_macro']}")
        logger.info(f"   Rappel    : {metrics['recall_macro']}")
        return metrics

    # ------------------------------------------------------------------
    # 3.10  ÉVALUATION PAR ASPECT (cellules de la matrice ABSA)
    # ------------------------------------------------------------------

    def evaluate_per_aspect(self, y_test: list) -> dict:
        """
        Évalue le Macro-F1 par aspect sur le jeu de test.

        Retourne un dict prêt pour matrix_absa.py :
            { "qualité": 0.72, "service": 0.68, "prix": 0.65, ... }

        Nécessite que _test_df soit rempli par _split_data().
        Si la colonne 'aspect' est absente, retourne un dict vide.
        """
        if self._test_df.empty or "aspect" not in self._test_df.columns:
            logger.info("   ℹ️  Évaluation par aspect ignorée (colonne 'aspect' absente).")
            return {}

        # Reconstruire les inputs ABSA du test set
        X_test_absa = self._test_df["absa_input"].tolist()
        y_pred_all  = self.predict(X_test_absa)
        y_test_all  = np.array(y_test)

        per_aspect: dict = {}
        for asp in ASPECTS:
            mask = (self._test_df["aspect"].values == asp)
            n = mask.sum()
            if n < 5:
                logger.info(f"   ⚠️  '{asp}' : {n} exemples test (< 5, ignoré)")
                continue
            f1 = f1_score(
                y_test_all[mask], y_pred_all[mask],
                average="macro", zero_division=0,
            )
            per_aspect[asp] = round(f1, 4)
            logger.info(f"   Aspect '{asp:<12}' → Macro-F1 = {f1:.4f}  (n={n})")

        return per_aspect

    # ------------------------------------------------------------------
    # 3.11  PIPELINE COMPLET
    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict:
        """
        Exécute l'intégralité du pipeline en une seule méthode.

        Étapes :
          1. Split stratifié 70 / 15 / 15
          2. Chargement du modèle
          3. Entraînement avec early stopping sur le jeu de validation
          4. Évaluation globale sur le jeu de test
          5. Évaluation par aspect (si disponible)

        Retourne un dict complet sauvegardé par save_results().
        """
        if not self.has_enough_data:
            logger.warning(f"⚠️  Pipeline annulé : {self.model_name} / lang={self.lang}")
            return {}

        logger.info("=" * 60)
        logger.info(f"▶  PIPELINE  {self.model_name}  |  lang={self.lang}")
        logger.info("=" * 60)

        # 1. Split
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data()

        # 2. & 3. Modèle + entraînement
        self._reset_model()
        self.train(X_train, y_train, X_val, y_val)

        # 4. Évaluation globale
        metrics_global = self.evaluate_on_test(X_test, y_test)

        # 5. Évaluation par aspect
        metrics_per_aspect = self.evaluate_per_aspect(y_test)

        # Assembler les résultats
        self.results = {
            "model_name":  self.model_name,
            "lang":        self.lang,
            "n_train":     len(X_train),
            "n_val":       len(X_val),
            "n_test":      len(X_test),
            "global":      metrics_global,
            "per_aspect":  metrics_per_aspect,
            "timestamp":   datetime.now().isoformat(),
        }

        logger.info(
            f"✅ Pipeline terminé — Macro-F1 global : {metrics_global['f1_macro']}"
        )
        return self.results

    # ------------------------------------------------------------------
    # 3.12  SAUVEGARDES
    # ------------------------------------------------------------------

    def save_model(self, filepath: str = None) -> str:
        """Sauvegarde le modèle et le tokenizer fine-tunés."""
        if self.model is None:
            raise RuntimeError("Entraîner le modèle avant de le sauvegarder.")
        if filepath is None:
            name   = self.model_name.replace("/", "_")
            suffix = self.lang if self.lang != "all" else "multilingual"
            filepath = os.path.join(MODELS_DIR, f"{name}_{suffix}")
        os.makedirs(filepath, exist_ok=True)
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"💾 Modèle sauvegardé : {filepath}")
        return filepath

    def save_results(self, filepath: str = None) -> str:
        """Sauvegarde les résultats JSON (compatible matrix_absa.py)."""
        if not self.results:
            raise RuntimeError("Pas de résultats. Appeler run_pipeline() d'abord.")
        if filepath is None:
            name   = self.model_name.replace("/", "_")
            suffix = self.lang if self.lang != "all" else "multilingual"
            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"transformer_{name}_{suffix}_{ts}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_json(self.results, filepath)
        logger.info(f"💾 Résultats sauvegardés : {filepath}")
        return filepath

    def plot_confusion_matrix(self, save: bool = True) -> None:
        """Génère et sauvegarde la matrice de confusion du jeu de test."""
        cm_data = self.results.get("global", {}).get("confusion_matrix")
        if cm_data is None:
            logger.warning("   Pas de matrice de confusion (run_pipeline() requis).")
            return
        cm  = np.array(cm_data)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=POLARITY_CLASSES,
            yticklabels=POLARITY_CLASSES,
            ax=ax,
        )
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Vérités terrain")
        ax.set_title(f"{self.model_name} ({self.lang})")
        plt.tight_layout()
        if save:
            name   = self.model_name.replace("/", "_")
            suffix = self.lang if self.lang != "all" else "multilingual"
            fig_dir = os.path.join(RESULTS_DIR, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            fpath = os.path.join(fig_dir, f"confusion_{name}_{suffix}.png")
            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            logger.info(f"💾 Matrice de confusion : {fpath}")
        plt.close(fig)


# ===========================================================================
# 4. FONCTIONS DE LANCEMENT PAR MODÈLE
# ===========================================================================

def run_xlmr(data_file: str = None) -> dict:
    """
    XLM-RoBERTa multilingue — toutes langues.
    Durée estimée sur T4 avec ~2 400 lignes : 40–55 min.
    """
    m = TransformerModel("xlm-roberta-base", lang="all")
    m.load_data(data_file)
    res = m.run_pipeline()
    if res:
        m.save_model()
        m.save_results()
        m.plot_confusion_matrix()
    return res


def run_camembert(data_file: str = None) -> dict:
    """
    CamemBERT — données françaises uniquement.
    Durée estimée sur T4 avec ~700 lignes FR : 15–20 min.
    """
    m = TransformerModel("camembert-base", lang="fr")
    m.load_data(data_file)
    res = m.run_pipeline()
    if res:
        m.save_model()
        m.save_results()
        m.plot_confusion_matrix()
    return res


def run_rubert(data_file: str = None) -> dict:
    """
    RuBERT — données russes uniquement.
    Durée estimée sur T4 avec ~800 lignes RU : 20–30 min.
    ⚠️  Nécessite corpus_absa_clean.csv avec aspects RU mappés.
    """
    m = TransformerModel("DeepPavlov/rubert-base-cased", lang="ru")
    m.load_data(data_file)
    res = m.run_pipeline()
    if res:
        m.save_model()
        m.save_results()
        m.plot_confusion_matrix()
    return res


# ===========================================================================
# 5. POINT D'ENTRÉE (exécution directe)
# ===========================================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("🚀  ENTRAÎNEMENT TRANSFORMERS — PIPELINE ABSA MULTILINGUE")
    logger.info("=" * 60)
    log_gpu_memory("démarrage")

    summary: dict = {}

    configs = [
        ("xlm-roberta-base",             "all", run_xlmr),
        ("camembert-base",               "fr",  run_camembert),
        ("DeepPavlov/rubert-base-cased", "ru",  run_rubert),
    ]

    for model_name, lang, run_fn in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"▶  {model_name}  |  lang={lang}")
        logger.info(f"{'='*60}")
        try:
            res = run_fn()
            if res:
                summary[f"{model_name}_{lang}"] = {
                    "f1_macro":        res["global"]["f1_macro"],
                    "precision_macro": res["global"]["precision_macro"],
                    "recall_macro":    res["global"]["recall_macro"],
                    "n_test":          res["global"]["n_test"],
                    "per_aspect":      res.get("per_aspect", {}),
                }
        except Exception as exc:
            logger.error(f"❌ Erreur {model_name} : {exc}", exc_info=True)

    # ── Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("📊  RÉSUMÉ FINAL DES TRANSFORMERS")
    logger.info("=" * 60)
    for key, r in summary.items():
        logger.info(f"\n  {key}")
        logger.info(f"    Macro-F1  : {r['f1_macro']}")
        logger.info(f"    Précision : {r['precision_macro']}")
        logger.info(f"    Rappel    : {r['recall_macro']}")
        logger.info(f"    Test (n)  : {r['n_test']}")
        if r["per_aspect"]:
            logger.info("    Par aspect :")
            for asp, f1 in r["per_aspect"].items():
                logger.info(f"      {asp:<14}: {f1}")

    # Sauvegarder le résumé global
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"transformers_summary_{ts}.json")
    save_json(summary, out)
    logger.info(f"\n💾 Résumé global → {out}")