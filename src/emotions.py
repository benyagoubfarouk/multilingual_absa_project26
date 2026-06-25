# src/emotions.py
# ===========================================================================
# Détection des 5 émotions de Plutchik (joie, satisfaction, insatisfaction,
# colère, tristesse) par fine-tuning d'un Transformer.
#
# CORRECTIONS vs ancienne version :
#   1. DONNÉES : lit corpus_absa_with_emotions.csv si disponible,
#      sinon applique une heuristique DÉTERMINISTE (plus de random)
#   2. SPLIT   : 70/15/15 stratifié (fini le 5-fold CV)
#   3. COLAB   : fp16, batch=8, max_length=96, gradient_checkpointing
#   4. LOSS    : WeightedTrainer pour compenser le déséquilibre
#   5. FORMAT  : DataCollatorWithPadding (padding dynamique)
#   6. eval_strategy (remplace evaluation_strategy déprécié)
# ===========================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PROCESSED, RESULTS_DIR, MODELS_DIR, LOGS_DIR,
    EMOTION_CLASSES, EMOTION_MAPPING, EMOTION_MODEL_PARAMS,
    RANDOM_SEED, POLARITY_MAPPING,
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
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

logger = setup_logger(__name__)

EMOTION_NAMES = {v: k for k, v in EMOTION_MAPPING.items()}

# ---------------------------------------------------------------------------
# HEURISTIQUE DÉTERMINISTE (utilisée si corpus_absa_with_emotions.csv absent)
# ---------------------------------------------------------------------------
# Remplacement du random.choice précédent par une règle basée sur
# le contenu lexical du texte + la polarité → résultat reproductible.

_EMOTION_KEYWORDS = {
    "joie": [
        "excellent", "parfait", "génial", "super", "fantastique", "adoré",
        "wonderful", "amazing", "love", "perfect", "great", "best",
        "отлично", "прекрасно", "замечательно",
    ],
    "satisfaction": [
        "bien", "bon", "correct", "satisfait", "bonne", "good", "nice",
        "ok", "хорошо", "нормально", "доволен",
    ],
    "insatisfaction": [
        "décevant", "insuffisant", "moyen", "mediocre", "disappointing",
        "average", "разочарован", "посредственно",
    ],
    "colère": [
        "horrible", "nul", "scandaleux", "inacceptable", "terrible",
        "awful", "worst", "hate", "disgusting", "ужасно", "отвратительно",
    ],
    "tristesse": [
        "déçu", "dommage", "regret", "unfortunate", "sad", "disappointed",
        "unfortunately", "жаль", "разочарование",
    ],
}


def _assign_emotion_deterministic(text: str, polarity: str) -> str:
    """
    Attribue une émotion de manière déterministe basée sur :
      1. Mots-clés détectés dans le texte
      2. Polarité de repli si aucun mot-clé trouvé

    Remplace le random.choice de l'ancienne version.
    """
    text_lower = str(text).lower()

    if polarity == "pos":
        for kw in _EMOTION_KEYWORDS["joie"]:
            if kw in text_lower:
                return "joie"
        return "satisfaction"

    elif polarity == "neg":
        for kw in _EMOTION_KEYWORDS["colère"]:
            if kw in text_lower:
                return "colère"
        for kw in _EMOTION_KEYWORDS["tristesse"]:
            if kw in text_lower:
                return "tristesse"
        return "insatisfaction"

    else:  # neutral
        for kw in _EMOTION_KEYWORDS["insatisfaction"]:
            if kw in text_lower:
                return "insatisfaction"
        return "satisfaction"


# ===========================================================================
# 1. DATASET PYTORCH
# ===========================================================================

class EmotionDataset(Dataset):
    """Dataset pour la classification des 5 émotions de Plutchik."""

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
            padding=False,     # DataCollatorWithPadding gère le padding
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ===========================================================================
# 2. TRAINER PONDÉRÉ (5 classes d'émotions souvent déséquilibrées)
# ===========================================================================

class WeightedEmotionTrainer(Trainer):
    """CrossEntropyLoss pondérée pour compenser le déséquilibre des émotions."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss    = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ===========================================================================
# 3. CLASSE PRINCIPALE EmotionModel
# ===========================================================================

class EmotionModel:
    """
    Pipeline complet pour la détection des 5 émotions de Plutchik.

    Priorité de chargement des données :
      1. corpus_absa_with_emotions.csv  (labels silver via emotion_silver_labeler.py)
      2. corpus_absa_clean.csv          (labels heuristiques déterministes)
      3. all_datasets_tf_ready.csv      (ancien fallback)
    """

    _DATA_PRIORITY = [
        "corpus_absa_with_emotions.csv",   # ← silver labels (meilleur)
        "corpus_absa_clean.csv",            # ← heuristique déterministe
        "all_datasets_tf_ready.csv",        # ← ancien fallback
    ]

    def __init__(self, model_name: str = "camembert-base", lang: str = "fr"):
        self.model_name = model_name
        self.lang       = lang
        self.num_labels = 5
        self.tokenizer  = None
        self.model      = None
        self.trainer    = None
        self.results: dict = {}
        self.has_enough_data: bool = False
        self.df: pd.DataFrame = pd.DataFrame()
        self._test_df: pd.DataFrame = pd.DataFrame()

        set_seed(RANDOM_SEED)
        self.params = EMOTION_MODEL_PARAMS.copy()
        logger.info(f"🔧 EmotionModel : {model_name} | lang={lang}")

    # ------------------------------------------------------------------
    # 3.1  CHARGEMENT DES DONNÉES
    # ------------------------------------------------------------------

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charge les données et prépare les labels d'émotions.

        Si la colonne 'emotion' est absente (anciens fichiers),
        elle est générée par heuristique déterministe — plus de random.
        """
        if filepath is None:
            filepath = self._find_data_file()

        logger.info(f"📂 Chargement : {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)

        if self.lang != "all":
            df = df[df["lang"] == self.lang].copy()
            logger.info(f"   Filtre lang='{self.lang}' → {len(df)} lignes")

        if len(df) < 30:
            logger.warning(f"⚠️  Pas assez de données ({len(df)}) pour lang='{self.lang}'.")
            self.has_enough_data = False
            self.df = df
            return df

        # ── Colonne emotion
        if "emotion" not in df.columns or df["emotion"].isna().all():
            logger.info("   ℹ️  Colonne 'emotion' absente → heuristique déterministe.")
            df["emotion"] = df.apply(
                lambda r: _assign_emotion_deterministic(
                    r.get("text", ""), r.get("polarity", "neutral")
                ),
                axis=1,
            )
        else:
            logger.info("   ✅ Colonne 'emotion' chargée depuis le fichier.")

        # ── Encodage
        df["label"] = df["emotion"].map(EMOTION_MAPPING)
        n_unknown = df["label"].isna().sum()
        if n_unknown > 0:
            logger.warning(f"   ⚠️  {n_unknown} émotions inconnues → supprimées.")
            df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        # ── Colonne texte
        text_col = "absa_input" if "absa_input" in df.columns else (
            "text_processed" if "text_processed" in df.columns else "text"
        )
        df["model_input"] = df[text_col].fillna("")

        self.df = df.reset_index(drop=True)
        self.has_enough_data = True

        logger.info(f"   ✅ {len(df)} lignes prêtes")
        logger.info(f"   Distribution émotions : {dict(df['emotion'].value_counts())}")
        return self.df

    def _find_data_file(self) -> str:
        for fname in self._DATA_PRIORITY:
            path = os.path.join(DATA_PROCESSED, fname)
            if os.path.exists(path):
                logger.info(f"   📁 Fichier : {fname}")
                return path
        raise FileNotFoundError(
            f"Aucun fichier de données trouvé dans {DATA_PROCESSED}.\n"
            "Lancer build_clean_corpus.py puis emotion_silver_labeler.py d'abord."
        )

    # ------------------------------------------------------------------
    # 3.2  SPLIT 70 / 15 / 15
    # ------------------------------------------------------------------

    def _split_data(self):
        X = self.df["model_input"].tolist()
        y = self.df["label"].tolist()

        X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
            X, y, range(len(X)),
            test_size=0.15, stratify=y, random_state=RANDOM_SEED,
        )
        val_ratio = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=val_ratio, stratify=y_tv, random_state=RANDOM_SEED,
        )
        self._test_df = self.df.iloc[list(idx_test)].reset_index(drop=True)
        logger.info(f"   ✂️  Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # 3.3  TOKENIZER & MODÈLE
    # ------------------------------------------------------------------

    def _load_tokenizer_and_model(self) -> None:
        logger.info(f"📥 Chargement : {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if torch.cuda.is_available():
            used  = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"   🖥️  GPU : {used:.1f}/{total:.1f} Go réservés")

    def _reset_model(self) -> None:
        self.tokenizer = None
        self.model     = None
        self.trainer   = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._load_tokenizer_and_model()

    # ------------------------------------------------------------------
    # 3.4  MÉTRIQUES
    # ------------------------------------------------------------------

    def _compute_metrics(self, eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "f1_macro":        f1_score(labels, preds, average="macro", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro":    recall_score(labels, preds, average="macro", zero_division=0),
            "accuracy":        accuracy_score(labels, preds),
        }

    # ------------------------------------------------------------------
    # 3.5  ENTRAÎNEMENT
    # ------------------------------------------------------------------

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if not self.has_enough_data:
            logger.warning("⚠️  Pas assez de données — entraînement ignoré.")
            return None
        if self.tokenizer is None or self.model is None:
            self._load_tokenizer_and_model()

        max_len  = self.params.get("max_length", 96)
        collator = DataCollatorWithPadding(self.tokenizer)

        train_ds = EmotionDataset(X_train, y_train, self.tokenizer, max_len)
        eval_ds  = EmotionDataset(X_val, y_val, self.tokenizer, max_len) if X_val else None

        # Pondération des 5 classes
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        w_tensor = torch.tensor(weights, dtype=torch.float)
        labels_str = [EMOTION_NAMES.get(c, str(c)) for c in classes]
        logger.info(f"   ⚖️  Poids : { {l: round(w, 3) for l, w in zip(labels_str, weights)} }")

        use_fp16 = torch.cuda.is_available()
        batch    = self.params.get("batch_size", 8)
        out_dir  = os.path.join(MODELS_DIR, f"emotion_{self.model_name.replace('/', '_')}")

        args = TrainingArguments(
            output_dir=out_dir,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=50 if eval_ds else None,
            save_strategy="steps" if eval_ds else "no",
            save_steps=100,
            save_total_limit=1,
            load_best_model_at_end=bool(eval_ds),
            metric_for_best_model="f1_macro" if eval_ds else None,
            greater_is_better=True,
            learning_rate=self.params.get("learning_rate", 2e-5),
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch * 2,
            num_train_epochs=self.params.get("epochs", 3),
            weight_decay=self.params.get("weight_decay", 0.01),
            warmup_ratio=self.params.get("warmup_ratio", 0.1),
            fp16=use_fp16,
            gradient_accumulation_steps=2,
            dataloader_num_workers=2,
            logging_dir=os.path.join(LOGS_DIR, "emotions"),
            logging_steps=25,
            report_to="none",
            seed=RANDOM_SEED,
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if eval_ds else []

        self.trainer = WeightedEmotionTrainer(
            class_weights=w_tensor,
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )

        logger.info(f"🚀 Entraînement émotions sur {len(X_train)} exemples...")
        self.trainer.train()
        logger.info("   ✅ Terminé")
        return self.trainer

    # ------------------------------------------------------------------
    # 3.6  PRÉDICTION & ÉVALUATION
    # ------------------------------------------------------------------

    def predict(self, texts: list) -> np.ndarray:
        if self.trainer is None:
            raise RuntimeError("Appeler train() avant predict().")
        max_len = self.params.get("max_length", 96)
        ds = EmotionDataset(texts, [0] * len(texts), self.tokenizer, max_len)
        return np.argmax(self.trainer.predict(ds).predictions, axis=1)

    def evaluate_on_test(self, X_test, y_test) -> dict:
        logger.info(f"🔍 Évaluation émotions sur {len(X_test)} exemples...")
        y_pred = self.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=EMOTION_CLASSES,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "f1_macro":              round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "precision_macro":       round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall_macro":          round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "accuracy":              round(accuracy_score(y_test, y_pred), 4),
            "classification_report": report,
            "confusion_matrix":      cm.tolist(),
            "n_test":                len(X_test),
        }
        logger.info(f"   Macro-F1 émotions : {metrics['f1_macro']}")
        return metrics

    # ------------------------------------------------------------------
    # 3.7  PIPELINE COMPLET
    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict:
        if not self.has_enough_data:
            logger.warning(f"⚠️  Pipeline émotions annulé : lang={self.lang}")
            return {}

        logger.info("=" * 60)
        logger.info(f"▶  ÉMOTIONS  {self.model_name}  |  lang={self.lang}")
        logger.info("=" * 60)

        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data()
        self._reset_model()
        self.train(X_train, y_train, X_val, y_val)
        metrics = self.evaluate_on_test(X_test, y_test)

        self.results = {
            "model_name": self.model_name,
            "lang":       self.lang,
            "n_train":    len(X_train),
            "n_val":      len(X_val),
            "n_test":     len(X_test),
            "global":     metrics,
            "timestamp":  datetime.now().isoformat(),
        }
        logger.info(f"✅ Émotions terminé — Macro-F1 : {metrics['f1_macro']}")
        return self.results

    # ------------------------------------------------------------------
    # 3.8  SAUVEGARDES
    # ------------------------------------------------------------------

    def save_model(self, filepath: str = None) -> str:
        if self.model is None:
            raise RuntimeError("Entraîner le modèle d'abord.")
        if filepath is None:
            name    = self.model_name.replace("/", "_")
            filepath = os.path.join(MODELS_DIR, f"emotion_{name}_{self.lang}")
        os.makedirs(filepath, exist_ok=True)
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"💾 Modèle émotions : {filepath}")
        return filepath

    def save_results(self, filepath: str = None) -> str:
        if not self.results:
            raise RuntimeError("Appeler run_pipeline() d'abord.")
        if filepath is None:
            name    = self.model_name.replace("/", "_")
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"emotion_{name}_{self.lang}_{ts}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_json(self.results, filepath)
        logger.info(f"💾 Résultats émotions : {filepath}")
        return filepath

    def plot_confusion_matrix(self, save: bool = True) -> None:
        cm_data = self.results.get("global", {}).get("confusion_matrix")
        if cm_data is None:
            return
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            np.array(cm_data), annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES, ax=ax,
        )
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Vérités terrain")
        ax.set_title(f"Émotions — {self.model_name} ({self.lang})")
        plt.tight_layout()
        if save:
            name  = self.model_name.replace("/", "_")
            fpath = os.path.join(FIGURES_DIR, f"emotion_confusion_{name}_{self.lang}.png")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            logger.info(f"💾 {fpath}")
        plt.close(fig)


# Import FIGURES_DIR pour plot_confusion_matrix
from config import FIGURES_DIR


# ===========================================================================
# 4. MATRICE ASPECT-SENTIMENT-ÉMOTION
# ===========================================================================

def generate_aspect_sentiment_emotion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère la matrice croisée aspect × sentiment → distribution des émotions.
    Retourne un DataFrame et sauvegarde un CSV.
    """
    if not all(c in df.columns for c in ("aspect", "polarity", "emotion")):
        logger.warning("Colonnes aspect/polarity/emotion requises.")
        return pd.DataFrame()

    matrix = (
        df.groupby(["aspect", "polarity", "emotion"])
        .size()
        .unstack(fill_value=0)
    )
    out = os.path.join(RESULTS_DIR, "aspect_sentiment_emotion_matrix.csv")
    matrix.to_csv(out)
    logger.info(f"💾 Matrice aspect-sentiment-émotion : {out}")
    return matrix


# ===========================================================================
# 5. POINT D'ENTRÉE
# ===========================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀  DÉTECTION DES ÉMOTIONS — PIPELINE")
    logger.info("=" * 60)

    configs = [
        ("camembert-base",               "fr"),
        ("xlm-roberta-base",             "all"),
        ("DeepPavlov/rubert-base-cased", "ru"),
    ]

    summary = {}
    for model_name, lang in configs:
        try:
            m = EmotionModel(model_name=model_name, lang=lang)
            m.load_data()
            res = m.run_pipeline()
            if res:
                m.save_model()
                m.save_results()
                m.plot_confusion_matrix()
                summary[f"{model_name}_{lang}"] = res["global"]["f1_macro"]
        except Exception as e:
            logger.error(f"❌ {model_name} : {e}", exc_info=True)

    logger.info("\n📊 RÉSUMÉ ÉMOTIONS")
    for k, f1 in summary.items():
        logger.info(f"   {k} → Macro-F1 = {f1}")

    # Matrice aspect-sentiment-émotion
    try:
        from config import DATA_WITH_EMOTIONS
        df = pd.read_csv(DATA_WITH_EMOTIONS)
        generate_aspect_sentiment_emotion_matrix(df)
    except Exception as e:
        logger.warning(f"Matrice aspect-sentiment-émotion ignorée : {e}")