# src/models_transformers.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PROCESSED, RESULTS_DIR, MODELS_DIR, LOGS_DIR,
    TRANSFORMERS_PARAMS, RANDOM_SEED, ASPECTS
)
from utils import setup_logger, set_seed, save_json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

logger = setup_logger(__name__)


# ============================================================
# 1. DATASET PYTORCH
# ============================================================

class ABSADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# 2. CLASSE MODÈLE PRINCIPAL
# ============================================================

class TransformerModel:
    def __init__(self, model_name, lang="all", num_labels=3):
        self.model_name = model_name
        self.lang = lang
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.results = {}
        self.df = None

        self.params = TRANSFORMERS_PARAMS.get(model_name, {})
        set_seed(RANDOM_SEED)
        logger.info(f"🔧 TransformerModel : {model_name} | lang={lang}")

    def load_data(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(DATA_PROCESSED, "corpus_absa_clean.csv")
        logger.info(f"📂 Chargement : {filepath}")
        self.df = pd.read_csv(filepath)
        if self.lang != "all":
            self.df = self.df[self.df['lang'] == self.lang]

        # Aspect mapping (pour le modèle)
        self.df['text_processed'] = self.df['text_processed'] + " [ASPECT: " + self.df['aspect'] + "]"

        self.X = self.df['text_processed'].tolist()
        self.y = self.df['polarity'].tolist()
        self.mapping = {'pos': 0, 'neutral': 1, 'neg': 2}
        self.y_int = [self.mapping[p] for p in self.y]
        
        logger.info(f"   ✅ {len(self.df)} lignes prêtes")
        logger.info(f"   Polarités  : {dict(self.df['polarity'].value_counts())}")
        logger.info(f"   Aspects    : {dict(self.df['aspect'].value_counts())}")
        return self.df

    def load_model(self):
        logger.info(f"📥 Chargement du modèle : {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels, ignore_mismatched_sizes=True
        )
        # Activation du gradient checkpointing pour économiser la VRAM
        self.model.gradient_checkpointing_enable()
        logger.info("   ✅ gradient_checkpointing activé")
        return self.tokenizer, self.model

    def _split_data(self, test_size=0.15, val_size=0.15):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y_int, test_size=test_size, random_state=RANDOM_SEED, stratify=self.y_int
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_train_val
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataset(self, texts, labels):
        if self.tokenizer is None:
            self.load_model()
        return ABSADataset(
            texts, labels, self.tokenizer, self.params.get('max_length', 128)
        )

    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0)
        }

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.load_model()

        train_dataset = self.create_dataset(X_train, y_train)
        eval_dataset = self.create_dataset(X_val, y_val) if X_val else None

        args = TrainingArguments(
            output_dir=os.path.join(MODELS_DIR, self.model_name.replace('/', '_')),
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            per_device_train_batch_size=self.params.get('batch_size', 8),
            per_device_eval_batch_size=self.params.get('batch_size', 8),
            learning_rate=self.params.get('learning_rate', 2e-5),
            num_train_epochs=self.params.get('epochs', 3),
            weight_decay=self.params.get('weight_decay', 0.01),
            warmup_ratio=self.params.get('warmup_ratio', 0.1),
            logging_dir=os.path.join(LOGS_DIR, 'transformers'),
            logging_steps=20,
            save_total_limit=2,
            seed=RANDOM_SEED,
            fp16=True,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="f1_macro" if eval_dataset else None,
            report_to="none",
            disable_tqdm=False,
        )

        # Calcul des poids des classes pour gérer le déséquilibre
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.model.device)

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset else None,
        )
        # On définit les poids des classes dans le modèle
        self.model.class_weights = class_weights
        logger.info(f"   ⚖️  Poids classes : {dict(zip(['pos','neutral','neg'], class_weights.cpu().numpy()))}")

        logger.info(f"🚀 Début de l'entraînement ({len(X_train)} échantillons)")
        self.trainer.train()
        logger.info("   ✅ Entraînement terminé")

    def predict(self, texts):
        if self.trainer is None:
            raise ValueError("Modèle non entraîné.")
        dataset = self.create_dataset(texts, [0]*len(texts))
        preds = self.trainer.predict(dataset)
        return np.argmax(preds.predictions, axis=1)

    def evaluate_test(self, X_test, y_test):
        y_pred = self.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        self.results = {
            'global_f1_macro': f1,
            'global_precision_macro': prec,
            'global_recall_macro': rec,
            'n_samples': len(y_test)
        }
        logger.info(f"📊 Test : F1 {f1:.4f} / Préc. {prec:.4f} / Rappel {rec:.4f}")
        return self.results

    def save_model(self):
        path = os.path.join(MODELS_DIR, self.model_name.replace('/', '_'))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"💾 Modèle sauvegardé : {path}")

    def save_results(self):
        path = os.path.join(RESULTS_DIR, f"transformer_{self.model_name.replace('/', '_')}.json")
        save_json(self.results, path)
        logger.info(f"💾 Résultats sauvegardés : {path}")


# ============================================================
# 3. FONCTIONS DE LANCEMENT
# ============================================================

def run_xlmr():
    logger.info("=" * 60)
    logger.info("▶  PIPELINE  xlm-roberta-base  |  lang=all")
    logger.info("=" * 60)
    m = TransformerModel("xlm-roberta-base", lang="all")
    m.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = m._split_data()
    logger.info(f"   ✂️  Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
    m.train(X_train, y_train, X_val, y_val)
    m.evaluate_test(X_test, y_test)
    m.save_model()
    m.save_results()
    return m.results

def run_camembert():
    logger.info("=" * 60)
    logger.info("▶  PIPELINE  camembert-base  |  lang=fr")
    logger.info("=" * 60)
    m = TransformerModel("camembert-base", lang="fr")
    m.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = m._split_data()
    logger.info(f"   ✂️  Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
    m.train(X_train, y_train, X_val, y_val)
    m.evaluate_test(X_test, y_test)
    m.save_model()
    m.save_results()
    return m.results

def run_rubert():
    logger.info("=" * 60)
    logger.info("▶  PIPELINE  DeepPavlov/rubert-base-cased  |  lang=ru")
    logger.info("=" * 60)
    m = TransformerModel("DeepPavlov/rubert-base-cased", lang="ru")
    m.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = m._split_data()
    logger.info(f"   ✂️  Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
    m.train(X_train, y_train, X_val, y_val)
    m.evaluate_test(X_test, y_test)
    m.save_model()
    m.save_results()
    return m.results


# ============================================================
# 4. POINT D'ENTRÉE (si exécuté directement)
# ============================================================

if __name__ == "__main__":
    print("🔁 Mode direct : exécution de XLM-R")
    run_xlmr()