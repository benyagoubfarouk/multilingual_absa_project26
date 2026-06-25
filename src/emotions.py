# src/emotions.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, RESULTS_DIR, MODELS_DIR, LOGS_DIR, EMOTION_CLASSES, EMOTION_MAPPING, RANDOM_SEED, CV_FOLDS
from utils import setup_logger, set_seed, save_json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = setup_logger(__name__)


# ============================================================
# 1. DATASET PYTORCH PERSONNALISÉ
# ============================================================

class EmotionDataset(Dataset):
    """
    Dataset PyTorch personnalisé pour la détection des émotions.
    """
    
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
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# 2. CLASSE MODÈLE ÉMOTIONS
# ============================================================

class EmotionModel:
    """
    Modèle pour la détection des 5 émotions (Plutchik).
    Utilise un Transformer fine-tuné sur les données d'émotions.
    """
    
    def __init__(self, model_name="camembert-base", lang="fr"):
        self.model_name = model_name
        self.lang = lang
        self.num_labels = 5
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.results = {}
        self.models = {}
        
        self.emotion_mapping = EMOTION_MAPPING
        self.emotion_names = {v: k for k, v in self.emotion_mapping.items()}
        set_seed(RANDOM_SEED)
        logger.info(f"🔧 Initialisation du modèle Émotions : {model_name} (langue: {lang})")
    
    def load_data(self, filepath=None):
        """
        Charge les données prétraitées contenant les labels d'émotions.
        """
        if filepath is None:
            # ✅ On lit le fichier avec les émotions générées par silver labeling
            filepath = os.path.join(DATA_PROCESSED, "corpus_absa_with_emotions.csv")
        
        logger.info(f"📂 Chargement des données : {filepath}")
        self.df = pd.read_csv(filepath)
        
        if self.lang != "all":
            self.df = self.df[self.df['lang'] == self.lang]
        
        if 'emotion' not in self.df.columns:
            logger.error("❌ Colonne 'emotion' non trouvée ! Exécutez d'abord emotion_silver_labeler.py")
            self.has_enough_classes = False
            return self.df
        
        logger.info(f"   ✅ {len(self.df)} lignes chargées pour {self.lang}")
        
        unique_classes = self.df['emotion'].unique()
        logger.info(f"   Émotions présentes : {unique_classes}")
        
        if len(unique_classes) < 2:
            logger.warning(f"⚠️ La langue {self.lang} n'a qu'une seule émotion. Ignorée.")
            self.has_enough_classes = False
            return self.df
        
        self.has_enough_classes = True
        
        self.X = self.df['text_processed'].tolist()
        self.y = self.df['emotion'].tolist()
        self.y_int = [self.emotion_mapping[e] for e in self.y]
        
        logger.info(f"   Distribution : {dict(pd.Series(self.y).value_counts())}")
        
        return self.df
    
    def load_tokenizer_and_model(self):
        logger.info(f"📂 Chargement du tokenizer et du modèle : {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        logger.info(f"   ✅ Modèle chargé : {self.model_name}")
        return self.tokenizer, self.model
    
    def create_dataset(self, texts, labels):
        if self.tokenizer is None:
            self.load_tokenizer_and_model()
        return EmotionDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=128
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1_macro = f1_score(labels, predictions, average='macro')
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes. Entraînement ignoré.")
            return None
        
        if self.tokenizer is None or self.model is None:
            self.load_tokenizer_and_model()
        
        train_dataset = self.create_dataset(X_train, y_train)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(MODELS_DIR, f"emotion_{self.model_name.replace('/', '_')}"),
            eval_strategy="steps" if X_val is not None else "no",
            eval_steps=50 if X_val is not None else None,
            save_strategy="steps",
            save_steps=100,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=os.path.join(LOGS_DIR, 'emotions'),
            logging_steps=10,
            load_best_model_at_end=True if X_val is not None else False,
            metric_for_best_model="f1_macro" if X_val is not None else None,
            greater_is_better=True,
            save_total_limit=2,
            seed=RANDOM_SEED,
            report_to="none",
            disable_tqdm=False,
        )
        
        eval_dataset = None
        if X_val is not None:
            eval_dataset = self.create_dataset(X_val, y_val)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if X_val is not None else None,
        )
        
        logger.info(f"🚀 Début de l'entraînement des émotions sur {len(X_train)} échantillons...")
        self.trainer.train()
        logger.info("   ✅ Entraînement terminé")
        return self.trainer
    
    def predict(self, texts):
        if self.trainer is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        dummy_labels = [0] * len(texts)
        dataset = self.create_dataset(texts, dummy_labels)
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        return preds
    
    def cross_validate(self, X=None, y=None, n_folds=CV_FOLDS):
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes. Validation croisée ignorée.")
            return None
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        logger.info(f"🔄 Validation croisée {n_folds} plis pour les émotions...")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        fold_results = []
        all_true = []
        all_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"   📊 Pli {fold + 1}/{n_folds}")
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            
            self.tokenizer = None
            self.model = None
            self.load_tokenizer_and_model()
            self.train(X_train, y_train, X_val, y_val)
            y_pred = self.predict(X_val)
            
            f1_macro = f1_score(y_val, y_pred, average='macro')
            precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
            
            fold_results.append({
                'fold': fold + 1,
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro
            })
            all_true.extend(y_val)
            all_pred.extend(y_pred)
            self.models[f'fold_{fold + 1}'] = self.model
        
        global_f1_macro = f1_score(all_true, all_pred, average='macro')
        global_precision_macro = precision_score(all_true, all_pred, average='macro', zero_division=0)
        global_recall_macro = recall_score(all_true, all_pred, average='macro', zero_division=0)
        
        report = classification_report(
            all_true, 
            all_pred, 
            target_names=['joie', 'satisfaction', 'insatisfaction', 'colère', 'tristesse'],
            output_dict=True
        )
        cm = confusion_matrix(all_true, all_pred)
        
        self.results = {
            'fold_results': fold_results,
            'global_f1_macro': global_f1_macro,
            'global_precision_macro': global_precision_macro,
            'global_recall_macro': global_recall_macro,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(all_true),
            'model_name': f"emotion_{self.model_name}",
            'lang': self.lang
        }
        
        logger.info(f"\n📊 Résultats de la validation croisée pour les émotions:")
        logger.info(f"   Macro-F1 moyen : {np.mean([r['f1_macro'] for r in fold_results]):.4f} (±{np.std([r['f1_macro'] for r in fold_results]):.4f})")
        logger.info(f"   Macro-F1 global : {global_f1_macro:.4f}")
        
        return self.results
    
    def train_final(self, X=None, y=None):
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes. Modèle final ignoré.")
            return None
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        logger.info(f"🚀 Entraînement du modèle final sur {len(X)} échantillons...")
        self.tokenizer = None
        self.model = None
        self.load_tokenizer_and_model()
        self.train(X, y)
        logger.info("   ✅ Modèle final entraîné")
        return self.trainer
    
    def save_model(self, filepath=None):
        if filepath is None:
            model_name_clean = self.model_name.replace('/', '_')
            filepath = os.path.join(MODELS_DIR, f"emotion_{model_name_clean}_{self.lang}")
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"💾 Modèle émotions sauvegardé : {filepath}")
        return filepath
    
    def save_results(self, filepath=None):
        if filepath is None:
            model_name_clean = self.model_name.replace('/', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"emotion_results_{model_name_clean}_{self.lang}_{timestamp}.json")
        save_json(self.results, filepath)
        logger.info(f"💾 Résultats émotions sauvegardés : {filepath}")
        return filepath
    
    def plot_confusion_matrix(self, save=True):
        if 'confusion_matrix' not in self.results:
            raise ValueError("Aucune matrice de confusion disponible.")
        cm = np.array(self.results['confusion_matrix'])
        emotion_names = ['joie', 'satisfaction', 'insatisfaction', 'colère', 'tristesse']
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=emotion_names,
            yticklabels=emotion_names,
            ax=ax
        )
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vérités')
        ax.set_title(f'Matrice de confusion - Émotions ({self.lang})')
        if save:
            model_name_clean = self.model_name.replace('/', '_')
            filepath = os.path.join(RESULTS_DIR, f"emotion_confusion_matrix_{model_name_clean}_{self.lang}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Matrice de confusion émotions sauvegardée : {filepath}")
        plt.show()
        return fig


# ============================================================
# 3. FONCTIONS DE BATCH
# ============================================================

def run_emotion_models():
    logger.info("=" * 60)
    logger.info("🚀 EXÉCUTION DES MODÈLES DE DÉTECTION DES ÉMOTIONS")
    logger.info("=" * 60)
    
    results_summary = {}
    model_configs = [
        {"model_name": "camembert-base", "lang": "fr"},
        {"model_name": "xlm-roberta-base", "lang": "all"},
        {"model_name": "DeepPavlov/rubert-base-cased", "lang": "ru"},
    ]
    
    for config in model_configs:
        model_name = config["model_name"]
        lang = config["lang"]
        logger.info(f"\n{'='*50}")
        logger.info(f"📍 MODÈLE ÉMOTIONS : {model_name} ({lang})")
        logger.info(f"{'='*50}")
        model = EmotionModel(model_name=model_name, lang=lang)
        model.load_data()
        if model.has_enough_classes:
            model.cross_validate()
            model.train_final()
            model.save_model()
            model.save_results()
            model.plot_confusion_matrix()
            key = f"emotion_{model_name}_{lang}"
            results_summary[key] = {
                'model_name': model_name,
                'lang': lang,
                'f1_macro': model.results['global_f1_macro'],
                'precision_macro': model.results['global_precision_macro'],
                'recall_macro': model.results['global_recall_macro'],
                'n_samples': len(model.df)
            }
        else:
            logger.warning(f"⚠️ {model_name} ignoré car une seule émotion")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ DES PERFORMANCES DES ÉMOTIONS")
    logger.info("=" * 60)
    for key, results in results_summary.items():
        logger.info(f"\n📍 {results['model_name']} ({results['lang']}):")
        logger.info(f"   Macro-F1 : {results['f1_macro']:.4f}")
        logger.info(f"   Précision: {results['precision_macro']:.4f}")
        logger.info(f"   Rappel   : {results['recall_macro']:.4f}")
        logger.info(f"   Échantillons : {results['n_samples']}")
    return results_summary

if __name__ == "__main__":
    run_emotion_models()