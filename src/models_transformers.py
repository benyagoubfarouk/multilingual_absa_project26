# src/models_transformers.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, RESULTS_DIR, MODELS_DIR, LOGS_DIR, TRANSFORMERS_PARAMS, CV_FOLDS, RANDOM_SEED
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
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = setup_logger(__name__)

# ============================================================
# 1. DATASET PYTORCH PERSONNALISÉ
# ============================================================

class ABSADataset(Dataset):
    """
    Dataset PyTorch personnalisé pour l'ABSA.
    Gère la tokenisation dynamique et l'alignement des tenseurs.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: Liste des textes
            labels: Liste des labels (entiers)
            tokenizer: Tokenizer Hugging Face
            max_length: Longueur maximale des séquences
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenisation
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
# 2. CLASSE MODÈLE TRANSFORMER
# ============================================================

class TransformerModel:
    """
    Modèle Transformer pour la classification de polarité.
    Supporte XLM-RoBERTa, CamemBERT, RuBERT.
    """
    
    def __init__(self, model_name, lang="all", num_labels=3):
        """
        Args:
            model_name: Nom du modèle sur Hugging Face Hub
            lang: "en", "fr", "ru" ou "all"
            num_labels: Nombre de classes (3 pour la polarité)
        """
        self.model_name = model_name
        self.lang = lang
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.results = {}
        self.models = {}
        
        # Fixer la graine aléatoire
        set_seed(RANDOM_SEED)
        
        # Récupérer les paramètres du modèle
        self.params = TRANSFORMERS_PARAMS.get(model_name, {})
        if not self.params:
            # Si le modèle n'est pas dans config, utiliser des paramètres par défaut
            self.params = {
                "model_name": model_name,
                "num_labels": num_labels,
                "learning_rate": 2e-5,
                "batch_size": 16,
                "epochs": 3,
                "max_length": 128,
                "warmup_steps": 0.1,
                "weight_decay": 0.01
            }
        
        logger.info(f"🔧 Initialisation du modèle Transformer : {model_name} (langue: {lang})")
    
    def load_data(self, filepath=None):
        """
        Charge les données prétraitées pour les Transformers.
        """
        if filepath is None:
            filepath = os.path.join(DATA_PROCESSED, "all_datasets_tf_ready.csv")
        
        logger.info(f"📂 Chargement des données : {filepath}")
        self.df = pd.read_csv(filepath)
        
        # Filtrer par langue si spécifiée
        if self.lang != "all":
            self.df = self.df[self.df['lang'] == self.lang]
        
        logger.info(f"   ✅ {len(self.df)} lignes chargées pour {self.lang}")
        
        # Vérifier le nombre de classes
        unique_classes = self.df['polarity'].unique()
        logger.info(f"   Classes présentes : {unique_classes}")
        
        if len(unique_classes) < 2:
            logger.warning(f"⚠️ La langue {self.lang} n'a qu'une seule classe ({unique_classes[0]}). Ignorée.")
            self.has_enough_classes = False
            return self.df
        
        self.has_enough_classes = True
        
        # Préparer X et y
        self.X = self.df['text_processed'].tolist()
        self.y = self.df['polarity'].tolist()
        
        # Mapping des polarités
        self.polarity_mapping = {
            'pos': 0,
            'neutral': 1,
            'neg': 2
        }
        self.polarity_names = {v: k for k, v in self.polarity_mapping.items()}
        self.y_int = [self.polarity_mapping[p] for p in self.y]
        
        logger.info(f"   Distribution : {dict(pd.Series(self.y).value_counts())}")
        
        return self.df
    
    def load_tokenizer_and_model(self):
        """
        Charge le tokenizer et le modèle pré-entraîné.
        """
        logger.info(f"📂 Chargement du tokenizer et du modèle : {self.model_name}")
        
        # Chargement du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Chargement du modèle
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        logger.info(f"   ✅ Modèle chargé : {self.model_name}")
        
        return self.tokenizer, self.model
    
    def create_dataset(self, texts, labels):
        """
        Crée un dataset PyTorch.
        """
        if self.tokenizer is None:
            self.load_tokenizer_and_model()
        
        return ABSADataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.params.get('max_length', 128)
        )
    
    def compute_metrics(self, eval_pred):
        """
        Fonction de calcul des métriques pour le Trainer.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calcul du F1 macro
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
        """
        Entraîne le modèle.
        """
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes pour {self.lang}. Entraînement ignoré.")
            return None
        
        if self.tokenizer is None or self.model is None:
            self.load_tokenizer_and_model()
        
        # Création des datasets
        train_dataset = self.create_dataset(X_train, y_train)
        
        # Paramètres d'entraînement
        training_args = TrainingArguments(
            output_dir=os.path.join(MODELS_DIR, self.model_name.replace('/', '_')),
            evaluation_strategy="steps" if X_val is not None else "no",
            eval_steps=50 if X_val is not None else None,
            save_strategy="steps",
            save_steps=100,
            learning_rate=self.params.get('learning_rate', 2e-5),
            per_device_train_batch_size=self.params.get('batch_size', 16),
            per_device_eval_batch_size=self.params.get('batch_size', 16),
            num_train_epochs=self.params.get('epochs', 3),
            weight_decay=self.params.get('weight_decay', 0.01),
            warmup_ratio=self.params.get('warmup_steps', 0.1),
            logging_dir=os.path.join(LOGS_DIR, 'transformers'),
            logging_steps=10,
            load_best_model_at_end=True if X_val is not None else False,
            metric_for_best_model="f1_macro" if X_val is not None else None,
            greater_is_better=True,
            save_total_limit=2,
            seed=RANDOM_SEED,
            report_to="none",
            disable_tqdm=False,
        )
        
        # Préparer les datasets de validation
        eval_dataset = None
        if X_val is not None:
            eval_dataset = self.create_dataset(X_val, y_val)
        
        # Création du Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if X_val is not None else None,
        )
        
        # Entraînement
        logger.info(f"🚀 Début de l'entraînement sur {len(X_train)} échantillons...")
        self.trainer.train()
        logger.info("   ✅ Entraînement terminé")
        
        return self.trainer
    
    def predict(self, texts):
        """
        Fait des prédictions sur de nouveaux textes.
        """
        if self.trainer is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        
        # Création du dataset
        dummy_labels = [0] * len(texts)
        dataset = self.create_dataset(texts, dummy_labels)
        
        # Prédictions
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        
        return preds
    
    def cross_validate(self, X=None, y=None, n_folds=CV_FOLDS):
        """
        Effectue une validation croisée stratifiée.
        """
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes pour {self.lang}. Validation croisée ignorée.")
            return None
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        logger.info(f"🔄 Validation croisée {n_folds} plis pour {self.model_name}...")
        
        # Initialisation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        fold_results = []
        all_true = []
        all_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"   📊 Pli {fold + 1}/{n_folds}")
            
            # Séparation des données
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            
            # Réinitialiser le modèle
            self.tokenizer = None
            self.model = None
            self.load_tokenizer_and_model()
            
            # Entraînement
            self.train(X_train, y_train, X_val, y_val)
            
            # Prédiction
            y_pred = self.predict(X_val)
            
            # Calcul des métriques
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
        
        # Métriques globales
        global_f1_macro = f1_score(all_true, all_pred, average='macro')
        global_precision_macro = precision_score(all_true, all_pred, average='macro', zero_division=0)
        global_recall_macro = recall_score(all_true, all_pred, average='macro', zero_division=0)
        
        # Rapport de classification
        report = classification_report(
            all_true, 
            all_pred, 
            target_names=['Positif', 'Neutre', 'Négatif'],
            output_dict=True
        )
        
        cm = confusion_matrix(all_true, all_pred)
        
        # Résultats
        self.results = {
            'fold_results': fold_results,
            'global_f1_macro': global_f1_macro,
            'global_precision_macro': global_precision_macro,
            'global_recall_macro': global_recall_macro,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(all_true),
            'model_name': self.model_name,
            'lang': self.lang
        }
        
        logger.info(f"\n📊 Résultats de la validation croisée pour {self.model_name}:")
        logger.info(f"   Macro-F1 moyen : {np.mean([r['f1_macro'] for r in fold_results]):.4f} (±{np.std([r['f1_macro'] for r in fold_results]):.4f})")
        logger.info(f"   Macro-F1 global : {global_f1_macro:.4f}")
        logger.info(f"   Précision macro : {global_precision_macro:.4f}")
        logger.info(f"   Rappel macro    : {global_recall_macro:.4f}")
        
        return self.results
    
    def train_final(self, X=None, y=None):
        """
        Entraîne le modèle final sur l'ensemble des données.
        """
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes pour {self.lang}. Modèle final ignoré.")
            return None
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        logger.info(f"🚀 Entraînement du modèle final sur {len(X)} échantillons...")
        
        # Réinitialiser le modèle
        self.tokenizer = None
        self.model = None
        self.load_tokenizer_and_model()
        
        # Entraînement sur l'ensemble des données
        self.train(X, y)
        
        logger.info("   ✅ Modèle final entraîné")
        return self.trainer
    
    def save_model(self, filepath=None):
        """
        Sauvegarde le modèle final.
        """
        if filepath is None:
            model_name_clean = self.model_name.replace('/', '_')
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            filepath = os.path.join(MODELS_DIR, f"{model_name_clean}_{lang_suffix}")
        
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"💾 Modèle sauvegardé : {filepath}")
        
        return filepath
    
    def save_results(self, filepath=None):
        """
        Sauvegarde les résultats de la validation croisée.
        """
        if filepath is None:
            model_name_clean = self.model_name.replace('/', '_')
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"transformer_results_{model_name_clean}_{lang_suffix}_{timestamp}.json")
        
        save_json(self.results, filepath)
        logger.info(f"💾 Résultats sauvegardés : {filepath}")
        
        return filepath
    
    def plot_confusion_matrix(self, save=True):
        """
        Affiche et sauvegarde la matrice de confusion.
        """
        if 'confusion_matrix' not in self.results:
            raise ValueError("Aucune matrice de confusion disponible. Exécutez cross_validate() d'abord.")
        
        cm = np.array(self.results['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Positif', 'Neutre', 'Négatif'],
            yticklabels=['Positif', 'Neutre', 'Négatif'],
            ax=ax
        )
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vérités')
        ax.set_title(f'Matrice de confusion - {self.model_name} ({self.lang})')
        
        if save:
            model_name_clean = self.model_name.replace('/', '_')
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            filepath = os.path.join(RESULTS_DIR, f"transformer_confusion_matrix_{model_name_clean}_{lang_suffix}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Matrice de confusion sauvegardée : {filepath}")
        
        plt.show()
        return fig


# ============================================================
# 3. FONCTIONS DE BATCH
# ============================================================

def run_transformers_for_all_languages():
    """
    Exécute les modèles Transformers pour toutes les langues.
    """
    logger.info("=" * 60)
    logger.info("🚀 EXÉCUTION DES TRANSFORMERS POUR TOUTES LES LANGUES")
    logger.info("=" * 60)
    
    results_summary = {}
    
    # Configuration des modèles par langue
    model_configs = [
        # XLM-RoBERTa (multilingue)
        {"model_name": "xlm-roberta-base", "lang": "all"},
        # CamemBERT (français)
        {"model_name": "camembert-base", "lang": "fr"},
        # RuBERT (russe)
        {"model_name": "DeepPavlov/rubert-base-cased", "lang": "ru"},
    ]
    
    for config in model_configs:
        model_name = config["model_name"]
        lang = config["lang"]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"📍 MODÈLE : {model_name} ({lang})")
        logger.info(f"{'='*50}")
        
        # Créer et exécuter le modèle
        model = TransformerModel(model_name=model_name, lang=lang)
        model.load_data()
        
        if model.has_enough_classes:
            # Validation croisée
            model.cross_validate()
            
            # Entraînement final
            model.train_final()
            
            # Sauvegardes
            model.save_model()
            model.save_results()
            model.plot_confusion_matrix()
            
            # Stocker les résultats
            key = f"{model_name}_{lang}"
            results_summary[key] = {
                'model_name': model_name,
                'lang': lang,
                'f1_macro': model.results['global_f1_macro'],
                'precision_macro': model.results['global_precision_macro'],
                'recall_macro': model.results['global_recall_macro'],
                'n_samples': len(model.df)
            }
        else:
            logger.warning(f"⚠️ {model_name} ignoré car une seule classe")
    
    # Afficher le résumé
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ DES PERFORMANCES DES TRANSFORMERS")
    logger.info("=" * 60)
    
    for key, results in results_summary.items():
        logger.info(f"\n📍 {results['model_name']} ({results['lang']}):")
        logger.info(f"   Macro-F1 : {results['f1_macro']:.4f}")
        logger.info(f"   Précision: {results['precision_macro']:.4f}")
        logger.info(f"   Rappel   : {results['recall_macro']:.4f}")
        logger.info(f"   Échantillons : {results['n_samples']}")
    
    return results_summary


# ============================================================
# 4. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    # Exécution des Transformers
    results = run_transformers_for_all_languages()