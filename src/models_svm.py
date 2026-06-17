# src/models_svm.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, RESULTS_DIR, MODELS_DIR, SVM_PARAMS, TFIDF_PARAMS, CV_FOLDS, RANDOM_SEED
from utils import setup_logger, set_seed, save_json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = setup_logger(__name__)


# ============================================================
# 1. CLASSE SVM MODEL
# ============================================================

class SVMModel:
    """
    Modèle SVM avec TF-IDF pour la classification de polarité.
    Gère l'entraînement, la validation croisée et l'évaluation.
    """
    
    def __init__(self, lang="all", svm_params=None, tfidf_params=None):
        self.lang = lang
        self.svm_params = svm_params or SVM_PARAMS.copy()
        self.tfidf_params = tfidf_params or TFIDF_PARAMS.copy()
        self.pipeline = None
        self.results = {}
        self.models = {}
        self.has_enough_classes = True
        
        set_seed(RANDOM_SEED)
        logger.info(f"🔧 Initialisation du modèle SVM pour la langue : {lang}")
    
    def load_data(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(DATA_PROCESSED, "all_datasets_svm_ready.csv")
        
        logger.info(f"📂 Chargement des données : {filepath}")
        self.df = pd.read_csv(filepath)
        
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
        
        self.X = self.df['text_processed'].tolist()
        self.y = self.df['polarity'].tolist()
        
        self.polarity_mapping = {
            'pos': 0,
            'neutral': 1,
            'neg': 2
        }
        self.polarity_names = {v: k for k, v in self.polarity_mapping.items()}
        self.y_int = [self.polarity_mapping[p] for p in self.y]
        
        logger.info(f"   Distribution : {dict(pd.Series(self.y).value_counts())}")
        
        return self.df
    
    def create_pipeline(self):
        logger.info("🔧 Création du pipeline TF-IDF + SVM...")
        
        vectorizer = TfidfVectorizer(**self.tfidf_params)
        svm = SVC(**self.svm_params)  # ← Utilisation de SVC au lieu de LinearSVC
        
        self.pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('svm', svm)
        ])
        
        logger.info("   ✅ Pipeline créé")
        return self.pipeline
    
    def train(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        if self.pipeline is None:
            self.create_pipeline()
        
        logger.info(f"🚀 Entraînement du SVM sur {len(X)} échantillons...")
        self.pipeline.fit(X, y)
        logger.info("   ✅ Entraînement terminé")
        return self.pipeline
    
    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        return self.pipeline.predict(X)
    
    def cross_validate(self, X=None, y=None, n_folds=CV_FOLDS):
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes pour {self.lang}. Validation croisée ignorée.")
            return None
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        logger.info(f"🔄 Validation croisée {n_folds} plis...")
        
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
            
            pipeline = self.create_pipeline()
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_val)
            
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
            self.models[f'fold_{fold + 1}'] = pipeline
        
        global_f1_macro = f1_score(all_true, all_pred, average='macro')
        global_precision_macro = precision_score(all_true, all_pred, average='macro', zero_division=0)
        global_recall_macro = recall_score(all_true, all_pred, average='macro', zero_division=0)
        
        report = classification_report(
            all_true, 
            all_pred, 
            target_names=['Positif', 'Neutre', 'Négatif'],
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
            'lang': self.lang
        }
        
        logger.info(f"\n📊 Résultats de la validation croisée :")
        logger.info(f"   Macro-F1 moyen : {np.mean([r['f1_macro'] for r in fold_results]):.4f} (±{np.std([r['f1_macro'] for r in fold_results]):.4f})")
        logger.info(f"   Macro-F1 global : {global_f1_macro:.4f}")
        logger.info(f"   Précision macro : {global_precision_macro:.4f}")
        logger.info(f"   Rappel macro    : {global_recall_macro:.4f}")
        
        return self.results
    
    def train_final(self):
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Pas assez de classes pour {self.lang}. Modèle final ignoré.")
            return None
        
        logger.info("🚀 Entraînement du modèle final sur l'ensemble des données...")
        self.create_pipeline()
        self.pipeline.fit(self.X, self.y_int)
        logger.info("   ✅ Modèle final entraîné")
        return self.pipeline
    
    def save_model(self, filepath=None):
        if filepath is None:
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            filepath = os.path.join(MODELS_DIR, f"svm_model_{lang_suffix}.pkl")
        
        if self.pipeline is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        
        joblib.dump(self.pipeline, filepath)
        logger.info(f"💾 Modèle sauvegardé : {filepath}")
        return filepath
    
    def save_results(self, filepath=None):
        if filepath is None:
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"svm_results_{lang_suffix}_{timestamp}.json")
        
        save_json(self.results, filepath)
        logger.info(f"💾 Résultats sauvegardés : {filepath}")
        return filepath
    
    def plot_confusion_matrix(self, save=True):
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
        ax.set_title(f'Matrice de confusion - SVM ({self.lang})')
        
        if save:
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            filepath = os.path.join(RESULTS_DIR, f"svm_confusion_matrix_{lang_suffix}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Matrice de confusion sauvegardée : {filepath}")
        
        plt.show()
        return fig


# ============================================================
# 2. FONCTIONS DE BATCH
# ============================================================

def run_svm_for_all_languages():
    """Exécute le SVM pour toutes les langues et sauvegarde les résultats."""
    logger.info("=" * 60)
    logger.info("🚀 EXÉCUTION DU SVM POUR TOUTES LES LANGUES")
    logger.info("=" * 60)
    
    results_summary = {}
    
    # Langues à tester
    languages = ['en', 'fr', 'ru', 'all']
    
    for lang in languages:
        logger.info(f"\n{'='*50}")
        logger.info(f"📍 TRAITEMENT POUR LA LANGUE : {lang}")
        logger.info(f"{'='*50}")
        
        model = SVMModel(lang=lang)
        model.load_data()
        
        if model.has_enough_classes:
            model.cross_validate()
            model.train_final()
            model.save_model()
            model.save_results()
            model.plot_confusion_matrix()
            
            results_summary[lang] = {
                'f1_macro': model.results['global_f1_macro'],
                'precision_macro': model.results['global_precision_macro'],
                'recall_macro': model.results['global_recall_macro'],
                'n_samples': len(model.df)
            }
        else:
            logger.warning(f"⚠️ {lang} ignoré car une seule classe")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ DES PERFORMANCES")
    logger.info("=" * 60)
    
    for lang, results in results_summary.items():
        logger.info(f"\n📍 {lang.upper() if lang != 'all' else 'MULTILINGUE'}:")
        logger.info(f"   Macro-F1 : {results['f1_macro']:.4f}")
        logger.info(f"   Précision: {results['precision_macro']:.4f}")
        logger.info(f"   Rappel   : {results['recall_macro']:.4f}")
        logger.info(f"   Échantillons : {results['n_samples']}")
    
    return results_summary


# ============================================================
# 3. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    results = run_svm_for_all_languages()