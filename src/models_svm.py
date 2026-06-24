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

# Labels en anglais
POLARITY_LABELS = ['Positive', 'Neutral', 'Negative']

# ============================================================
# 1. CLASSE SVM MODEL (Version par aspect)
# ============================================================

class SVMModel:
    """
    SVM model with TF-IDF for polarity classification.
    Handles training, cross-validation and evaluation.
    """
    
    def __init__(self, lang="all", aspect=None, svm_params=None, tfidf_params=None):
        self.lang = lang
        self.aspect = aspect  # Nouveau : l'aspect à entraîner
        self.svm_params = svm_params or SVM_PARAMS.copy()
        self.tfidf_params = tfidf_params or TFIDF_PARAMS.copy()
        self.pipeline = None
        self.results = {}
        self.models = {}
        self.has_enough_classes = True
        
        set_seed(RANDOM_SEED)
        logger.info(f"🔧 Initializing SVM model for language: {lang}, aspect: {aspect}")
    
    def load_data(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(DATA_PROCESSED, "all_datasets_svm_ready.csv")
        
        logger.info(f"📂 Loading data: {filepath}")
        self.df = pd.read_csv(filepath)
        
        # Filtrer par langue
        if self.lang != "all":
            self.df = self.df[self.df['lang'] == self.lang]
        
        # ✅ NOUVEAU : Filtrer par aspect si spécifié
        if self.aspect is not None:
            self.df = self.df[self.df['aspect'] == self.aspect]
        
        logger.info(f"   ✅ {len(self.df)} rows loaded for {self.lang} - aspect: {self.aspect}")
        
        # ✅ Vérification : Si le DataFrame est vide, on lève une exception pour l'ignorer
        if len(self.df) == 0:
            logger.warning(f"⚠️ No data found for {self.lang} - aspect: {self.aspect}. Skipping...")
            self.has_enough_classes = False
            return self.df
        
        # Vérifier le nombre de classes
        unique_classes = self.df['polarity'].unique()
        logger.info(f"   Classes present: {unique_classes}")
        
        if len(unique_classes) < 2:
            logger.warning(f"⚠️ Language {self.lang} has only one class ({unique_classes[0]}). Ignored.")
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
        
        logger.info(f"   Distribution: {dict(pd.Series(self.y).value_counts())}")
        
        return self.df
    
    def create_pipeline(self):
        logger.info("🔧 Creating TF-IDF + SVM pipeline...")
        
        vectorizer = TfidfVectorizer(**self.tfidf_params)
        svm = SVC(**self.svm_params)
        
        self.pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('svm', svm)
        ])
        
        logger.info("   ✅ Pipeline created")
        return self.pipeline
    
    def train(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        if self.pipeline is None:
            self.create_pipeline()
        
        logger.info(f"🚀 Training SVM on {len(X)} samples...")
        self.pipeline.fit(X, y)
        logger.info("   ✅ Training completed")
        return self.pipeline
    
    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.pipeline.predict(X)
    
    def cross_validate(self, X=None, y=None, n_folds=CV_FOLDS):
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Not enough classes for {self.lang}-{self.aspect}. Cross-validation ignored.")
            return None
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y_int
        
        logger.info(f"🔄 Cross-validation with {n_folds} folds...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        fold_results = []
        all_true = []
        all_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"   📊 Fold {fold + 1}/{n_folds}")
            
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
            target_names=POLARITY_LABELS,
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
            'lang': self.lang,
            'aspect': self.aspect  # ✅ Nouveau : stocker l'aspect
        }
        
        logger.info(f"\n📊 Cross-validation results for {self.lang}-{self.aspect}:")
        logger.info(f"   Mean Macro-F1 : {np.mean([r['f1_macro'] for r in fold_results]):.4f} (±{np.std([r['f1_macro'] for r in fold_results]):.4f})")
        logger.info(f"   Global Macro-F1: {global_f1_macro:.4f}")
        
        return self.results
    
    def train_final(self):
        if not self.has_enough_classes:
            logger.warning(f"⚠️ Not enough classes for {self.lang}. Final model ignored.")
            return None
        
        logger.info("🚀 Training final model on all data...")
        self.create_pipeline()
        self.pipeline.fit(self.X, self.y_int)
        logger.info("   ✅ Final model trained")
        return self.pipeline
    
    def save_model(self, filepath=None):
        if filepath is None:
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            aspect_suffix = self.aspect if self.aspect else "all_aspects"
            filepath = os.path.join(MODELS_DIR, f"svm_model_{lang_suffix}_{aspect_suffix}.pkl")
        
        if self.pipeline is None:
            raise ValueError("Model not trained.")
        
        joblib.dump(self.pipeline, filepath)
        logger.info(f"💾 Model saved: {filepath}")
        return filepath
    
    def save_results(self, filepath=None):
        if filepath is None:
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            aspect_suffix = self.aspect if self.aspect else "all_aspects"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"svm_results_{lang_suffix}_{aspect_suffix}_{timestamp}.json")
        
        save_json(self.results, filepath)
        logger.info(f"💾 Results saved: {filepath}")
        return filepath
    
    def plot_confusion_matrix(self, save=True):
        if 'confusion_matrix' not in self.results:
            raise ValueError("No confusion matrix available. Run cross_validate() first.")
        
        cm = np.array(self.results['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=POLARITY_LABELS,
            yticklabels=POLARITY_LABELS,
            ax=ax
        )
        ax.set_xlabel('Predictions')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix - SVM ({self.lang}) - {self.aspect}')
        
        if save:
            lang_suffix = self.lang if self.lang != "all" else "multilingual"
            aspect_suffix = self.aspect if self.aspect else "all_aspects"
            filepath = os.path.join(RESULTS_DIR, f"svm_confusion_matrix_{lang_suffix}_{aspect_suffix}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Confusion matrix saved: {filepath}")
        
        plt.show()
        return fig


# ============================================================
# 2. FONCTION D'EXÉCUTION PAR ASPECT
# ============================================================

def run_svm_for_aspect(lang, aspect):
    """Exécute le SVM pour une langue et un aspect spécifique."""
    logger.info(f"\n{'='*50}")
    logger.info(f"📍 LANGUAGE: {lang} - ASPECT: {aspect}")
    logger.info(f"{'='*50}")
    
    model = SVMModel(lang=lang, aspect=aspect)
    model.load_data()
    
    if model.has_enough_classes:
        model.cross_validate()
        model.train_final()
        model.save_model()
        model.save_results()
        model.plot_confusion_matrix()
        
        return {
            'aspect': aspect,
            'lang': lang,
            'f1_macro': model.results['global_f1_macro'],
            'precision_macro': model.results['global_precision_macro'],
            'recall_macro': model.results['global_recall_macro'],
            'n_samples': len(model.df)
        }
    else:
        logger.warning(f"⚠️ {lang}-{aspect} ignored (only one class or no data)")
        return None


# ============================================================
# 3. BOUCLE PRINCIPALE : TOUS LES ASPECTS, TOUTES LES LANGUES
# ============================================================

def run_svm_for_all_aspects():
    """Exécute le SVM pour toutes les langues et tous les aspects."""
    logger.info("=" * 60)
    logger.info("🚀 EXECUTING SVM FOR ALL LANGUAGES AND ASPECTS")
    logger.info("=" * 60)
    
    # Définir les aspects à analyser
    aspects = ["service", "qualité", "prix", "livraison", "interface"]
    languages = ['en', 'fr', 'ru', 'all']
    
    results_summary = {}
    
    for lang in languages:
        for aspect in aspects:
            result = run_svm_for_aspect(lang, aspect)
            if result:
                key = f"{lang}_{aspect}"
                results_summary[key] = result
    
    # Afficher un résumé
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE SUMMARY (BY ASPECT)")
    logger.info("=" * 60)
    
    # Convertir en DataFrame pour un affichage plus lisible
    try:
        import pandas as pd
        summary_df = pd.DataFrame(results_summary).T
        logger.info(f"\n{summary_df.to_string()}")
    except:
        logger.info(f"\n{results_summary}")
    
    return results_summary


# ============================================================
# 4. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    results = run_svm_for_all_aspects()