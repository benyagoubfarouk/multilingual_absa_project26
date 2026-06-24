# src/evaluation.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR
from utils import setup_logger, save_json
import pandas as pd
import numpy as np
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
import json
import glob

logger = setup_logger(__name__)


# ============================================================
# 1. CLASSE D'ÉVALUATION
# ============================================================

class Evaluator:
    """
    Classe pour l'évaluation et la comparaison des modèles.
    """
    
    def __init__(self):
        self.results = {}
        self.comparison_df = None
    
    def load_results(self, filepath):
        """
        Charge un fichier de résultats JSON.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_all_results(self, results_dir=None):
        """
        Charge tous les fichiers de résultats du dossier results/.
        """
        if results_dir is None:
            results_dir = RESULTS_DIR
        
        all_results = {}
        
        # Charger les résultats SVM
        svm_files = glob.glob(os.path.join(results_dir, "svm_results_*.json"))
        for f in svm_files:
            name = os.path.basename(f).replace('svm_results_', '').replace('.json', '')
            all_results[f"svm_{name}"] = self.load_results(f)
        
        # Charger les résultats Transformers
        transformer_files = glob.glob(os.path.join(results_dir, "transformer_results_*.json"))
        for f in transformer_files:
            name = os.path.basename(f).replace('transformer_results_', '').replace('.json', '')
            all_results[f"transformer_{name}"] = self.load_results(f)
        
        # Charger les résultats Émotions
        emotion_files = glob.glob(os.path.join(results_dir, "emotion_results_*.json"))
        for f in emotion_files:
            name = os.path.basename(f).replace('emotion_results_', '').replace('.json', '')
            all_results[f"emotion_{name}"] = self.load_results(f)
        
        logger.info(f"📂 {len(all_results)} fichiers de résultats chargés")
        
        self.results = all_results
        return all_results
    
    def create_comparison_table(self):
        """
        Crée un tableau de comparaison des performances.
        """
        if not self.results:
            self.load_all_results()
        
        data = []
        
        for name, result in self.results.items():
            # Extraire les métriques principales
            row = {
                'modèle': name,
                'langue': result.get('lang', 'N/A'),
                'macro_f1': result.get('global_f1_macro', 0),
                'precision': result.get('global_precision_macro', 0),
                'rappel': result.get('global_recall_macro', 0),
                'échantillons': result.get('n_samples', 0),
                'model_name': result.get('model_name', '')
            }
            data.append(row)
        
        self.comparison_df = pd.DataFrame(data)
        self.comparison_df = self.comparison_df.sort_values('macro_f1', ascending=False)
        
        logger.info("📊 Tableau de comparaison généré")
        
        return self.comparison_df
    
    def print_comparison(self):
        """
        Affiche le tableau de comparaison.
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        print("\n" + "=" * 80)
        print("📊 COMPARAISON DES MODÈLES")
        print("=" * 80)
        
        # Formatage du tableau
        display_df = self.comparison_df.copy()
        display_df['macro_f1'] = display_df['macro_f1'].apply(lambda x: f"{x:.4f}")
        display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.4f}")
        display_df['rappel'] = display_df['rappel'].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        print("=" * 80)
        
        return self.comparison_df
    
    def plot_comparison(self, save=True):
        """
        Crée un graphique de comparaison des modèles.
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Préparer les données
        models = self.comparison_df['modèle'].tolist()
        f1_scores = self.comparison_df['macro_f1'].tolist()
        
        # Couleurs selon le type de modèle
        colors = []
        for m in models:
            if m.startswith('svm'):
                colors.append('#3498db')  # Bleu
            elif m.startswith('transformer'):
                colors.append('#2ecc71')  # Vert
            elif m.startswith('emotion'):
                colors.append('#e74c3c')  # Rouge
            else:
                colors.append('#95a5a6')  # Gris
        
        # Barres horizontales
        bars = ax.barh(models, f1_scores, color=colors, alpha=0.8)
        
        # Ajouter les valeurs sur les barres
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=10)
        
        ax.set_xlabel('Macro-F1')
        ax.set_title('Comparaison des performances des modèles')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='SVM'),
            Patch(facecolor='#2ecc71', label='Transformers'),
            Patch(facecolor='#e74c3c', label='Émotions')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"model_comparison_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Graphique sauvegardé : {filepath}")
        
        plt.show()
        return fig
    
    def plot_comparison_by_lang(self, save=True):
        """
        Crée un graphique de comparaison par langue.
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        # Filtrer les modèles ayant une langue
        df = self.comparison_df[self.comparison_df['langue'] != 'N/A']
        
        if df.empty:
            logger.warning("Pas de données par langue")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Groupement par langue
        languages = df['langue'].unique()
        x = np.arange(len(languages))
        width = 0.25
        
        for i, model_type in enumerate(['svm', 'transformer']):
            model_df = df[df['modèle'].str.startswith(model_type)]
            scores = []
            for lang in languages:
                lang_df = model_df[model_df['langue'] == lang]
                if not lang_df.empty:
                    scores.append(lang_df['macro_f1'].iloc[0])
                else:
                    scores.append(0)
            
            ax.bar(x + i*width, scores, width, label=model_type.capitalize())
        
        ax.set_xlabel('Langue')
        ax.set_ylabel('Macro-F1')
        ax.set_title('Comparaison des performances par langue')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(languages)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"comparison_by_lang_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Graphique sauvegardé : {filepath}")
        
        plt.show()
        return fig
    
    def generate_report(self, save=True):
        """
        Génère un rapport complet en texte.
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        report = []
        report.append("=" * 80)
        report.append(" RAPPORT D'ÉVALUATION DES MODÈLES")
        report.append("=" * 80)
        report.append(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Résumé des performances
        report.append(" RÉSUMÉ DES PERFORMANCES")
        report.append("-" * 40)
        
        best_model = self.comparison_df.iloc[0]
        report.append(f"🏆 Meilleur modèle : {best_model['modèle']}")
        report.append(f"   Macro-F1 : {best_model['macro_f1']:.4f}")
        report.append(f"   Précision : {best_model['precision']:.4f}")
        report.append(f"   Rappel    : {best_model['rappel']:.4f}")
        report.append("")
        
        # Tableau complet
        report.append("📊 TABLEAU COMPLET")
        report.append("-" * 40)
        display_df = self.comparison_df.copy()
        display_df['macro_f1'] = display_df['macro_f1'].apply(lambda x: f"{x:.4f}")
        report.append(display_df.to_string(index=False))
        report.append("")
        
        # Analyse par type de modèle
        report.append(" ANALYSE PAR TYPE DE MODÈLE")
        report.append("-" * 40)
        
        for model_type in ['svm', 'transformer', 'emotion']:
            type_df = self.comparison_df[self.comparison_df['modèle'].str.startswith(model_type)]
            if not type_df.empty:
                avg_f1 = type_df['macro_f1'].mean()
                report.append(f"🔹 {model_type.upper()} :")
                report.append(f"   Moyenne Macro-F1 : {avg_f1:.4f}")
                report.append(f"   Meilleur : {type_df.iloc[0]['modèle']} ({type_df.iloc[0]['macro_f1']:.4f})")
                report.append("")
        
        report.append("=" * 80)
        report.append(" FIN DU RAPPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"evaluation_report_{timestamp}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f" Rapport sauvegardé : {filepath}")
        
        print(report_text)
        return report_text


# ============================================================
# 2. FONCTION DE BATCH
# ============================================================

def run_evaluation():
    """
    Exécute l'évaluation complète de tous les modèles.
    """
    logger.info("=" * 60)
    logger.info(" EXÉCUTION DE L'ÉVALUATION")
    logger.info("=" * 60)
    
    evaluator = Evaluator()
    evaluator.load_all_results()
    evaluator.create_comparison_table()
    evaluator.print_comparison()
    evaluator.plot_comparison()
    evaluator.plot_comparison_by_lang()
    evaluator.generate_report()
    
    logger.info(" Évaluation terminée")


# ============================================================
# 3. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    run_evaluation()