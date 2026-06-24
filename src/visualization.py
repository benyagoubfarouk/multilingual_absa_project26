# src/visualization.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, FIGURES_DIR
from utils import setup_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import glob

logger = setup_logger(__name__)


# ============================================================
# 1. CLASSE DE VISUALISATION
# ============================================================

class Visualizer:
    """
    Classe pour la visualisation des résultats.
    """
    
    def __init__(self):
        self.figures = {}
    
    def plot_confusion_matrix_from_file(self, json_file, save=True):
        """
        Crée une matrice de confusion à partir d'un fichier JSON.
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'confusion_matrix' not in data:
            logger.warning(f"Pas de matrice de confusion dans {json_file}")
            return None
        
        cm = np.array(data['confusion_matrix'])
        labels = data.get('target_names', ['Positif', 'Neutre', 'Négatif'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vérités')
        
        model_name = os.path.basename(json_file).replace('.json', '')
        ax.set_title(f'Matrice de confusion - {model_name}')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_name}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Matrice sauvegardée : {filepath}")
        
        plt.show()
        return fig
    
    def plot_learning_curves(self, log_file, save=True):
        """
        Trace les courbes d'apprentissage à partir d'un fichier log.
        """
        # Cette fonction nécessite des logs d'entraînement
        # Pour l'instant, affichage d'un message
        logger.info("📈 Courbes d'apprentissage - à implémenter avec les logs")
        return None


# ============================================================
# 2. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    # Créer les dossiers nécessaires
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    viz = Visualizer()
    
    # Trouver toutes les matrices de confusion
    cm_files = glob.glob(os.path.join(RESULTS_DIR, "*confusion_matrix*.png"))
    logger.info(f"📊 {len(cm_files)} matrices de confusion trouvées")
    
    # Afficher les matrices
    for f in cm_files:
        # Pour les fichiers PNG, on ne peut pas les réafficher facilement
        logger.info(f"   - {os.path.basename(f)}")
    
    logger.info(" Visualisation terminée")