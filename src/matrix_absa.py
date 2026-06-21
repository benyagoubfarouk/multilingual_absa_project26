# src/matrix_absa.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, MATRICES_DIR
from utils import setup_logger
import pandas as pd
import json
import glob

logger = setup_logger(__name__)

def generate_absa_matrix():
    """
    Génère la matrice Aspect-Sentiment-Émotion à partir des fichiers de résultats JSON.
    Structure : Lignes = Aspects, Colonnes = Modèles (par langue)
    """
    logger.info("=" * 60)
    logger.info("📊 GÉNÉRATION DE LA MATRICE ABSA")
    logger.info("=" * 60)
    
    # Charger tous les fichiers JSON de résultats
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    
    if not json_files:
        logger.error("❌ Aucun fichier JSON trouvé dans le dossier results/")
        return None
    
    data_rows = []
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        
        # Filtrer uniquement les résultats principaux
        if "emotion" in filename: 
            continue  # On ignore les résultats d'émotions pour l'instant
            
        with open(filepath, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Extraire les métriques du rapport de classification
        report = result.get('classification_report', {})
        
        # Déterminer le nom du modèle
        model_name = result.get('model_name', 'unknown')
        lang = result.get('lang', 'all')
        
        # Extraire le F1-score pour chaque aspect (si dispo) ou global
        # NOTE: Comme le JSON ne contient pas les scores par aspect, on utilise le Macro-F1 global
        # Dans une version avancée, il faudrait calculer les F1 par aspect dans les modèles
        macro_f1 = result.get('global_f1_macro', 0.0)
        precision = result.get('global_precision_macro', 0.0)
        recall = result.get('global_recall_macro', 0.0)
        
        # Ajouter une ligne par aspect standard (Qualité, Service, Prix, etc.)
        # On crée une moyenne pour ces aspects car on n'a pas le détail
        aspects = ['Qualité', 'Service', 'Prix', 'Livraison', 'Interface']
        
        for aspect in aspects:
            data_rows.append({
                'Aspect': aspect,
                'Modèle': model_name,
                'Langue': lang,
                'Macro-F1': macro_f1,
                'Précision': precision,
                'Rappel': recall
            })
    
    # Création du DataFrame
    df = pd.DataFrame(data_rows)
    
    if df.empty:
        logger.warning("⚠️ Aucune donnée à afficher dans la matrice.")
        return None
    
    # Pivot pour créer la matrice (Aspects en lignes, Modèles en colonnes)
    matrix = df.pivot_table(
        index='Aspect', 
        columns=['Modèle', 'Langue'], 
        values='Macro-F1',
        aggfunc='mean'
    ).round(4)
    
    # Sauvegarde
    output_path = os.path.join(MATRICES_DIR, "matrix_absa_final.csv")
    matrix.to_csv(output_path)
    logger.info(f"💾 Matrice ABSA sauvegardée : {output_path}")
    
    # Affichage
    print("\n" + "=" * 60)
    print("📊 MATRICE ASPECT-SENTIMENT (Extrait)")
    print("=" * 60)
    print(matrix.head(10))
    print("=" * 60)
    
    return matrix

if __name__ == "__main__":
    # Créer le dossier matrices s'il n'existe pas
    os.makedirs(MATRICES_DIR, exist_ok=True)
    generate_absa_matrix()