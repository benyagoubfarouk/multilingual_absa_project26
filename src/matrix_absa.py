# src/matrix_absa.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR, MATRICES_DIR, FIGURES_DIR
from utils import setup_logger

logger = setup_logger(__name__)

# ============================================================
# 1. TABLEAU GLOBAL (SANS ASPECTS)
# ============================================================

def generate_global_matrix():
    """Génère la matrice globale et sauvegarde une image PNG."""
    logger.info("=" * 60)
    logger.info("📊 MATRICE GLOBALE (SANS ASPECTS)")
    logger.info("=" * 60)
    
    results_dir = RESULTS_DIR
    global_files = [f for f in os.listdir(results_dir) if f.startswith("svm_results_") and "_service" not in f and "_qualité" not in f and "_prix" not in f and "_livraison" not in f and "_interface" not in f and f.endswith(".json")]
    
    valid_langs = ['en', 'fr', 'ru', 'all']
    global_files = [f for f in global_files if any(f"svm_results_{lang}" in f for lang in valid_langs)]
    
    all_data = []
    
    for file in global_files:
        filepath = os.path.join(results_dir, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lang = file.split("_")[2]
        if lang not in valid_langs:
            continue
        
        all_data.append({
            'Langue': lang.upper(),
            'F1': data.get('global_f1_macro', 0.0),
            'Précision': data.get('global_precision_macro', 0.0),
            'Rappel': data.get('global_recall_macro', 0.0),
            'Échantillons': data.get('n_samples', 0)
        })
    
    if not all_data:
        logger.warning("⚠️ Aucun fichier global trouvé.")
        return None
    
    df = pd.DataFrame(all_data)
    
    # Sauvegarder en CSV
    os.makedirs(MATRICES_DIR, exist_ok=True)
    csv_file = os.path.join(MATRICES_DIR, "matrix_global_svm.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"💾 Matrice globale sauvegardée : {csv_file}")
    
    # ---- Génération de l'image PNG ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Préparer les données pour l'affichage
    table_data = df[['Langue', 'F1', 'Précision', 'Rappel', 'Échantillons']].values
    columns = ['Langue', 'F1', 'Précision', 'Rappel', 'Échantillons']
    
    # Créer le tableau
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Matrice globale SVM (par langue)', fontsize=14, pad=20)
    
    # Sauvegarder l'image
    os.makedirs(FIGURES_DIR, exist_ok=True)
    img_file = os.path.join(FIGURES_DIR, "matrix_global_svm.png")
    plt.savefig(img_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"💾 Image globale sauvegardée : {img_file}")
    
    print("\n" + "=" * 60)
    print("📊 MATRICE GLOBALE (SVM - PAR LANGUE)")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n✅ Image sauvegardée dans : {img_file}")
    
    return df

# ============================================================
# 2. TABLEAU PAR ASPECT
# ============================================================

def generate_aspect_matrix():
    """Génère la matrice par aspect et sauvegarde une image PNG."""
    logger.info("=" * 60)
    logger.info("📊 MATRICE PAR ASPECT (SVM)")
    logger.info("=" * 60)
    
    aspects = ["service", "qualité", "prix", "livraison", "interface"]
    languages = ['en', 'fr', 'ru', 'all']
    
    results_dir = RESULTS_DIR
    all_data = []
    
    for lang in languages:
        for aspect in aspects:
            pattern = f"svm_results_{lang}_{aspect}_"
            files = [f for f in os.listdir(results_dir) if f.startswith(pattern) and f.endswith(".json")]
            if not files:
                continue
            latest_file = sorted(files)[-1]
            filepath = os.path.join(results_dir, latest_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_data.append({
                'Langue': lang.upper(),
                'Aspect': aspect.capitalize(),
                'F1': data.get('global_f1_macro', 0.0),
                'Précision': data.get('global_precision_macro', 0.0),
                'Rappel': data.get('global_recall_macro', 0.0),
                'Échantillons': data.get('n_samples', 0)
            })
    
    if not all_data:
        logger.warning("⚠️ Aucun fichier par aspect trouvé.")
        return None
    
    df = pd.DataFrame(all_data)
    matrix_f1 = df.pivot(index='Aspect', columns='Langue', values='F1').round(4)
    matrix_samples = df.pivot(index='Aspect', columns='Langue', values='Échantillons').fillna(0).astype(int)
    
    # Sauvegarder en CSV
    os.makedirs(MATRICES_DIR, exist_ok=True)
    csv_file = os.path.join(MATRICES_DIR, "matrix_aspect_svm_f1.csv")
    matrix_f1.to_csv(csv_file)
    logger.info(f"💾 Matrice F1 par aspect sauvegardée : {csv_file}")
    
    # ---- Génération de l'image PNG (Heatmap) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix_f1, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar_kws={'label': 'F1-Score'})
    ax.set_title('Matrice ABSA SVM - F1-Score par aspect et par langue', fontsize=14, pad=20)
    ax.set_xlabel('Langue', fontsize=12)
    ax.set_ylabel('Aspect', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    img_file = os.path.join(FIGURES_DIR, "matrix_aspect_svm_heatmap.png")
    plt.savefig(img_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"💾 Heatmap par aspect sauvegardée : {img_file}")
    
    print("\n" + "=" * 60)
    print("📊 MATRICE PAR ASPECT (F1-Score)")
    print("=" * 60)
    print(matrix_f1)
    print(f"\n✅ Heatmap sauvegardée dans : {img_file}")
    
    return matrix_f1

# ============================================================
# 3. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔍 GÉNÉRATION DES MATRICES SVM AVEC IMAGES")
    print("=" * 60)
    generate_global_matrix()
    generate_aspect_matrix()
    print("\n" + "=" * 60)
    print("✅ Toutes les matrices et images ont été générées !")
    print("=" * 60)