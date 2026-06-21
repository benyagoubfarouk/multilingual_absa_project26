# run.py
import os
import sys
import subprocess
import time

# Ajouter le dossier src au path pour importer les modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import print_config
from src.utils import setup_logger

logger = setup_logger("RUNNER")

def run_script(script_name, description):
    """Exécute un script Python et gère les erreurs."""
    logger.info(f"🚀 EXÉCUTION : {description} ({script_name})")
    start_time = time.time()
    
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ ERREUR lors de l'exécution de {script_name}")
        logger.error(f"   Code retour : {result.returncode}")
        logger.error(f"   Erreur : {result.stderr}")
        return False
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Terminé en {elapsed:.2f} secondes")
    return True

def main():
    print_config()
    
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET")
    logger.info("=" * 60)
    
    # Étape 1 : Préparation des données brutes
    if not run_script("src/prepare_data.py", "Extraction et préparation des données"):
        return
    
    # Étape 2 : Prétraitement linguistique
    if not run_script("src/preprocessing.py", "Prétraitement SVM et Transformers"):
        return
    
    # Étape 3 : Augmentation des données (Classe neutre)
    if not run_script("src/augmentation.py", "Augmentation par substitution synonymique"):
        logger.warning("⚠️ Augmentation ignorée ou échouée. Utilisation des données brutes.")
    
    # Étape 4 : Entraînement SVM (Baseline)
    if not run_script("src/models_svm.py", "Entraînement SVM/TF-IDF"):
        logger.warning("⚠️ SVM ignoré. Poursuite avec Transformers.")
    
    # Étape 5 : Entraînement Transformers (Sur Colab, ce sera long)
    if not run_script("src/models_transformers.py", "Fine-tuning des Transformers"):
        logger.error("❌ Échec des Transformers. Arrêt.")
        return
    
    # Étape 6 : Évaluation et matrices
    if not run_script("src/evaluation.py", "Calcul des métriques et comparaison"):
        logger.warning("⚠️ Évaluation échouée.")
    
    # Étape 7 : Matrice ABSA
    if not run_script("src/matrix_absa.py", "Génération de la matrice Aspect-Sentiment"):
        logger.warning("⚠️ Génération de matrice échouée.")
    
    logger.info("=" * 60)
    logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS !")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()