import sys
import os

# ✅ Ajout du chemin ABSOLU vers le dossier src/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ✅ Maintenant, l'éditeur et Python trouveront le module sans erreur
from src.models_transformers import run_xlmr

if __name__ == "__main__":
    print("🚀 Lancement de XLM-RoBERTa (multilingue)...")
    run_xlmr()