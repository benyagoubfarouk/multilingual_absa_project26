# run_camembert.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models_transformers import run_transformers_for_model

if __name__ == "__main__":
    print("🚀 Lancement de CamemBERT sur le dataset réduit (5 plis)...")
    run_transformers_for_model("camembert-base", lang="fr")