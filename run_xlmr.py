# run_xlmr.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from models_transformers import run_transformers_for_model

if __name__ == "__main__":
    print("🚀 Lancement de XLM-RoBERTa sur le dataset réduit (5 plis)...")
    run_transformers_for_model("xlm-roberta-base", lang="all")