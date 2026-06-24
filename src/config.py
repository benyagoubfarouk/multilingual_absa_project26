# src/config.py
import os
import sys

# ============================================================
# 1. CHEMINS DES DOSSIERS
# ============================================================

# Dossier racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossiers de données
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
DATA_AUGMENTED = os.path.join(DATA_DIR, "augmented")

# Dossiers des modèles et résultats
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Sous-dossiers de résultats
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MATRICES_DIR = os.path.join(RESULTS_DIR, "matrices")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# Création automatique de tous les dossiers
for d in [DATA_RAW, DATA_PROCESSED, DATA_AUGMENTED, 
          MODELS_DIR, RESULTS_DIR, LOGS_DIR,
          FIGURES_DIR, MATRICES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 2. REPRODUCTIBILITÉ
# ============================================================

RANDOM_SEED = 42

# ============================================================
# 3. LANGUES ET CLASSES
# ============================================================

LANGUAGES = ["fr", "en", "ru"]

# Classes de polarité (3 classes)
POLARITY_CLASSES = ["Négatif", "Neutre", "Positif"]
POLARITY_MAPPING = {
    "neg": 0,      # Négatif
    "neutral": 1,  # Neutre
    "pos": 2       # Positif
}
POLARITY_NAMES = {v: k for k, v in POLARITY_MAPPING.items()}

# Classes d'émotions (5 classes - modèle de Plutchik)
EMOTION_CLASSES = ["joie", "satisfaction", "insatisfaction", "colère", "tristesse"]
EMOTION_MAPPING = {
    "joie": 0,
    "satisfaction": 1,
    "insatisfaction": 2,
    "colère": 3,
    "tristesse": 4
}

# ============================================================
# 4. HYPERPARAMÈTRES SVM
# ============================================================

SVM_PARAMS = {
    "C": 1.0,
    "kernel": "linear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "max_iter": 10000
}

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 50000,
    "min_df": 2,
    "stop_words": None
}

# ============================================================
# 5. HYPERPARAMÈTRES TRANSFORMERS
# ============================================================

TRANSFORMERS_PARAMS = {
    "xlm-roberta-base": {
        "model_name": "xlm-roberta-base",
        "num_labels": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,           # On garde 3 époques (dataset petit -> apprentissage rapide)
        "max_length": 128,
        "warmup_steps": 0.1,
        "weight_decay": 0.01
    },
    "camembert-base": {
        "model_name": "camembert-base",
        "num_labels": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_length": 128,
        "warmup_steps": 0.1,
        "weight_decay": 0.01
    },
    "DeepPavlov/rubert-base-cased": {
        "model_name": "DeepPavlov/rubert-base-cased",
        "num_labels": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_length": 128,
        "warmup_steps": 0.1,
        "weight_decay": 0.01
    }
}

EMOTION_MODEL_PARAMS = {
    "model_name": "camembert-base",
    "num_labels": 5,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "max_length": 128
}

# ============================================================
# 6. VALIDATION CROISÉE
# ============================================================

CV_FOLDS = 5
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================
# 7. AUGMENTATION DES DONNÉES
# ============================================================

AUGMENTATION_PARAMS = {
    "synonym_prob": 0.3,
    "target_class": "neutral",
    "max_augment_per_sample": 3,
    "augment_factor": 2
}

# ============================================================
# 8. LOGGING
# ============================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================
# 9. FICHIERS D'ENTRÉE (AJOUT IMPORTANT)
# ============================================================

# Mapping des fichiers bruts vers les langues
RAW_FILES_MAPPING = {
    "en": ["Laptops_Train.xml", "SemEval.xml"],
    "fr": ["ABSA16FR_Restaurants_TestA.xml"],
    "ru": ["rusentiment_random_posts.csv"]
}

# ✅ NOUVEAU : Chemin vers le dataset réduit avec aspects
DATA_REDUCED_WITH_ASPECTS = os.path.join(DATA_PROCESSED, "all_datasets_reduced_with_aspects.csv")

# ============================================================
# 10. INFORMATIONS SUR LE PROJET
# ============================================================

PROJECT_NAME = "Multilingual ABSA Project"
PROJECT_VERSION = "1.0.0"
AUTHOR = "Votre Nom"
DESCRIPTION = "Analyse de tonalité multilingue (FR, EN, RU) avec ABSA et détection d'émotions"

# ============================================================
# 11. FONCTION UTILITAIRE
# ============================================================

def print_config():
    print("=" * 60)
    print(f"📁 Projet : {PROJECT_NAME}")
    print(f"📌 Version : {PROJECT_VERSION}")
    print("=" * 60)
    print(f"\n📂 Chemins :")
    print(f"   Base      : {BASE_DIR}")
    print(f"   Raw       : {DATA_RAW}")
    print(f"   Processed : {DATA_PROCESSED}")
    print(f"   Réduit    : {DATA_REDUCED_WITH_ASPECTS}")
    print(f"   Models    : {MODELS_DIR}")
    print(f"   Results   : {RESULTS_DIR}")
    print(f"   Logs      : {LOGS_DIR}")
    print(f"\n🌐 Langues : {LANGUAGES}")
    print(f"\n🏷️  Classes de polarité : {POLARITY_CLASSES}")
    print(f"🎭 Classes d'émotions : {EMOTION_CLASSES}")
    print(f"\n🔄 Validation croisée : {CV_FOLDS} folds")
    print(f"🎲 Seed : {RANDOM_SEED}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print_config()