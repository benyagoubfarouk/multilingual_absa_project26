# src/config.py
import os
import sys

# ============================================================
# 1. CHEMINS DES DOSSIERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR       = os.path.join(BASE_DIR, "data")
DATA_RAW       = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
DATA_AUGMENTED = os.path.join(DATA_DIR, "augmented")

MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
MATRICES_DIR = os.path.join(RESULTS_DIR, "matrices")
REPORTS_DIR  = os.path.join(RESULTS_DIR, "reports")

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

# ── Polarité (3 classes) ─────────────────────────────────────
# Ordre imposé par POLARITY_MAPPING : neg→0, neutral→1, pos→2
# POLARITY_CLASSES[i] doit correspondre à l'entier i
POLARITY_CLASSES = ["Négatif", "Neutre", "Positif"]
POLARITY_MAPPING = {
    "neg":     0,   # Négatif
    "neutral": 1,   # Neutre
    "pos":     2,   # Positif
}
POLARITY_NAMES = {v: k for k, v in POLARITY_MAPPING.items()}

# ── Émotions Plutchik (5 classes) ────────────────────────────
EMOTION_CLASSES = ["joie", "satisfaction", "insatisfaction", "colère", "tristesse"]
EMOTION_MAPPING = {
    "joie":           0,
    "satisfaction":   1,
    "insatisfaction": 2,
    "colère":         3,
    "tristesse":      4,
}

# ── Aspects ABSA retenus (5 catégories cibles) ───────────────
# CORRECTION : cette constante était absente — elle est maintenant
# centralisée ici et importée par models_transformers.py,
# matrix_absa.py, evaluation.py, etc.
ASPECTS = ["qualité", "service", "prix", "livraison", "interface"]

# ============================================================
# 4. HYPERPARAMÈTRES SVM  (inchangés)
# ============================================================

SVM_PARAMS = {
    "C":            1.0,
    "kernel":       "linear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "max_iter":     10000,
}

TFIDF_PARAMS = {
    "ngram_range":  (1, 2),
    "max_features": 50000,
    "min_df":       2,
    "stop_words":   None,
}

# ============================================================
# 5. HYPERPARAMÈTRES TRANSFORMERS
# ============================================================
#
# CORRECTIONS par rapport à l'ancienne version :
#   batch_size  : 16 → 8   (évite les OOM sur T4 16 Go)
#   max_length  : 128 → 96 (couvre 95 % des avis courts, +25 % rapidité)
#   warmup_steps → warmup_ratio  (le bon nom pour TrainingArguments HF)
#   fp16, gradient_checkpointing, gradient_accumulation_steps
#       → gérés directement dans models_transformers.py/_build_training_args
#         (dépendent de torch.cuda.is_available() au runtime)
# ─────────────────────────────────────────────────────────────

TRANSFORMERS_PARAMS = {
    "xlm-roberta-base": {
        "model_name":    "xlm-roberta-base",
        "num_labels":    3,
        "learning_rate": 2e-5,
        "batch_size":    8,      # CORRIGÉ : était 16
        "epochs":        3,
        "max_length":    96,     # CORRIGÉ : était 128
        "warmup_ratio":  0.1,    # CORRIGÉ : était "warmup_steps" (mauvaise clé)
        "weight_decay":  0.01,
    },
    "camembert-base": {
        "model_name":    "camembert-base",
        "num_labels":    3,
        "learning_rate": 2e-5,
        "batch_size":    8,      # CORRIGÉ
        "epochs":        3,
        "max_length":    96,     # CORRIGÉ
        "warmup_ratio":  0.1,    # CORRIGÉ
        "weight_decay":  0.01,
    },
    "DeepPavlov/rubert-base-cased": {
        "model_name":    "DeepPavlov/rubert-base-cased",
        "num_labels":    3,
        "learning_rate": 2e-5,
        "batch_size":    8,      # CORRIGÉ
        "epochs":        3,
        "max_length":    96,     # CORRIGÉ
        "warmup_ratio":  0.1,    # CORRIGÉ
        "weight_decay":  0.01,
    },
}

# Modèle d'émotions (fine-tuning sur 5 classes Plutchik)
EMOTION_MODEL_PARAMS = {
    "model_name":    "camembert-base",
    "num_labels":    5,
    "learning_rate": 2e-5,
    "batch_size":    8,          # CORRIGÉ : cohérent avec les autres
    "epochs":        3,
    "max_length":    96,         # CORRIGÉ
    "warmup_ratio":  0.1,        # CORRIGÉ
    "weight_decay":  0.01,
}

# ============================================================
# 6. SPLITS TRAIN / VAL / TEST
# ============================================================
#
# §2.4.5 du mémoire : répartition stratifiée 70 / 15 / 15
# CV_FOLDS est conservé pour le SVM uniquement (5-fold CV)

CV_FOLDS    = 5
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ============================================================
# 7. AUGMENTATION DES DONNÉES
# ============================================================

AUGMENTATION_PARAMS = {
    "synonym_prob":          0.3,
    "target_class":          "neutral",
    "max_augment_per_sample": 3,
    "augment_factor":        2,
}

# ============================================================
# 8. LOGGING
# ============================================================

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================
# 9. FICHIERS DE DONNÉES
# ============================================================

RAW_FILES_MAPPING = {
    "en": ["Laptops_Train.xml", "SemEval.xml"],
    "fr": ["ABSA16FR_Restaurants_TestA.xml"],
    "ru": ["rusentiment_random_posts.csv"],
}

# Corpus propre avec aspects mappés (produit par build_clean_corpus.py)
# C'est le fichier principal utilisé par models_transformers.py
DATA_CLEAN_CORPUS = os.path.join(DATA_PROCESSED, "corpus_absa_clean.csv")

# Corpus avec labels d'émotions (produit par emotion_silver_labeler.py)
DATA_WITH_EMOTIONS = os.path.join(DATA_PROCESSED, "corpus_absa_with_emotions.csv")

# Ancien fichier (conservé pour compatibilité, mais plus utilisé en priorité)
DATA_REDUCED_WITH_ASPECTS = os.path.join(DATA_PROCESSED, "all_datasets_reduced_with_aspects.csv")

# ============================================================
# 10. INFORMATIONS SUR LE PROJET
# ============================================================

PROJECT_NAME  = "Multilingual ABSA Project"
PROJECT_VERSION = "1.0.0"
AUTHOR        = "Votre Nom"
DESCRIPTION   = (
    "Analyse de tonalité multilingue (FR, EN, RU) "
    "avec ABSA et détection d'émotions (modèle de Plutchik)"
)

# ============================================================
# 11. FONCTION UTILITAIRE
# ============================================================

def print_config() -> None:
    print("=" * 60)
    print(f"📁 Projet  : {PROJECT_NAME}  v{PROJECT_VERSION}")
    print("=" * 60)

    print(f"\n📂 Chemins :")
    print(f"   Base            : {BASE_DIR}")
    print(f"   Raw             : {DATA_RAW}")
    print(f"   Processed       : {DATA_PROCESSED}")
    print(f"   Corpus propre   : {DATA_CLEAN_CORPUS}")
    print(f"   Avec émotions   : {DATA_WITH_EMOTIONS}")
    print(f"   Models          : {MODELS_DIR}")
    print(f"   Results         : {RESULTS_DIR}")
    print(f"   Logs            : {LOGS_DIR}")

    print(f"\n🌐 Langues            : {LANGUAGES}")
    print(f"🏷️  Classes polarité   : {POLARITY_CLASSES}")
    print(f"   Mapping           : {POLARITY_MAPPING}")
    print(f"🎭 Classes émotions   : {EMOTION_CLASSES}")
    print(f"🔍 Aspects ABSA       : {ASPECTS}")

    print(f"\n🔄 CV folds (SVM)     : {CV_FOLDS}")
    print(f"✂️  Split train/val/test : {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    print(f"🎲 Seed               : {RANDOM_SEED}")

    print(f"\n⚙️  Transformers (batch / max_len) :")
    for name, p in TRANSFORMERS_PARAMS.items():
        print(f"   {name:<35} batch={p['batch_size']}  max_len={p['max_length']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config()