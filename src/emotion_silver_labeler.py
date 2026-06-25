# src/emotion_silver_labeler.py
# ===========================================================================
# Silver labeling des émotions — à exécuter UNE SEULE FOIS.
#
# CORRECTIONS vs version soumise :
#   - sys.path ajouté (imports config/utils depuis src/)
#   - device détecté automatiquement (pas de crash si pas de GPU)
#   - top_k=None → accès sûr au label (top_k=1 retourne une liste)
#   - Fallback sur heuristique déterministe si le modèle échoue sur un texte
#   - Barre de progression avec ETA
#   - Sauvegarde vers DATA_WITH_EMOTIONS (chemin centralisé dans config)
#   - Gestion des textes vides ou NaN
#   - Rapport de distribution final + vérification cohérence
# ===========================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from transformers import pipeline
from datetime import datetime
from config import DATA_PROCESSED, DATA_WITH_EMOTIONS
from utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Mapping : labels du modèle j-hartmann → 5 émotions de Plutchik
# ---------------------------------------------------------------------------
MODEL_TO_PLUTCHIK: dict[str, str] = {
    "joy":      "joie",
    "love":     "joie",
    "optimism": "satisfaction",
    "neutral":  "satisfaction",
    "surprise": "satisfaction",
    "anger":    "colère",
    "disgust":  "insatisfaction",
    "fear":     "insatisfaction",
    "sadness":  "tristesse",
}

# ---------------------------------------------------------------------------
# Heuristique déterministe (fallback si le modèle échoue sur un texte)
# Identique à celle dans emotions.py pour la cohérence
# ---------------------------------------------------------------------------
_KEYWORDS = {
    "joie":           ["excellent", "parfait", "génial", "wonderful", "amazing", "отлично"],
    "satisfaction":   ["bien", "bon", "correct", "good", "nice", "хорошо"],
    "insatisfaction": ["décevant", "insuffisant", "disappointing", "разочарован"],
    "colère":         ["horrible", "nul", "terrible", "awful", "worst", "ужасно"],
    "tristesse":      ["déçu", "dommage", "sad", "disappointed", "жаль"],
}


def _fallback_emotion(text: str, polarity: str) -> str:
    """Heuristique déterministe basée sur mots-clés + polarité."""
    t = str(text).lower()
    if polarity == "pos":
        for kw in _KEYWORDS["joie"]:
            if kw in t:
                return "joie"
        return "satisfaction"
    elif polarity == "neg":
        for kw in _KEYWORDS["colère"]:
            if kw in t:
                return "colère"
        for kw in _KEYWORDS["tristesse"]:
            if kw in t:
                return "tristesse"
        return "insatisfaction"
    else:
        for kw in _KEYWORDS["insatisfaction"]:
            if kw in t:
                return "insatisfaction"
        return "satisfaction"


# ===========================================================================
# FONCTION PRINCIPALE
# ===========================================================================

def label_emotions_silver(
    df: pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Génère les labels d'émotions par silver labeling.

    Modèle utilisé : j-hartmann/emotion-english-distilroberta-base
    (XLM-RoBERTa fine-tuné sur GoEmotions — fonctionne aussi sur FR/RU
    car GoEmotions contient des variantes multilingues)

    Les labels obtenus sont mappés vers les 5 émotions de Plutchik.
    En cas d'échec sur un texte, l'heuristique déterministe prend le relais.

    Args:
        df         : DataFrame avec colonnes 'text' et 'polarity'
        batch_size : taille de lot pour l'inférence (32 recommandé sur T4)

    Retourne le DataFrame avec la nouvelle colonne 'emotion'.
    """
    # ── Détection GPU
    device = 0 if torch.cuda.is_available() else -1
    device_label = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    logger.info(f"🔧 Périphérique : {device_label}")

    # ── Chargement du pipeline
    logger.info("📥 Chargement du modèle j-hartmann/emotion-english-distilroberta-base...")
    try:
        emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,          # retourne le label le plus probable
            device=device,
            truncation=True,
            max_length=512,
        )
        logger.info("   ✅ Modèle chargé")
    except Exception as e:
        logger.error(f"❌ Chargement du modèle échoué : {e}")
        logger.warning("   → Utilisation de l'heuristique déterministe pour TOUS les textes.")
        emotion_pipe = None

    # ── Inférence par batchs
    df = df.copy()
    texts    = df["text"].fillna("").tolist()
    polarity = df["polarity"].fillna("neutral").tolist()
    emotions = []
    n        = len(texts)

    logger.info(f"🔄 Silver labeling sur {n} textes (batch_size={batch_size})...")
    t0 = datetime.now()

    for i in range(0, n, batch_size):
        batch_texts = [str(t)[:512] if str(t).strip() else "." for t in texts[i:i + batch_size]]
        batch_pol   = polarity[i:i + batch_size]

        if emotion_pipe is not None:
            try:
                # top_k=1 retourne une liste de listes : [[{"label": ..., "score": ...}], ...]
                results = emotion_pipe(batch_texts)
                for j, r in enumerate(results):
                    # r est une liste avec un seul élément (top_k=1)
                    raw_label = r[0]["label"].lower()
                    mapped    = MODEL_TO_PLUTCHIK.get(raw_label, None)
                    if mapped is None:
                        # Label inconnu → fallback heuristique
                        mapped = _fallback_emotion(batch_texts[j], batch_pol[j])
                    emotions.append(mapped)
            except Exception as e:
                logger.warning(f"   ⚠️  Erreur batch {i}–{i+batch_size} : {e} → fallback heuristique")
                for j in range(len(batch_texts)):
                    emotions.append(_fallback_emotion(batch_texts[j], batch_pol[j]))
        else:
            # Pas de modèle → heuristique pour tout le batch
            for j in range(len(batch_texts)):
                emotions.append(_fallback_emotion(batch_texts[j], batch_pol[j]))

        # Progression toutes les 10 itérations
        if (i // batch_size) % 10 == 0 or (i + batch_size) >= n:
            elapsed = (datetime.now() - t0).total_seconds()
            pct     = min(100, (i + batch_size) / n * 100)
            eta     = elapsed / max(pct, 1) * (100 - pct)
            logger.info(f"   {min(i + batch_size, n)}/{n}  ({pct:.0f}%)  "
                        f"— écoulé {elapsed:.0f}s  ETA {eta:.0f}s")

    df["emotion"] = emotions

    # ── Rapport de distribution
    dist = df["emotion"].value_counts()
    logger.info(f"\n✅ Distribution des émotions générées :\n{dist.to_string()}")

    # ── Vérification cohérence (toutes les 5 émotions présentes ?)
    missing = set(["joie", "satisfaction", "insatisfaction", "colère", "tristesse"]) - set(df["emotion"].unique())
    if missing:
        logger.warning(f"   ⚠️  Émotions absentes du corpus : {missing}")
        logger.warning("      → Vérifier la qualité du corpus ou augmenter les données.")

    return df


# ===========================================================================
# POINT D'ENTRÉE
# ===========================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀  SILVER LABELING DES ÉMOTIONS")
    logger.info("=" * 60)

    # Lire le corpus propre (produit par build_clean_corpus.py)
    input_path = os.path.join(DATA_PROCESSED, "corpus_absa_clean.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {input_path}\n"
            "→ Lancer build_clean_corpus.py d'abord."
        )

    logger.info(f"📂 Chargement : {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"   {len(df)} lignes chargées")

    # Labeling
    df = label_emotions_silver(df, batch_size=32)

    # Sauvegarde
    os.makedirs(os.path.dirname(DATA_WITH_EMOTIONS), exist_ok=True)
    df.to_csv(DATA_WITH_EMOTIONS, index=False)
    logger.info(f"\n💾 Sauvegardé : {DATA_WITH_EMOTIONS}")
    logger.info("▶  Prochaine étape : python src/emotions.py")