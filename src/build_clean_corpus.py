# src/build_clean_corpus.py
# ===========================================================================
# Construit le corpus ABSA propre à partir des données brutes combinées.
#
# CORRECTIONS vs version précédente :
#   - Aspect mapper intégré directement (plus besoin d'aspect_mapper.py séparé)
#   - sys.path ajouté pour les imports depuis src/
#   - Validation des colonnes et rapport détaillé
#   - Gestion explicite du cas russe (RuSentiment sans annotations ABSA)
#   - Texte original préservé + colonne absa_input pré-calculée
# ===========================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import DATA_PROCESSED, RANDOM_SEED, ASPECTS
from utils import setup_logger

logger = setup_logger(__name__)


# ===========================================================================
# 1. TABLE DE MAPPING DES ASPECTS
# ===========================================================================
# Mappe les termes bruts des datasets (SemEval, RuSentiment, SemEval-FR)
# vers les 5 catégories cibles définies dans le projet.
#
# SemEval-2014 (EN) : "battery life", "screen", "staff", "price", ...
# SemEval-2016 (FR) : "service", "plats", "cadre", "prix", ...
# RuSentiment (RU)  : "general" uniquement → NON mappé → filtré

ASPECT_MAP: dict[str, str] = {
    # ── qualité ───────────────────────────────────────────────
    "quality":        "qualité",
    "food":           "qualité",
    "meal":           "qualité",
    "taste":          "qualité",
    "plats":          "qualité",
    "cuisine":        "qualité",
    "repas":          "qualité",
    "produits":       "qualité",
    "nourriture":     "qualité",
    "battery life":   "qualité",
    "battery":        "qualité",
    "screen":         "qualité",
    "display":        "qualité",
    "build quality":  "qualité",
    "hardware":       "qualité",
    "performance":    "qualité",
    "product":        "qualité",
    "camera":         "qualité",
    "sound":          "qualité",
    # ── service ───────────────────────────────────────────────
    "service":        "service",
    "staff":          "service",
    "waiter":         "service",
    "waitress":       "service",
    "accueil":        "service",
    "personnel":      "service",
    "support":        "service",
    "customer service": "service",
    "management":     "service",
    "server":         "service",
    # ── prix ──────────────────────────────────────────────────
    "price":          "prix",
    "cost":           "prix",
    "value":          "prix",
    "prix":           "prix",
    "tarif":          "prix",
    "rapport qualité/prix": "prix",
    "rapport qualité-prix": "prix",
    "prices":         "prix",
    # ── livraison ─────────────────────────────────────────────
    "delivery":       "livraison",
    "shipping":       "livraison",
    "livraison":      "livraison",
    "delai":          "livraison",
    "packaging":      "livraison",
    "emballage":      "livraison",
    # ── interface ─────────────────────────────────────────────
    "interface":      "interface",
    "software":       "interface",
    "app":            "interface",
    "features":       "interface",
    "usability":      "interface",
    "design":         "interface",
    "menu":           "interface",
    "carte":          "interface",
    "os":             "interface",
    "keyboard":       "interface",
    "touchpad":       "interface",
}


# ===========================================================================
# 2. FONCTION DE MAPPING
# ===========================================================================

def map_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mappe la colonne 'aspect' brute vers les 5 catégories cibles.

    Les aspects non reconnus (ex. "general", "atmosphere", "ambiance")
    sont supprimés car ils ne correspondent à aucune catégorie ABSA définie.

    Retourne le DataFrame filtré avec uniquement les aspects mappés.
    """
    if "aspect" not in df.columns:
        logger.warning("Colonne 'aspect' absente — mapping ignoré.")
        return df

    def _normalize(raw: str) -> str | None:
        if pd.isna(raw):
            return None
        return ASPECT_MAP.get(str(raw).lower().strip(), None)

    df = df.copy()
    df["aspect"] = df["aspect"].apply(_normalize)

    n_before = len(df)
    df = df.dropna(subset=["aspect"]).reset_index(drop=True)
    n_after  = len(df)

    logger.info(f"   Mapping aspects : {n_before} → {n_after} lignes "
                f"({n_before - n_after} supprimées — aspects non reconnus)")

    return df


# ===========================================================================
# 3. CONSTRUCTION DU CORPUS PROPRE
# ===========================================================================

def build_clean_corpus(
    input_file:  str = None,
    output_file: str = None,
    target_per_lang: int = 800,
) -> pd.DataFrame:
    """
    Pipeline complet de nettoyage et d'équilibrage du corpus ABSA.

    Étapes :
      1. Chargement du corpus combiné brut
      2. Mapping des aspects bruts → 5 catégories cibles
      3. Rapport sur les données russes (RuSentiment non annoté ABSA)
      4. Équilibrage : 800 lignes max/langue, répartition 40/40/20
      5. Construction de la colonne absa_input ("[aspect] : [texte]")
      6. Sauvegarde en CSV

    Args:
        input_file      : chemin vers all_datasets_combined.csv
        output_file     : chemin de sortie (corpus_absa_clean.csv)
        target_per_lang : nombre cible de lignes par langue (défaut : 800)

    Retourne le DataFrame final équilibré.
    """
    # ── Chemins par défaut
    if input_file is None:
        input_file = os.path.join(DATA_PROCESSED, "all_datasets_combined.csv")
    if output_file is None:
        output_file = os.path.join(DATA_PROCESSED, "corpus_absa_clean.csv")

    # ── 1. Chargement
    logger.info("=" * 60)
    logger.info("📂 CONSTRUCTION DU CORPUS ABSA PROPRE")
    logger.info("=" * 60)
    logger.info(f"   Source : {os.path.basename(input_file)}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Fichier source introuvable : {input_file}")

    df = pd.read_csv(input_file)
    logger.info(f"   Lignes brutes : {len(df)}")
    logger.info(f"   Colonnes      : {df.columns.tolist()}")

    # ── Vérifications minimales
    for col in ("text", "polarity", "lang"):
        if col not in df.columns:
            raise ValueError(f"Colonne obligatoire absente : '{col}'")

    # ── 2. Mapping des aspects
    logger.info("\n📌 Mapping des aspects...")
    df = map_aspects(df)

    # ── 3. Rapport sur les données russes
    df_ru = df[df["lang"] == "ru"]
    if len(df_ru) < 100:
        logger.warning(
            f"\n⚠️  ATTENTION — Données russes insuffisantes après mapping :\n"
            f"   {len(df_ru)} lignes seulement (RuSentiment ne contient pas "
            f"d'annotations ABSA au niveau de l'aspect).\n"
            f"   → La comparaison principale portera sur FR + EN.\n"
            f"   → À mentionner comme LIMITE dans le §2.2.2 du mémoire."
        )
    else:
        logger.info(f"   Données RU après mapping : {len(df_ru)} lignes")

    # ── Distribution avant équilibrage
    logger.info("\n📊 Distribution avant équilibrage :")
    logger.info(df.groupby(["lang", "polarity"]).size().unstack(fill_value=0).to_string())

    # ── 4. Équilibrage stratifié 40/40/20
    logger.info(f"\n⚖️  Équilibrage : {target_per_lang} lignes max/langue "
                f"(pos 40 % / neg 40 % / neutral 20 %)...")

    RATIOS = {"pos": 0.40, "neg": 0.40, "neutral": 0.20}
    balanced_parts = []

    for lang in df["lang"].unique():
        df_lang = df[df["lang"] == lang]
        lang_parts = []

        for polarity, ratio in RATIOS.items():
            df_pol = df_lang[df_lang["polarity"] == polarity]
            n_target = int(target_per_lang * ratio)
            n_actual = min(n_target, len(df_pol))

            if n_actual < n_target:
                logger.warning(
                    f"   ⚠️  lang={lang} polarity={polarity} : "
                    f"seulement {n_actual}/{n_target} exemples disponibles."
                )

            if n_actual > 0:
                sample = df_pol.sample(n=n_actual, random_state=RANDOM_SEED)
                lang_parts.append(sample)

        if lang_parts:
            balanced_parts.append(pd.concat(lang_parts))

    if not balanced_parts:
        raise RuntimeError(
            "Corpus vide après mapping et équilibrage.\n"
            "Vérifiez que build_clean_corpus.py peut trouver "
            "all_datasets_combined.csv avec les bons aspects."
        )

    df_final = pd.concat(balanced_parts).reset_index(drop=True)

    # ── 5. Colonne absa_input (format "[aspect] : [texte]")
    # Pré-calculée ici pour cohérence avec models_transformers.py
    df_final["absa_input"] = (
        df_final["aspect"].str.lower().str.strip()
        + " : "
        + df_final["text"].fillna("")
    )

    # ── 6. Rapport final
    logger.info("\n✅ Corpus final :")
    report = df_final.groupby(["lang", "polarity"]).size().unstack(fill_value=0)
    logger.info(report.to_string())
    logger.info(f"\n   Total lignes : {len(df_final)}")
    logger.info(f"   Distribution aspects : {dict(df_final['aspect'].value_counts())}")

    # ── 7. Sauvegarde
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    logger.info(f"\n💾 Sauvegardé : {output_file}")

    return df_final


# ===========================================================================
# 4. POINT D'ENTRÉE
# ===========================================================================

if __name__ == "__main__":
    df = build_clean_corpus()

    print("\n" + "=" * 60)
    print("✅ CORPUS ABSA PROPRE GÉNÉRÉ")
    print("=" * 60)
    print(f"Lignes totales : {len(df)}")
    print("\nDistribution par langue et polarité :")
    print(df.groupby(["lang", "polarity"]).size().unstack(fill_value=0))
    print("\nDistribution par aspect :")
    print(df["aspect"].value_counts())
    print(f"\nColonnes disponibles : {df.columns.tolist()}")
    print("\n▶  Prochaine étape : python src/emotion_silver_labeler.py")