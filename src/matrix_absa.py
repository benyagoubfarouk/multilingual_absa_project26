# src/matrix_absa.py
# ===========================================================================
# Génère la matrice ABSA comparative — principal livrable analytique du projet.
#
# CORRECTIONS vs ancienne version :
#   - Ancienne version : lit uniquement les résultats SVM
#   - Nouvelle version : lit SVM + XLM-RoBERTa + CamemBERT + RuBERT
#   - Génère la matrice aspect × modèle × langue (§2.4.6 du mémoire)
#   - Calcule le gain = Macro-F1(transformer) − Macro-F1(SVM baseline)
#   - Normalise les deux formats JSON (SVM et Transformer)
#   - Produit 3 heatmaps : F1 global, F1 par aspect, gains vs SVM
#   - Sauvegarde CSV + PNG dans results/matrices/ et results/figures/
#
# FORMATS JSON attendus :
#   SVM        : {"global_f1_macro": 0.65, "lang": "fr", "aspect": "service", ...}
#   Transformer: {"model_name": "...", "lang": "fr",
#                 "global": {"f1_macro": 0.72, ...},
#                 "per_aspect": {"qualité": 0.70, ...}}
# ===========================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, MATRICES_DIR, FIGURES_DIR, ASPECTS, LANGUAGES
from utils import setup_logger, save_json

logger = setup_logger(__name__)

# Noms courts des modèles pour les colonnes de la matrice
MODEL_SHORT = {
    "xlm-roberta-base":             "XLM-R",
    "camembert-base":               "CamemBERT",
    "DeepPavlov/rubert-base-cased": "RuBERT",
    "SVM_TF-IDF":                   "SVM",
    "SVM":                          "SVM",
}
LANG_LABEL = {"fr": "FR", "en": "EN", "ru": "RU", "all": "ALL"}


# ===========================================================================
# 1. CHARGEMENT ET NORMALISATION DES RÉSULTATS
# ===========================================================================

def _load_svm_results(results_dir: str) -> list[dict]:
    """
    Charge tous les fichiers JSON du SVM.
    Format attendu : svm_results_{lang}_{aspect}_{ts}.json

    Retourne une liste de dicts normalisés :
        {"model": "SVM", "lang": str, "aspect": str|None, "f1_macro": float}
    """
    records = []
    pattern = os.path.join(results_dir, "svm_results_*.json")
    for fpath in glob.glob(pattern):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            records.append({
                "model":    "SVM",
                "lang":     d.get("lang", "?"),
                "aspect":   d.get("aspect", None),
                "f1_macro": round(float(d.get("global_f1_macro", 0.0)), 4),
                "precision_macro": round(float(d.get("global_precision_macro", 0.0)), 4),
                "recall_macro":    round(float(d.get("global_recall_macro", 0.0)), 4),
                "n_samples": d.get("n_samples", 0),
            })
        except Exception as e:
            logger.warning(f"   ⚠️  SVM JSON ignoré ({os.path.basename(fpath)}) : {e}")
    logger.info(f"   SVM : {len(records)} fichiers chargés")
    return records


def _load_transformer_results(results_dir: str) -> list[dict]:
    """
    Charge tous les fichiers JSON des Transformers.
    Format attendu : transformer_{model}_{lang}_{ts}.json

    Retourne une liste de dicts normalisés, un par couple (modèle, lang),
    avec les F1 globaux ET par aspect.
    """
    records = []
    pattern = os.path.join(results_dir, "transformer_*.json")
    for fpath in glob.glob(pattern):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)

            # Certains fichiers peuvent avoir l'ancien format (avant correction)
            # Gérer les deux structures
            if "global" in d and isinstance(d["global"], dict):
                # Nouveau format (models_transformers.py corrigé)
                global_metrics = d["global"]
                f1  = global_metrics.get("f1_macro", 0.0)
                pre = global_metrics.get("precision_macro", 0.0)
                rec = global_metrics.get("recall_macro", 0.0)
                n   = global_metrics.get("n_test", 0)
            else:
                # Ancien format (clés à la racine)
                f1  = d.get("global_f1_macro", 0.0)
                pre = d.get("global_precision_macro", 0.0)
                rec = d.get("global_recall_macro", 0.0)
                n   = d.get("n_samples", 0)

            per_aspect = d.get("per_aspect", {})
            model_name = d.get("model_name", "unknown")
            lang       = d.get("lang", "?")

            # Entrée globale (sans aspect)
            records.append({
                "model":         model_name,
                "model_short":   MODEL_SHORT.get(model_name, model_name[:8]),
                "lang":          lang,
                "aspect":        None,
                "f1_macro":      round(float(f1), 4),
                "precision_macro": round(float(pre), 4),
                "recall_macro":    round(float(rec), 4),
                "n_samples":     n,
            })

            # Entrées par aspect
            for asp, f1_asp in per_aspect.items():
                records.append({
                    "model":         model_name,
                    "model_short":   MODEL_SHORT.get(model_name, model_name[:8]),
                    "lang":          lang,
                    "aspect":        asp,
                    "f1_macro":      round(float(f1_asp), 4),
                    "precision_macro": None,
                    "recall_macro":    None,
                    "n_samples":     None,
                })
        except Exception as e:
            logger.warning(f"   ⚠️  Transformer JSON ignoré ({os.path.basename(fpath)}) : {e}")

    logger.info(f"   Transformers : {len([r for r in records if r['aspect'] is None])} modèles chargés")
    return records


def load_all_results(results_dir: str = None) -> pd.DataFrame:
    """
    Charge et consolide tous les résultats (SVM + Transformers).
    Retourne un DataFrame avec colonnes : model, lang, aspect, f1_macro, ...
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    logger.info(f"📂 Chargement des résultats depuis : {results_dir}")
    svm_records = _load_svm_results(results_dir)
    tf_records  = _load_transformer_results(results_dir)

    all_records = svm_records + tf_records
    if not all_records:
        logger.warning("⚠️  Aucun résultat trouvé. Vérifiez que les modèles ont été entraînés.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["model_short"] = df.apply(
        lambda r: MODEL_SHORT.get(r["model"], r["model"][:8])
        if "model_short" not in r or pd.isna(r.get("model_short"))
        else r["model_short"],
        axis=1,
    )
    logger.info(f"   {len(df)} entrées consolidées")
    return df


# ===========================================================================
# 2. MATRICE GLOBALE — F1 par modèle et langue
# ===========================================================================

def build_global_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la matrice globale : lignes = modèles, colonnes = langues.
    Cellules = Macro-F1 sur l'ensemble du jeu de test.
    """
    df_global = df[df["aspect"].isna()].copy()
    if df_global.empty:
        logger.warning("Pas de résultats globaux disponibles.")
        return pd.DataFrame()

    matrix = df_global.pivot_table(
        index="model_short", columns="lang",
        values="f1_macro", aggfunc="max",
    ).round(4)

    # Renommer les colonnes avec le label complet
    matrix.columns = [LANG_LABEL.get(c, c.upper()) for c in matrix.columns]
    return matrix


# ===========================================================================
# 3. MATRICE PRINCIPALE — aspect × colonne(modèle+langue)
# ===========================================================================

def build_absa_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la matrice ABSA centrale du mémoire (§2.4.6) :

        Colonnes : SVM(FR), XLM-R(FR), CamemBERT(FR),
                   SVM(RU), XLM-R(RU), RuBERT(RU),
                   SVM(EN), XLM-R(EN)
        Lignes   : qualité, service, prix, livraison, interface
        Cellules : Macro-F1

    Pour les Transformers multilingues (lang="all"), leur résultat par aspect
    est affecté à chaque langue individuellement.
    """
    df_asp = df[df["aspect"].notna()].copy()
    if df_asp.empty:
        logger.warning("Pas de résultats par aspect. Évaluation per_aspect requise.")
        return pd.DataFrame()

    # Dupliquer les résultats multilingues (lang="all") vers FR, EN, RU
    expand_rows = []
    for _, row in df_asp.iterrows():
        if row["lang"] == "all":
            for l in ["fr", "en", "ru"]:
                r = row.copy()
                r["lang"] = l
                expand_rows.append(r)
        else:
            expand_rows.append(row)

    df_exp = pd.DataFrame(expand_rows)

    # Construire la colonne identifiant du modèle
    df_exp["col_label"] = (
        df_exp["model_short"] + "(" + df_exp["lang"].str.upper() + ")"
    )

    # Pivot : aspect × col_label
    matrix = df_exp.pivot_table(
        index="aspect", columns="col_label",
        values="f1_macro", aggfunc="max",
    ).round(4)

    # Réordonner les lignes selon ASPECTS
    aspects_available = [a for a in ASPECTS if a in matrix.index]
    matrix = matrix.reindex(aspects_available)

    # Réordonner les colonnes : SVM → XLM-R → CamemBERT/RuBERT, par langue
    desired_order = []
    for lang in ["fr", "en", "ru"]:
        L = lang.upper()
        for m in [f"SVM({L})", f"XLM-R({L})", f"CamemBERT({L})", f"RuBERT({L})"]:
            if m in matrix.columns:
                desired_order.append(m)
    remaining = [c for c in matrix.columns if c not in desired_order]
    matrix = matrix[desired_order + remaining]

    return matrix


# ===========================================================================
# 4. MATRICE DES GAINS vs SVM BASELINE
# ===========================================================================

def build_gain_matrix(absa_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le gain de chaque Transformer vs SVM pour la même langue.

    Gain = Macro-F1(transformer) − Macro-F1(SVM)
    Formule du §2.4.6 : « Gain = Macro-F1(modèle) − Macro-F1(SVM) »
    """
    if absa_matrix.empty:
        return pd.DataFrame()

    gain_matrix = absa_matrix.copy() * np.nan

    for lang in ["FR", "EN", "RU"]:
        svm_col = f"SVM({lang})"
        if svm_col not in absa_matrix.columns:
            continue
        svm_f1 = absa_matrix[svm_col]
        for col in absa_matrix.columns:
            if col != svm_col and f"({lang})" in col:
                gain_matrix[col] = (absa_matrix[col] - svm_f1).round(4)

    # Supprimer les colonnes SVM (gain = 0 par définition)
    gain_matrix = gain_matrix.drop(
        columns=[c for c in gain_matrix.columns if c.startswith("SVM(")],
        errors="ignore",
    )
    gain_matrix = gain_matrix.dropna(axis=1, how="all")
    return gain_matrix


# ===========================================================================
# 5. VISUALISATIONS
# ===========================================================================

def _save_heatmap(
    matrix: pd.DataFrame,
    title:  str,
    fname:  str,
    fmt:    str = ".3f",
    cmap:   str = "YlGnBu",
    center: float = None,
    annot_kws: dict = None,
) -> str:
    """Génère et sauvegarde une heatmap seaborn."""
    if matrix.empty:
        logger.warning(f"Heatmap '{title}' ignorée (matrice vide).")
        return ""

    nrows, ncols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(8, ncols * 1.4), max(4, nrows * 0.9)))

    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        annot_kws=annot_kws or {"size": 9},
        cbar_kws={"label": "Macro-F1"},
    )
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fpath = os.path.join(FIGURES_DIR, fname)
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"💾 Heatmap → {fpath}")
    return fpath


# ===========================================================================
# 6. RAPPORT TEXTE
# ===========================================================================

def _print_section(title: str, df: pd.DataFrame) -> None:
    """Affiche une matrice dans la console avec un titre."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    if df.empty:
        print("  (aucune donnée)")
    else:
        print(df.to_string())


# ===========================================================================
# 7. PIPELINE PRINCIPAL
# ===========================================================================

def generate_all_matrices(results_dir: str = None) -> dict:
    """
    Pipeline complet de génération des matrices ABSA.

    Produit :
      1. matrix_global.csv         — F1 par modèle × langue (sans aspect)
      2. matrix_absa_main.csv      — F1 par aspect × (modèle+langue)
      3. matrix_gains.csv          — gain Transformer vs SVM
      4. heatmap_global.png
      5. heatmap_absa_main.png
      6. heatmap_gains.png

    Retourne un dict avec les DataFrames et chemins produits.
    """
    logger.info("=" * 60)
    logger.info("📊 GÉNÉRATION DES MATRICES ABSA")
    logger.info("=" * 60)

    # ── Chargement
    df_all = load_all_results(results_dir)
    if df_all.empty:
        logger.error("❌ Aucun résultat disponible — matrices non générées.")
        return {}

    os.makedirs(MATRICES_DIR, exist_ok=True)
    outputs = {}

    # ── 1. Matrice globale
    m_global = build_global_matrix(df_all)
    _print_section("MATRICE GLOBALE  (Macro-F1 par modèle × langue)", m_global)
    if not m_global.empty:
        p = os.path.join(MATRICES_DIR, "matrix_global.csv")
        m_global.to_csv(p)
        logger.info(f"💾 {p}")
        _save_heatmap(
            m_global,
            "Macro-F1 global par modèle et langue",
            "heatmap_global.png",
        )
        outputs["global"] = {"df": m_global, "csv": p}

    # ── 2. Matrice ABSA principale (aspect × modèle+langue)
    m_absa = build_absa_matrix(df_all)
    _print_section("MATRICE ABSA PRINCIPALE  (aspect × modèle+langue)", m_absa)
    if not m_absa.empty:
        p = os.path.join(MATRICES_DIR, "matrix_absa_main.csv")
        m_absa.to_csv(p)
        logger.info(f"💾 {p}")
        _save_heatmap(
            m_absa,
            "Matrice ABSA — Macro-F1 par aspect, modèle et langue",
            "heatmap_absa_main.png",
        )
        outputs["absa_main"] = {"df": m_absa, "csv": p}

    # ── 3. Matrice des gains
    m_gains = build_gain_matrix(m_absa)
    _print_section("MATRICE DES GAINS  (Transformer − SVM baseline)", m_gains)
    if not m_gains.empty:
        p = os.path.join(MATRICES_DIR, "matrix_gains.csv")
        m_gains.to_csv(p)
        logger.info(f"💾 {p}")
        _save_heatmap(
            m_gains,
            "Gain Macro-F1 : Transformer vs SVM baseline",
            "heatmap_gains.png",
            fmt="+.3f",
            cmap="RdYlGn",
            center=0.0,
        )
        outputs["gains"] = {"df": m_gains, "csv": p}

    logger.info("\n✅ Toutes les matrices générées.")
    return outputs


# ===========================================================================
# 8. FONCTIONS HÉRITÉES (conservées pour compatibilité)
# ===========================================================================

def generate_global_matrix() -> pd.DataFrame:
    """Alias conservé pour compatibilité avec l'ancien code."""
    df = load_all_results()
    m  = build_global_matrix(df)
    if not m.empty:
        p = os.path.join(MATRICES_DIR, "matrix_global_svm.csv")
        m.to_csv(p)
        logger.info(f"💾 {p}")
    return m


def generate_aspect_matrix() -> pd.DataFrame:
    """Alias conservé pour compatibilité avec l'ancien code."""
    df = load_all_results()
    m  = build_absa_matrix(df)
    if not m.empty:
        p = os.path.join(MATRICES_DIR, "matrix_aspect_svm_f1.csv")
        m.to_csv(p)
        logger.info(f"💾 {p}")
    return m


# ===========================================================================
# 9. POINT D'ENTRÉE
# ===========================================================================

if __name__ == "__main__":
    outputs = generate_all_matrices()

    print("\n" + "=" * 70)
    print("✅ MATRICES ABSA GÉNÉRÉES")
    print("=" * 70)
    for name, o in outputs.items():
        df = o["df"]
        print(f"\n  [{name}]  {df.shape[0]} lignes × {df.shape[1]} colonnes → {o['csv']}")
    print(f"\n  📁 Figures   : {FIGURES_DIR}")
    print(f"  📁 Matrices  : {MATRICES_DIR}")