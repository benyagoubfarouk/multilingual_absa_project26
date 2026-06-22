# src/prepare_data.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_RAW, DATA_PROCESSED
from utils import setup_logger
import pandas as pd
import xml.etree.ElementTree as ET
import re

logger = setup_logger(__name__)


# ============================================================
# 1. PARSING SPÉCIFIQUE POUR LE FRANÇAIS (SemEval-2016)
# ============================================================

def parse_french_xml(xml_path, lang="fr"):
    """
    Parse spécifique pour le fichier français SemEval-2016.
    Format : <Reviews><Review><sentences><sentence><Opinions><Opinion/>
    """
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Erreur lecture XML français {xml_path} : {e}")
        return pd.DataFrame()
    
    reviews = root.findall(".//Review")
    if not reviews:
        reviews = root.findall(".//review")
    
    logger.info(f"   {len(reviews)} reviews trouvés")
    
    for review in reviews:
        sentences = review.findall(".//sentence")
        
        for sentence in sentences:
            text_elem = sentence.find("text")
            if text_elem is None or not text_elem.text:
                continue
            text = text_elem.text.strip()
            
            opinions = sentence.findall(".//Opinion")
            
            for opinion in opinions:
                target = opinion.get("target", "general")
                polarity = opinion.get("polarity", "").lower()
                
                # Standardisation de la polarité
                if polarity in ["positive", "pos"]:
                    polarity = "pos"
                elif polarity in ["negative", "neg"]:
                    polarity = "neg"
                elif polarity in ["neutral", "neu"]:
                    polarity = "neutral"
                else:
                    continue
                
                if target == "NULL" or not target:
                    target = "general"
                
                records.append({
                    "text": text,
                    "aspect": target.strip(),
                    "polarity": polarity,
                    "lang": lang
                })
    
    logger.info(f"   {len(records)} aspects extraits")
    
    if not records:
        logger.warning("Aucun aspect trouvé. Utilisation du texte entier.")
        for review in reviews:
            for sentence in review.findall(".//sentence"):
                text_elem = sentence.find("text")
                if text_elem is not None and text_elem.text:
                    records.append({
                        "text": text_elem.text.strip(),
                        "aspect": "general",
                        "polarity": "neutral",
                        "lang": lang
                    })
    
    return pd.DataFrame(records)


# ============================================================
# 2. PARSING DES FICHIERS XML SEMEVAL (Anglais)
# ============================================================

def parse_semeval_xml(xml_path, lang="en"):
    """Parse les fichiers XML de SemEval (Anglais)."""
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Erreur lecture XML {xml_path} : {e}")
        return pd.DataFrame()
    
    sentences = root.findall(".//sentence")
    if not sentences:
        sentences = root.findall(".//review")
    
    for sentence in sentences:
        text_elem = sentence.find("text")
        if text_elem is None or not text_elem.text:
            continue
        text = text_elem.text.strip()
        
        aspect_terms = sentence.find(".//aspectTerms")
        if aspect_terms is not None:
            for aspect in aspect_terms.findall("aspectTerm"):
                term = aspect.get("term")
                polarity = aspect.get("polarity", "").lower()
                
                if polarity in ["positive", "pos"]:
                    polarity = "pos"
                elif polarity in ["negative", "neg"]:
                    polarity = "neg"
                elif polarity in ["neutral", "neu"]:
                    polarity = "neutral"
                else:
                    continue
                
                if term and polarity:
                    records.append({
                        "text": text,
                        "aspect": term.strip(),
                        "polarity": polarity,
                        "lang": lang
                    })
    
    if not records:
        for sentence in sentences:
            text_elem = sentence.find("text")
            if text_elem is not None and text_elem.text:
                records.append({
                    "text": text_elem.text.strip(),
                    "aspect": "general",
                    "polarity": "neutral",
                    "lang": lang
                })
    
    return pd.DataFrame(records)


# ============================================================
# 3. PARSING DU CSV RUSENTIMENT (Russe)
# ============================================================

def parse_rusentiment_csv(csv_path, lang="ru"):
    """Parse le dataset RuSentiment (CSV)."""
    
    # ✅ CORRECTION DE L'ENCODAGE RUSSE
    try:
        # Tentative de lecture en cp1251 (Windows) car c'est souvent l'encodage du russe
        df = pd.read_csv(csv_path, encoding='cp1251')
    except:
        try:
            # Si cp1251 échoue, on essaie utf-8
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            # En dernier recours, latin-1
            df = pd.read_csv(csv_path, encoding='latin-1')
    
    logger.info(f"Colonnes trouvées dans RuSentiment : {df.columns.tolist()}")
    
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["text", "sentence", "review", "content", "post", "body", "tweet"]:
            text_col = col
        if col_lower in ["label", "polarity", "sentiment", "class", "category", "tag"]:
            label_col = col
    
    if text_col is None or label_col is None:
        logger.error(f"Colonnes non trouvées. Colonnes : {df.columns.tolist()}")
        return pd.DataFrame()
    
    def map_polarity(val):
        val = str(val).lower().strip()
        if val in ["positive", "pos"]:
            return "pos"
        elif val in ["negative", "neg"]:
            return "neg"
        elif val in ["neutral", "neu"]:
            return "neutral"
        else:
            return None
    
    df['polarity'] = df[label_col].apply(map_polarity)
    df = df[df['polarity'].notna()]
    
    df['text'] = df[text_col].astype(str).str.strip()
    df['aspect'] = "general"
    df['lang'] = "ru"  # FORCER à "ru"
    
    df = df[df['text'].str.len() > 0]
    df = df[['text', 'aspect', 'polarity', 'lang']]
    
    logger.info(f"RuSentiment après nettoyage : {len(df)} lignes")
    logger.info(f"Distribution :\n{df['polarity'].value_counts()}")
    
    return df


# ============================================================
# 4. DÉTECTION AUTOMATIQUE ET TRAITEMENT
# ============================================================

def detect_and_prepare():
    """Détecte et prépare tous les fichiers."""
    logger.info("=" * 60)
    logger.info("🚀 DÉBUT DE LA PRÉPARATION DES DATASETS")
    logger.info("=" * 60)
    
    files = os.listdir(DATA_RAW)
    logger.info(f"Fichiers trouvés dans data/raw/ : {files}")
    
    all_dfs = []
    
    for filename in files:
        filepath = os.path.join(DATA_RAW, filename)
        
        # ---------- Fichiers XML ----------
        if filename.endswith('.xml'):
            logger.info(f"\n📄 Traitement du XML : {filename}")
            
            # Détection automatique du format
            with open(filepath, 'r', encoding='utf-8') as f:
                first_500 = f.read(500)
                if '<Reviews>' in first_500 or '<Opinion' in first_500:
                    # Format français
                    lang = "fr"
                    df = parse_french_xml(filepath, lang)
                else:
                    # Format SemEval standard
                    lang = "en"
                    if "FR" in filename or "French" in filename or "fr" in filename.lower():
                        lang = "fr"
                    df = parse_semeval_xml(filepath, lang)
            
            if not df.empty:
                output_file = os.path.join(DATA_PROCESSED, f"{lang}_processed_{filename.replace('.xml', '.csv')}")
                df.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"   ✅ Sauvegardé : {output_file} ({len(df)} lignes)")
                all_dfs.append(df)
            else:
                logger.warning(f"   ⚠️ Aucune donnée extraite de {filename}")
        
        # ---------- Fichiers CSV ----------
        elif filename.endswith('.csv'):
            logger.info(f"\n📄 Traitement du CSV : {filename}")
            
            if "rusentiment" in filename.lower():
                lang = "ru"
            else:
                lang = "en"
            
            df = parse_rusentiment_csv(filepath, lang)
            
            if not df.empty:
                output_file = os.path.join(DATA_PROCESSED, f"{lang}_processed_{filename}")
                df.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"   ✅ Sauvegardé : {output_file} ({len(df)} lignes)")
                all_dfs.append(df)
            else:
                logger.warning(f"   ⚠️ Aucune donnée extraite de {filename}")
        
        else:
            logger.info(f"   ⏭️ Extension non reconnue : {filename}")
    
    # ---------- Fusion et Réduction (AJOUT IMPORTANT) ----------
    if all_dfs:
        # 1. Fusion brute
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_file = os.path.join(DATA_PROCESSED, "all_datasets_combined.csv")
        combined.to_csv(combined_file, index=False, encoding='utf-8')
        logger.info(f"\n📊 Dataset combiné brut : {len(combined)} lignes")
        logger.info(f"   Distribution par langue :\n{combined['lang'].value_counts()}")
        logger.info(f"   Distribution par polarité :\n{combined['polarity'].value_counts()}")
        
        # 2. ✅ RÉDUCTION AUTOMATIQUE À 800 PHRASES PAR LANGUE
        logger.info("\n🔄 Réduction du dataset à 800 phrases par langue...")
        reduced_dfs = []
        for lang in combined['lang'].unique():
            lang_df = combined[combined['lang'] == lang]
            # Prendre 800 échantillons aléatoires (ou moins si pas assez)
            sample_size = min(800, len(lang_df))
            reduced_dfs.append(lang_df.sample(n=sample_size, random_state=42))
        
        final_df = pd.concat(reduced_dfs, ignore_index=True)
        logger.info(f"   ✅ Dataset réduit : {len(final_df)} lignes (800/langue)")
        logger.info(f"   Distribution par langue :\n{final_df['lang'].value_counts()}")
        
        # 3. Sauvegarder le dataset réduit AVEC LES ASPECTS (POUR L'ABSA)
        reduced_file = os.path.join(DATA_PROCESSED, "all_datasets_reduced_with_aspects.csv")
        final_df.to_csv(reduced_file, index=False, encoding='utf-8')
        logger.info(f"💾 Sauvegardé (ABSA) : {reduced_file}")
        
        # 4. Sauvegarder une version sans colonne "aspect" (pour compatibilité SVM/Transformers classiques si besoin)
        reduced_no_aspect = final_df.drop(columns=['aspect'])
        reduced_no_aspect_file = os.path.join(DATA_PROCESSED, "all_datasets_reduced_no_aspect.csv")
        reduced_no_aspect.to_csv(reduced_no_aspect_file, index=False, encoding='utf-8')
        logger.info(f"💾 Sauvegardé (Classique) : {reduced_no_aspect_file}")

    logger.info("\n" + "=" * 60)
    logger.info("🎉 PRÉPARATION TERMINÉE !")
    logger.info("=" * 60)


# ============================================================
# 5. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    detect_and_prepare()