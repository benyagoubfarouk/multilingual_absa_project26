# src/preprocessing.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, DATA_AUGMENTED, LANGUAGES
from utils import setup_logger
import pandas as pd
import re
import unicodedata

logger = setup_logger(__name__)

# ============================================================
# 1. CHARGEMENT DES MODÈLES LINGUISTIQUES
# ============================================================

_SPACY_MODELS = {}
_PYMORPHY = None
_STOPWORDS = {}

def get_spacy_model(lang):
    """Charge et retourne le modèle spaCy pour la langue donnée."""
    global _SPACY_MODELS
    if lang not in _SPACY_MODELS:
        import spacy
        model_name = {
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "ru": "ru_core_news_sm"
        }.get(lang, "en_core_web_sm")
        
        try:
            _SPACY_MODELS[lang] = spacy.load(model_name)
            logger.info(f"✅ Modèle spaCy chargé : {model_name}")
        except OSError:
            logger.warning(f"⚠️ Modèle {model_name} non trouvé. Téléchargement...")
            spacy.cli.download(model_name)
            _SPACY_MODELS[lang] = spacy.load(model_name)
    
    return _SPACY_MODELS[lang]

def get_pymorphy2():
    """Charge et retourne le modèle pymorphy2 pour le russe."""
    global _PYMORPHY
    if _PYMORPHY is None:
        try:
            import pymorphy2
            _PYMORPHY = pymorphy2.MorphAnalyzer()
            logger.info("✅ Pymorphy2 chargé")
        except ImportError:
            # On laisse l'erreur remonter pour que l'utilisateur installe la librairie
            logger.error("❌ pymorphy2 non installé. Installez avec : pip install pymorphy2")
            raise
    return _PYMORPHY

def get_stopwords(lang):
    """Retourne la liste des stopwords pour la langue donnée."""
    global _STOPWORDS
    if lang not in _STOPWORDS:
        if lang == "fr":
            try:
                from nltk.corpus import stopwords as nltk_stopwords
                import nltk
                nltk.download('stopwords', quiet=True)
                _STOPWORDS[lang] = set(nltk_stopwords.words('french'))
            except:
                _STOPWORDS[lang] = {
                    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
                    "et", "ou", "mais", "donc", "car", "ni", "or", "pour", "par",
                    "avec", "sans", "chez", "dans", "sur", "sous", "entre", "parmi",
                    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
                    "me", "te", "se", "le", "la", "lui", "leur", "y", "en",
                    "ce", "cet", "cette", "ces", "mon", "ton", "son", "notre",
                    "votre", "leur", "ma", "ta", "sa", "mes", "tes", "ses",
                    "nos", "vos", "leurs", "quel", "quels", "quelle", "quelles"
                }
        elif lang == "en":
            try:
                from nltk.corpus import stopwords as nltk_stopwords
                import nltk
                nltk.download('stopwords', quiet=True)
                _STOPWORDS[lang] = set(nltk_stopwords.words('english'))
            except:
                _STOPWORDS[lang] = {
                    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her",
                    "it", "we", "us", "our", "they", "them", "their", "a", "an",
                    "the", "of", "and", "to", "in", "is", "for", "on", "at", "by",
                    "with", "as", "was", "were", "are", "been", "being", "have",
                    "has", "had", "do", "does", "did", "will", "would", "could",
                    "should", "may", "might", "must", "shall", "from", "up", "down"
                }
        elif lang == "ru":
            try:
                from nltk.corpus import stopwords as nltk_stopwords
                import nltk
                nltk.download('stopwords', quiet=True)
                _STOPWORDS[lang] = set(nltk_stopwords.words('russian'))
            except:
                _STOPWORDS[lang] = {
                    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со",
                    "как", "а", "то", "все", "она", "так", "его", "но", "да",
                    "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
                    "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
                    "о", "из", "ним", "теперь", "когда", "даже", "ну", "вдруг"
                }
        else:
            _STOPWORDS[lang] = set()
    
    return _STOPWORDS[lang]

# ============================================================
# 2. FONCTIONS DE NORMALISATION
# ============================================================

def normalize_text(text, lang="en"):
    if not isinstance(text, str) or not text:
        return ""
    
    text = text.lower()
    
    if lang == "en":
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    if lang == "fr":
        text = re.sub(r'[^\w\s\']', '', text)
    else:
        text = re.sub(r'[^\w\s]', '', text)
    
    if lang != "ru":
        text = re.sub(r'\d+', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ============================================================
# 3. FONCTIONS DE LEMMATISATION
# ============================================================

def lemmatize_text(text, lang="en"):
    if not text:
        return ""
    
    if lang == "ru":
        morph = get_pymorphy2()
        words = text.split()
        lemmatized = []
        for word in words:
            parsed = morph.parse(word)
            if parsed:
                lemmatized.append(parsed[0].normal_form)
            else:
                lemmatized.append(word)
        return " ".join(lemmatized)
    
    else:
        nlp = get_spacy_model(lang)
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_punct])

# ============================================================
# 4. SUPPRESSION DES STOPWORDS
# ============================================================

def remove_stopwords(text, lang="en", custom_stopwords=None):
    if not text:
        return ""
    
    stopwords = get_stopwords(lang)
    if custom_stopwords:
        stopwords = stopwords.union(set(custom_stopwords))
    
    words = text.split()
    filtered = [w for w in words if w not in stopwords]
    
    return " ".join(filtered)

# ============================================================
# 5. PIPELINE SVM
# ============================================================

def preprocess_for_svm(text, lang="en", remove_stop=True, lemmatize=True):
    if not isinstance(text, str) or not text:
        return ""
    
    text = normalize_text(text, lang)
    if not text:
        return ""
    
    if remove_stop:
        text = remove_stopwords(text, lang)
    
    if lemmatize:
        text = lemmatize_text(text, lang)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ============================================================
# 6. PIPELINE TRANSFORMERS
# ============================================================

def preprocess_for_transformers(text, lang="en"):
    if not isinstance(text, str) or not text:
        return ""
    
    text = normalize_text(text, lang)
    return text

# ============================================================
# 7. TRAITEMENT DES DATAFRAMES
# ============================================================

def preprocess_dataframe(df, text_col="text", lang_col="lang", 
                         model_type="svm", remove_stop=True, lemmatize=True):
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    if model_type == "svm":
        def process_row(row):
            return preprocess_for_svm(
                row[text_col], 
                lang=row[lang_col],
                remove_stop=remove_stop,
                lemmatize=lemmatize
            )
    else:
        def process_row(row):
            return preprocess_for_transformers(row[text_col], lang=row[lang_col])
    
    logger.info(f"🔄 Prétraitement pour {model_type} sur {len(df)} lignes...")
    df['text_processed'] = df.apply(process_row, axis=1)
    
    df = df[df['text_processed'].str.len() > 0]
    logger.info(f"   ✅ {len(df)} lignes après prétraitement")
    
    return df

# ============================================================
# 8. PRÉPARATION POUR SVM
# ============================================================

def prepare_for_svm(df, text_col="text_processed", lang_col="lang"):
    results = {}
    
    for lang in df[lang_col].unique():
        lang_df = df[df[lang_col] == lang]
        texts = lang_df[text_col].tolist()
        labels = lang_df['polarity'].tolist()
        results[lang] = (texts, labels)
        logger.info(f"   {lang}: {len(texts)} échantillons")
    
    return results

# ============================================================
# 9. FONCTION DE TEST (Pour vérifier sur Colab)
# ============================================================

def test_preprocessing():
    test_texts = {
        "en": "I absolutely loved the food! The service was terrible though...",
        "fr": "La nourriture était excellente mais le service était vraiment mauvais.",
        "ru": "Еда была отличной, но обслуживание было ужасным."
    }
    
    logger.info("=" * 60)
    logger.info("🧪 TEST DES FONCTIONS DE PRÉTRAITEMENT")
    logger.info("=" * 60)
    
    for lang, text in test_texts.items():
        logger.info(f"\n📌 Langue : {lang}")
        logger.info(f"   Texte original : {text}")
        
        processed_svm = preprocess_for_svm(text, lang)
        logger.info(f"   SVM processed  : {processed_svm}")
        
        processed_tf = preprocess_for_transformers(text, lang)
        logger.info(f"   Transformers   : {processed_tf}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Test terminé")
    logger.info("=" * 60)

# ============================================================
# 10. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    test_preprocessing()
    
    # Chargement du dataset réduit (celui généré par prepare_data.py)
    reduced_path = os.path.join(DATA_PROCESSED, "all_datasets_reduced_with_aspects.csv")
    
    if os.path.exists(reduced_path):
        df = pd.read_csv(reduced_path)
        logger.info(f"   ✅ {len(df)} lignes chargées depuis le dataset réduit")
        
        # Prétraitement SVM
        logger.info("\n🔄 Prétraitement pour SVM...")
        df_svm = preprocess_dataframe(df, model_type="svm")
        df_svm.to_csv(os.path.join(DATA_PROCESSED, "all_datasets_reduced_svm.csv"), index=False)
        logger.info("   ✅ Sauvegardé : all_datasets_reduced_svm.csv")
        
        # Prétraitement Transformers
        logger.info("\n🔄 Prétraitement pour Transformers...")
        df_tf = preprocess_dataframe(df, model_type="transformers")
        df_tf.to_csv(os.path.join(DATA_PROCESSED, "all_datasets_reduced_tf.csv"), index=False)
        logger.info("   ✅ Sauvegardé : all_datasets_reduced_tf.csv")
    else:
        logger.warning(f"⚠️ Fichier réduit non trouvé. Exécutez d'abord prepare_data.py")