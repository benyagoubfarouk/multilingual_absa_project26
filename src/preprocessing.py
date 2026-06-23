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

# Chargement différé des modèles (pour éviter les imports inutiles)
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
    """
    Charge et retourne le modèle pymorphy2 pour le russe.
    Gère l'incompatibilité avec Python 3.12 (Colab) sans planter.
    """
    global _PYMORPHY
    if _PYMORPHY is None:
        try:
            import pymorphy2
            # Test pour vérifier si la version est compatible
            try:
                _PYMORPHY = pymorphy2.MorphAnalyzer()
                logger.info("✅ Pymorphy2 chargé (version compatible)")
            except AttributeError:
                # Si on a une erreur AttributeError (le bug de Python 3.12)
                logger.warning("⚠️ Version de pymorphy2 incompatible avec Python 3.12. Désactivation de la lemmatisation russe.")
                _PYMORPHY = None
        except ImportError:
            logger.warning("⚠️ pymorphy2 non installé. Désactivation de la lemmatisation russe.")
            _PYMORPHY = None
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
                # Fallback : liste manuelle
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
                # Liste de base pour le russe
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
    """
    Normalise le texte : minuscules, suppression des caractères spéciaux,
    normalisation des espaces.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Mise en minuscules
    text = text.lower()
    
    # Normalisation Unicode (NFKD pour décomposer les caractères accentués)
    # On conserve les accents pour le français et le russe
    # Pour l'anglais, on peut les supprimer
    if lang == "en":
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Suppression des URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Suppression des mentions et hashtags (Twitter)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Suppression des caractères spéciaux (garde lettres, chiffres, espaces, apostrophes)
    # On garde l'apostrophe pour le français
    if lang == "fr":
        text = re.sub(r'[^\w\s\']', '', text)
    else:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Suppression des chiffres (sauf pour le russe où les chiffres sont parfois importants)
    if lang != "ru":
        text = re.sub(r'\d+', '', text)
    
    # Normalisation des espaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================
# 3. FONCTIONS DE LEMMATISATION
# ============================================================

def lemmatize_text(text, lang="en"):
    """
    Lemmatisation du texte selon la langue.
    Utilise spaCy pour l'anglais et le français, pymorphy2 pour le russe.
    """
    if not text:
        return ""
    
    if lang == "ru":
        # Lemmatisation russe avec pymorphy2 (si disponible)
        morph = get_pymorphy2()
        if morph is None:
            # Si pymorphy2 n'est pas disponible (bug Python 3.12), on retourne le texte brut
            # Les modèles Transformers gèrent le russe très bien sans lemmatisation
            return text
        
        words = text.split()
        lemmatized = []
        for word in words:
            parsed = morph.parse(word)
            if parsed:
                # Prendre la forme normale (infinitif / nominatif)
                lemmatized.append(parsed[0].normal_form)
            else:
                lemmatized.append(word)
        return " ".join(lemmatized)
    
    else:
        # Lemmatisation avec spaCy (en, fr)
        nlp = get_spacy_model(lang)
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_punct])


# ============================================================
# 4. SUPPRESSION DES STOPWORDS
# ============================================================

def remove_stopwords(text, lang="en", custom_stopwords=None):
    """
    Supprime les mots-outils (stopwords) du texte.
    """
    if not text:
        return ""
    
    stopwords = get_stopwords(lang)
    if custom_stopwords:
        stopwords = stopwords.union(set(custom_stopwords))
    
    words = text.split()
    filtered = [w for w in words if w not in stopwords]
    
    return " ".join(filtered)


# ============================================================
# 5. PIPELINE COMPLET POUR MODÈLES CLASSIQUES (SVM)
# ============================================================

def preprocess_for_svm(text, lang="en", remove_stop=True, lemmatize=True):
    """
    Pipeline complet de prétraitement pour les modèles classiques (SVM/TF-IDF).
    5 étapes :
      1. Normalisation
      2. Tokenisation (via split)
      3. Suppression des stopwords (optionnel)
      4. Lemmatisation (optionnel)
      5. Nettoyage final
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Étape 1 : Normalisation
    text = normalize_text(text, lang)
    
    if not text:
        return ""
    
    # Étape 2 : Suppression des stopwords
    if remove_stop:
        text = remove_stopwords(text, lang)
    
    # Étape 3 : Lemmatisation
    if lemmatize:
        text = lemmatize_text(text, lang)
    
    # Étape 4 : Nettoyage final (espaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================
# 6. PIPELINE SIMPLIFIÉ POUR MODÈLES TRANSFORMERS
# ============================================================

def preprocess_for_transformers(text, lang="en"):
    """
    Pipeline minimal pour les modèles Transformers.
    Seulement une normalisation de base.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Normalisation légère (minuscules, suppressions des URLs)
    text = normalize_text(text, lang)
    
    # Pas de suppression de stopwords ni de lemmatisation
    # Les Transformers gèrent cela via leur tokeniseur
    
    return text


# ============================================================
# 7. TRAITEMENT DES DATAFRAMES EN MASSE
# ============================================================

def preprocess_dataframe(df, text_col="text", lang_col="lang", 
                         model_type="svm", remove_stop=True, lemmatize=True):
    """
    Applique le prétraitement à toutes les lignes d'un DataFrame.
    
    Args:
        df: DataFrame avec les colonnes text et lang
        text_col: Nom de la colonne contenant le texte
        lang_col: Nom de la colonne contenant la langue
        model_type: "svm" ou "transformers"
        remove_stop: Supprimer les stopwords (pour SVM)
        lemmatize: Lemmatiser (pour SVM)
    
    Returns:
        DataFrame avec une nouvelle colonne 'text_processed'
    """
    if df is None or df.empty:
        return df
    
    # Copie pour ne pas modifier l'original
    df = df.copy()
    
    # Choisir la fonction de prétraitement
    if model_type == "svm":
        def process_row(row):
            return preprocess_for_svm(
                row[text_col], 
                lang=row[lang_col],
                remove_stop=remove_stop,
                lemmatize=lemmatize
            )
    else:  # transformers
        def process_row(row):
            return preprocess_for_transformers(row[text_col], lang=row[lang_col])
    
    # Application du prétraitement
    logger.info(f"🔄 Prétraitement pour {model_type} sur {len(df)} lignes...")
    df['text_processed'] = df.apply(process_row, axis=1)
    
    # Suppression des lignes vides après prétraitement
    df = df[df['text_processed'].str.len() > 0]
    logger.info(f"   ✅ {len(df)} lignes après prétraitement")
    
    return df


# ============================================================
# 8. PRÉPARATION POUR SVM (VECTORISATION)
# ============================================================

def prepare_for_svm(df, text_col="text_processed", lang_col="lang"):
    """
    Prépare les données pour le SVM :
    - Sépare par langue
    - Retourne un dictionnaire {lang: (texts, labels)}
    """
    results = {}
    
    for lang in df[lang_col].unique():
        lang_df = df[df[lang_col] == lang]
        texts = lang_df[text_col].tolist()
        labels = lang_df['polarity'].tolist()
        results[lang] = (texts, labels)
        logger.info(f"   {lang}: {len(texts)} échantillons")
    
    return results


# ============================================================
# 9. FONCTION DE TEST
# ============================================================

def test_preprocessing():
    """Test simple des fonctions de prétraitement."""
    
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
        
        # Pour SVM
        processed_svm = preprocess_for_svm(text, lang)
        logger.info(f"   SVM processed  : {processed_svm}")
        
        # Pour Transformers
        processed_tf = preprocess_for_transformers(text, lang)
        logger.info(f"   Transformers   : {processed_tf}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Test terminé")
    logger.info("=" * 60)


# ============================================================
# 10. POINT D'ENTRÉE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Test des fonctions
    test_preprocessing()
    
    # Exemple de chargement et prétraitement
    logger.info("\n📂 Chargement du dataset combiné...")
    combined_path = os.path.join(DATA_PROCESSED, "all_datasets_combined.csv")
    
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        logger.info(f"   ✅ {len(df)} lignes chargées")
        
        # Prétraitement pour SVM
        logger.info("\n🔄 Prétraitement pour SVM...")
        df_svm = preprocess_dataframe(df, model_type="svm")
        df_svm.to_csv(os.path.join(DATA_PROCESSED, "all_datasets_svm_ready.csv"), index=False)
        logger.info("   ✅ Sauvegardé : all_datasets_svm_ready.csv")
        
        # Prétraitement pour Transformers
        logger.info("\n🔄 Prétraitement pour Transformers...")
        df_tf = preprocess_dataframe(df, model_type="transformers")
        df_tf.to_csv(os.path.join(DATA_PROCESSED, "all_datasets_tf_ready.csv"), index=False)
        logger.info("   ✅ Sauvegardé : all_datasets_tf_ready.csv")
    else:
        logger.warning(f"⚠️ Fichier {combined_path} non trouvé. Exécutez d'abord prepare_data.py")