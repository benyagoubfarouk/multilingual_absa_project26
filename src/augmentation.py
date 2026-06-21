# src/augmentation.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, DATA_AUGMENTED, RANDOM_SEED, AUGMENTATION_PARAMS
from utils import setup_logger, set_seed
import pandas as pd
import random
import re

# Pour les synonymes, on utilise des librairies simples ou des dictionnaires
# Note : Pour une version robuste, il faudrait utiliser des API WordNet/WOLF/RuWordNet, 
# mais voici une version algorithmique de base qui fonctionne.
try:
    from nltk.corpus import wordnet
    import nltk
    nltk.download('wordnet', quiet=True)
    HAS_WORDNET = True
except:
    HAS_WORDNET = False

logger = setup_logger(__name__)

class TextAugmenter:
    def __init__(self):
        set_seed(RANDOM_SEED)
        
    def get_synonyms(self, word, lang="en"):
        """Retourne une liste de synonymes pour un mot donné selon la langue."""
        if not HAS_WORDNET:
            return []
        
        synonyms = set()
        # WordNet est principalement en anglais, on fait une approximation pour le français/russe
        # Dans un vrai projet, on utiliserait WOLF pour le français et RuWordNet pour le russe
        if lang == "en":
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word and "_" not in lemma.name():
                        synonyms.add(lemma.name().replace('_', ' '))
        elif lang == "fr":
            # Fallback pour le français : on utilise quelques règles simples si WordNet n'est pas dispo
            if word in ["bon", "bonne"]: return ["excellent", "super", "correct"]
            if word in ["mauvais", "mauvaise"]: return ["médiocre", "déplorable"]
        elif lang == "ru":
            # Fallback pour le russe
            if word in ["хороший", "хорошо"]: return ["отличный", "прекрасный"]
            if word in ["плохой", "плохо"]: return ["ужасный", "скверный"]
            
        return list(synonyms)

    def augment_sentence(self, text, lang="en", prob=0.3):
        """Remplace certains mots par des synonymes."""
        words = text.split()
        new_words = []
        
        for word in words:
            if random.random() < prob:
                synonyms = self.get_synonyms(word, lang)
                if synonyms:
                    new_word = random.choice(synonyms)
                    new_words.append(new_word)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return " ".join(new_words)

    def augment_dataframe(self, df, target_class="neutral", factor=2):
        """Augmente la classe cible pour atteindre un équilibre."""
        logger.info(f"🚀 Début de l'augmentation pour la classe : {target_class}")
        
        # Séparer les classes
        df_neutral = df[df['polarity'] == target_class].copy()
        df_others = df[df['polarity'] != target_class].copy()
        
        logger.info(f"   Taille actuelle neutre : {len(df_neutral)}")
        logger.info(f"   Taille autres classes : {len(df_others)}")
        
        # Calculer combien il en faut pour équilibrer
        target_size = len(df_others) // 2  # Pour avoir 33% de neutre (approx)
        if len(df_neutral) >= target_size:
            logger.info("   La classe neutre est déjà équilibrée. Pas d'augmentation nécessaire.")
            return df
        
        num_to_generate = target_size - len(df_neutral)
        logger.info(f"   Génération de {num_to_generate} nouveaux échantillons neutres...")
        
        augmented_rows = []
        # On boucle sur les neutres existants pour en générer de nouveaux
        for _ in range(num_to_generate):
            # Sélection aléatoire d'une phrase neutre existante
            row = df_neutral.sample(n=1).iloc[0]
            new_text = self.augment_sentence(row['text_processed'], lang=row['lang'], prob=AUGMENTATION_PARAMS['synonym_prob'])
            
            # Créer une nouvelle ligne
            new_row = row.copy()
            new_row['text_processed'] = new_text
            new_row['text'] = new_text  # On garde le texte original modifié
            augmented_rows.append(new_row)
        
        # Créer le nouveau dataframe équilibré
        df_augmented = pd.concat([df_others, df_neutral, pd.DataFrame(augmented_rows)], ignore_index=True)
        
        logger.info(f"✅ Augmentation terminée. Nouvelle taille : {len(df_augmented)}")
        logger.info(f"   Distribution :\n{df_augmented['polarity'].value_counts()}")
        
        return df_augmented

if __name__ == "__main__":
    # Chargement du dataset
    input_path = os.path.join(DATA_PROCESSED, "all_datasets_svm_ready.csv")
    df = pd.read_csv(input_path)
    
    # Création de l'augmenteur
    augmenter = TextAugmenter()
    
    # Augmentation
    df_aug = augmenter.augment_dataframe(df, target_class="neutral")
    
    # Sauvegarde
    output_path = os.path.join(DATA_AUGMENTED, "all_datasets_augmented.csv")
    df_aug.to_csv(output_path, index=False)
    logger.info(f"💾 Fichier augmenté sauvegardé : {output_path}")