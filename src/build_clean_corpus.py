# src/build_clean_corpus.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import DATA_PROCESSED, RANDOM_SEED
from src.aspect_mapper import map_aspects

def build_clean_corpus():
    # 1. Charger le corpus combiné brut
    df = pd.read_csv(os.path.join(DATA_PROCESSED, "all_datasets_combined.csv"))
    
    # 2. Mapper les aspects
    df = map_aspects(df)
    
    # 3. Analyse du russe (RuSentiment n'a pas d'aspects ABSA)
    df_ru = df[df['lang'] == 'ru']
    if len(df_ru) < 100:
        print("⚠️  Russe : données ABSA insuffisantes après mapping.")
        print("   → Utiliser uniquement FR + EN pour la comparaison principale.")
        print("   → Mentionner dans le rapport comme limite du corpus RuSentiment.")
    
    # 4. Équilibrer : max 800 lignes par langue, équilibre neg/pos/neutral
    TARGET_PER_LANG = 800
    RATIOS = {"pos": 0.40, "neg": 0.40, "neutral": 0.20}
    
    balanced_parts = []
    for lang in df['lang'].unique():
        df_lang = df[df['lang'] == lang]
        for polarity, ratio in RATIOS.items():
            df_pol = df_lang[df_lang['polarity'] == polarity]
            n = int(TARGET_PER_LANG * ratio)
            n = min(n, len(df_pol))
            sample = df_pol.sample(n=n, random_state=RANDOM_SEED)
            balanced_parts.append(sample)
    
    df_final = pd.concat(balanced_parts).reset_index(drop=True)
    
    print(f"\n✅ Corpus final : {len(df_final)} lignes")
    print(df_final.groupby(['lang', 'polarity']).size().unstack(fill_value=0))
    
    # 5. Sauvegarder
    out = os.path.join(DATA_PROCESSED, "corpus_absa_clean.csv")
    df_final.to_csv(out, index=False)
    print(f"\n💾 Sauvegardé : {out}")
    return df_final

if __name__ == "__main__":
    build_clean_corpus()