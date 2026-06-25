# src/emotion_silver_labeler.py
# À exécuter UNE FOIS pour générer les labels d'émotions

from transformers import pipeline
import pandas as pd
import os
from config import DATA_PROCESSED

def label_emotions_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des labels d'émotions via silver labeling.
    Utilise un modèle multilingue pré-entraîné comme proxy.
    """
    # Modèle multilingue de détection d'émotions (EN/FR/autres)
    # Pour le russe : utilise XLM-RoBERTa fine-tuné sur GoEmotions
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=0  # GPU si disponible, sinon device=-1
    )
    
    # Mapping du modèle vers tes 5 classes Plutchik
    MODEL_TO_PLUTCHIK = {
        "joy": "joie",
        "love": "joie",
        "optimism": "satisfaction",
        "neutral": "satisfaction",
        "anger": "colère",
        "disgust": "insatisfaction",
        "fear": "insatisfaction",
        "sadness": "tristesse",
        "surprise": "satisfaction",
    }
    
    print("🔄 Génération des labels d'émotions...")
    
    texts = df["text"].tolist()
    batch_size = 32
    emotions = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Tronquer à 512 tokens
        batch = [t[:512] if isinstance(t, str) else "" for t in batch]
        results = emotion_pipe(batch)
        for r in results:
            label_raw = r[0]["label"].lower()
            mapped = MODEL_TO_PLUTCHIK.get(label_raw, "satisfaction")
            emotions.append(mapped)
        
        if (i // batch_size) % 10 == 0:
            print(f"   {i}/{len(texts)}...")
    
    df = df.copy()
    df["emotion"] = emotions
    
    print(f"\n✅ Distribution émotions :\n{df['emotion'].value_counts()}")
    return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_PROCESSED, "corpus_absa_clean.csv"))
    df = label_emotions_silver(df)
    df.to_csv(os.path.join(DATA_PROCESSED, "corpus_absa_with_emotions.csv"), index=False)
    print("💾 Sauvegardé avec émotions")