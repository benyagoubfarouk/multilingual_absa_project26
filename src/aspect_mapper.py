# src/aspect_mapper.py
import pandas as pd

ASPECT_MAP = {
    # → Qualité
    "quality": "qualité", "food": "qualité", "meal": "qualité", "taste": "qualité",
    "plats": "qualité", "cuisine": "qualité", "repas": "qualité", "produits": "qualité",
    "battery life": "qualité", "battery": "qualité", "screen": "qualité",
    "build quality": "qualité", "hardware": "qualité", "performance": "qualité",
    # → Service
    "service": "service", "staff": "service", "waiter": "service", "waitress": "service",
    "accueil": "service", "personnel": "service", "support": "service",
    "customer service": "service",
    # → Prix
    "price": "prix", "cost": "prix", "value": "prix", "prix": "prix",
    "rapport qualité/prix": "prix", "tarif": "prix",
    # → Livraison
    "delivery": "livraison", "shipping": "livraison", "livraison": "livraison",
    "delai": "livraison", "packaging": "livraison",
    # → Interface
    "interface": "interface", "software": "interface", "app": "interface",
    "features": "interface", "usability": "interface", "design": "interface",
    "menu": "interface", "carte": "interface",
}

def map_aspects(df: pd.DataFrame) -> pd.DataFrame:
    def normalize(aspect):
        if pd.isna(aspect):
            return None
        a = str(aspect).lower().strip()
        return ASPECT_MAP.get(a, None)
    
    df = df.copy()
    df["aspect_mapped"] = df["aspect"].apply(normalize)
    before = len(df)
    df = df.dropna(subset=["aspect_mapped"])
    df["aspect"] = df["aspect_mapped"]
    df = df.drop(columns=["aspect_mapped"])
    after = len(df)
    
    print(f"Lignes avant mapping : {before}")
    print(f"Lignes après mapping : {after} (supprimé {before - after} non mappées)")
    print(f"\nDistribution finale :\n{df['aspect'].value_counts()}")
    print(f"\nPar langue :\n{df.groupby(['lang','aspect']).size().unstack(fill_value=0)}")
    
    return df