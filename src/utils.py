# src/utils.py
import random
import numpy as np
import torch
import logging
import json
import os
from pathlib import Path
from config import RANDOM_SEED, LOGS_DIR  # ✅ Plus de 'src.'

def set_seed(seed=RANDOM_SEED):
    """Fixer toutes les graines aléatoires pour la reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name="ABSA", log_file=None):
    """Configurer un logger simple"""
    if log_file is None:
        log_file = os.path.join(LOGS_DIR, "pipeline.log")
    # Assurer que le dossier logs existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def save_json(data, filepath):
    """Sauvegarder un dictionnaire en JSON"""
    # Assurer que le dossier existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)