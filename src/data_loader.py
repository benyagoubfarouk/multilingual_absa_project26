from transformers import (
    XLMRobertaTokenizer,
    CamembertTokenizer,
    AutoTokenizer
)

# XLM-RoBERTa (многоязычный) - использует SentencePiece
tokenizer_xlm = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# CamemBERT (французский) - токенизатор, адаптированный к французским особенностям
tokenizer_camembert = CamembertTokenizer.from_pretrained("camembert-base")

# RuBERT (русский) - токенизатор WordPiece для кириллицы
tokenizer_rubert = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")