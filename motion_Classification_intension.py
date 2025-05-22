
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time

# 指定本地模型路徑
model_path = "./DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# bart-large-mnli
# deberta-v3-large-zeroshot-v1
# nli-deberta-v3-large 檔案很大
# xlm-roberta-large-xnli
# DeBERTa-v3-large-mnli-fever-anli-ling-wanli

# 加載本地模型
classifier = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path)

text = ""

labels_Positive_Expressions = [
    "Nodding",
    "Smiling",
    "Arms open wide",
    "Excitedly waving hands",
    "Excitedly waving hands",
    "Reaching out",
    ]

labels_Negative_Expressions = [
    "Shaking head",
    "Frowning",
    "Arms crossed",
    "Looking down",
    "Sighing",
    "Leaning back",
    "Looking skeptically at the person",
    ]

labels_Emphasizing_Gestures = [
    "Pointing at the person or the table",
    "Waving hands to emphasize speech",
    "Firmly slapping the table",
    "Hands open wide",
    ]

labels_Interactive_Gestures = [
    "Extending hand for a handshake",
    "Opening arms for a hug",
    "Leaning slightly forward",
    "Gesturing with hand",
    "Reaching out to assist",
    "Making eye contact",
    "Turning body towards the person",
    ]

labels_Thinking_Gestures = [
    "Touching chin",
    "Frowning in thought",
    "Tapping fingers on forehead",
    "Scratching head",
    "Looking down, deep in thought",
    "Tilting head slightly to show confusion",
    ]

labels_ = [
    "Fidgeting with fingers or an object",
    "Avoiding eye contact",
    "Hugging oneself",
    "Slightly stepping back",
    "Hands crossed in front",
    "Shuffling feet",
    "Clenching fists",
    ]

for sentence in test_sentences:
    result = classifier(sentence, labels)
    print(f"句子: {sentence}")
    print(f"預測類別: {result['labels'][0]} (信心度: {result['scores'][0]:.4f})\n")
    print(f"預測類別: {result['labels'][1]} (信心度: {result['scores'][1]:.4f})\n")
    print("==========================================================================")