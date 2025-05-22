from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time
import spacy
import re
import json

nlp = spacy.load("en_core_web_sm")  # 下載並載入英語模型

# 指定本地模型路徑
model_path = "./DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# bart-large-mnli
# deberta-v3-large-zeroshot-v1
# nli-deberta-v3-large 檔案很大
# xlm-roberta-large-xnli
# DeBERTa-v3-large-mnli-fever-anli-ling-wanli

# 加載本地模型
classifier = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path)

text = "I’d be happy to help you explore our latest products!"

labels = [
    "Greeting Group : Actions used to welcome a user when they approach.",
    "Farewell Set : Actions used to bid farewell at the end of an interaction.",
    "Emotion Set : Actions that express the user's emotions.",
    "Product Showcase : Actions used to showcase or display a product.",
    "Navigation : Actions performed when moving from one location to another.",
    "Error Handling : Actions shown when the system encounters an error.",
    "Listening State : Actions indicating the system is listening to the user's input.",
    "Talking State : Gestures that accompany the system's spoken response.",
    "Idle Animations : Actions displayed during periods of no user interaction."
]
labels2 = [
    "positive",
    "negative",
    "neutral",
]

start_time = time.time()

result = classifier(text, labels)
print(f"句子: {text}")
# print(f"預測類別: {result['labels'][0]} (信心度: {result['scores'][0]:.4f})\n")
# print(f"預測類別: {result['labels'][1]} (信心度: {result['scores'][1]:.4f})\n")
# print(f"預測類別: {result['labels'][2]} (信心度: {result['scores'][2]:.4f})\n")
motion_tag  = re.match(r"^(.*?)(?=:)", result['labels'][0])
motion = motion_tag.group(1) if motion_tag else None

result2 = classifier(text, labels2)
# print(f"預測類別: {result2['labels'][0]} (信心度: {result2['scores'][0]:.4f})\n")
# print(f"預測類別: {result['labels'][1]} (信心度: {result['scores'][1]:.4f})\n")
# print(f"預測類別: {result['labels'][2]} (信心度: {result['scores'][2]:.4f})\n")

doc = nlp(text)

verbs = [token.text for token in doc if token.pos_ == "VERB"]
# 替換動詞並插入 [@timestamp_begin] 標記
modified_text = text
for verb in verbs:
    modified_text = modified_text.replace(verb, f"[@timestamp_begin]{verb}")
print(modified_text)

output = {
    "text": modified_text,
    "intention": result2['labels'][0],
    "motion_tag": [
        {
            "ID": 1,
            "motion": motion
        }
    ]
}

print(json.dumps(output, indent=2, ensure_ascii=False))

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")