from transformers import pipeline
import time
import spacy
import re
import json

def load_models(model_path: str):
    """載入 NLP 模型"""
    nlp = spacy.load("en_core_web_sm")
    classifier = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path)
    return nlp, classifier


def classify_text(classifier, text: str, labels: list) -> str:
    """執行零樣本分類並回傳最高置信度的標籤"""
    result = classifier(text, labels)
    return result["labels"][0] if result and "labels" in result else None


def extract_motion(label: str) -> str:
    """從標籤中提取動作名稱"""
    match = re.match(r"^(.*?)(?=:)", label)
    return match.group(1) if match else None


def insert_timestamps(nlp, text: str) -> str:
    """在動詞前插入 [@timestamp_begin] 標記"""
    doc = nlp(text)

    for token in doc:
        if token.pos_ == "VERB":
            return text.replace(token.text, f"[@timestamp_begin]{token.text}", 1)

    return text


def main(text: str):
    model_path = "./DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    # 載入模型
    nlp, classifier = load_models(model_path)

    # 定義標籤
    motion_labels = [
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
    sentiment_labels = ["positive", "negative", "neutral"]

    start_time = time.time()

    # 預測動作標籤
    motion_label = classify_text(classifier, text, motion_labels)
    motion = extract_motion(motion_label)

    # 預測情感分類
    intention = classify_text(classifier, text, sentiment_labels)

    # 插入時間戳記
    modified_text = insert_timestamps(nlp, text)

    # 構建輸出 JSON
    output = {
        "labeled_text": modified_text,
        "intention": intention,
        "motion_tags": [{"ID": 1, "motion": motion}]
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    test_sentence = "I’d be happy to help you explore our latest phone!"
    main(test_sentence)
