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
    print("==================================================================")
    print("目前處理的語句是 : ", text)
    print(f"預測類別: {result['labels'][0]} (信心度: {result['scores'][0]:.4f})\n")
    print(f"預測類別: {result['labels'][1]} (信心度: {result['scores'][1]:.4f})\n")
    print(f"預測類別: {result['labels'][2]} (信心度: {result['scores'][2]:.4f})\n")
    return result["labels"][0] if result and "labels" in result else None


def extract_motion(label: str) -> str:
    """從標籤中提取動作名稱"""
    match = re.match(r"^(.*?)(?=:)", label)
    return match.group(1).strip() if match else None


def insert_timestamps(nlp, text: str) -> str:
    """在動詞前插入 [@timestamp_begin] 標記"""
    doc = nlp(text)
    for token in doc:
        # 當 token 是動詞且不是 's 或 'm 時
        if token.pos_ == "VERB" and token.text not in ["’s", "’m"]:
            return text.replace(token.text, f"[@timestamp_begin]{token.text}", 1)

        # 如果是 be 動詞
        elif token.pos_ == "AUX":
            for adj in doc:
                if adj.pos_ == "ADJ":
                    return text.replace(adj.text, f"[@timestamp_begin]{adj.text}", 1)
    return text


def main(texts: str):
    model_path = "./deberta-v3-large-zeroshot-v1"
    # DeBERTa-v3-large-mnli-fever-anli-ling-wanli
    # deberta-v3-large-zeroshot-v1
    # deberta-v3-large

    # 載入模型
    nlp, classifier = load_models(model_path)

    motion_labels = [
        "Interaction Openers & Closers: This category includes statements that mark the beginning or conclusion of an interaction, such as greetings, introductions, and farewells. Example: 'Hello! How can I assist you today?' or 'Goodbye! Hope to see you soon!'",
        "Emotion Set: This category covers expressions of emotional states, including happiness, sadness, excitement, frustration, and empathy. It includes statements that convey understanding or emotional support. Example: 'I’m so excited about this!' or 'That must have been really challenging, I completely understand.'",
        "Idle Animations: This category consists of statements related to moments of inactivity or waiting, often indicating a lack of immediate engagement. Example: 'Please hold on while I look that up.' or 'I’m waiting for your input.'",
        "Product Showcase: This category involves descriptions, promotions, or demonstrations of products, often highlighting key features or benefits. Example: 'This phone has an amazing camera and a long-lasting battery.' or 'Check out this sleek new smartwatch!'",
        "Navigation: This category includes statements related to movement, direction, or transitioning from one place to another. Example: 'Let me guide you to the help section.' or 'Click here to go to your profile.'",
        "Error Handling: This category contains statements addressing system failures, misunderstandings, or unexpected issues, often providing explanations or corrective guidance. Example: 'Sorry, something went wrong. Let me fix that for you.' or 'There was an issue with your request, please try again.'",
        "Speaking & Listening Mode: This category includes statements that indicate an ongoing conversation state, such as the system actively responding, prompting for input, or acknowledging a request. It does NOT include expressions of empathy or emotions. Example: 'I’m listening, please continue.' or 'Could you clarify that for me?'"
    ]

    sentiment_labels = ["positive", "negative", "neutral"]

    count = 0
    for text in texts:
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

        json_string = json.dumps(output, ensure_ascii=False, indent=2)

        # 顯示輸出
        print(json_string)

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")

        count += 1
        # if count % 5 == 0:
        #     with open("output_DeBERTa_4.txt", "a", encoding="utf-8") as file:
        #         file.write(f"Processing sentence: {text}\n")
        #         file.write(json_string)
        #         file.write(f"\nExecution time: {end_time - start_time:.2f} seconds\n")
        #         file.write("=======================================\n")
        # else:
        #     with open("output_DeBERTa_4.txt", "a", encoding="utf-8") as file:
        #         file.write(f"Processing sentence: {text}\n")
        #         file.write(json_string)
        #         file.write(f"\nExecution time: {end_time - start_time:.2f} seconds\n")

if __name__ == "__main__":
    test_sentences = [
        "If you have any questions, feel free to ask. Thank you for visiting, and have a great day!",
        "Thanks for stopping by! Hope to see you again soon!",
        "Take care and have a fantastic day!",
        "Goodbye! Let me know if you ever need any help.",
        "It was great assisting you! Wishing you all the best!",
        "Hey there! It’s great to see you!",
        "Welcome! Let me know if there’s anything I can assist you with.",
        "Good to have you here! How can I make your day better?",
        "Hi! I’m here to help. What can I do for you today?",
        "Hello and welcome! Feel free to ask me anything.",

        "I truly appreciate your patience, thank you so much!",
        "That must have been really challenging, I completely understand.",
        "Wow, that’s incredible news! I’m thrilled for you!",
        "I can see how much this means to you, and I’m here to support you.",
        "That’s such a wonderful achievement, you should be proud!",

        "Here’s our brand-new smartwatch, packed with cutting-edge health tracking features.",
        "Check out this high-performance gaming laptop, built for speed and power.",
        "This stylish wireless headset offers crystal-clear sound and all-day comfort.",
        "I’d be happy to help you explore our latest products! Can I show you something specific today?",
        "This model offers amazing features that can really improve your daily experience. Would you like to know more?",

        "Let me guide you to the support section for further assistance.",
        "Click here to move to your account settings.",
        "Let me show you how to get to the product details.",
        "You can find the section you need by following this link.",
        "Let me direct you to the help center for more information.",

        "I’m sorry, but it seems there’s a system issue at the moment. ",
        "Let me quickly resolve that for you.",
        "Apologies, we ran into an unexpected issue. ",
        "Please try again later.",
        "An error has occurred. ",
        "Let me fix that for you right away.",
        "We’re experiencing technical difficulties. ",
        "Please be patient as we resolve the issue.",
        "Something isn’t working as expected. ",
        "I’ll look into it now.",

        "I'm here. You can continue whenever you're ready.",
        "Just to confirm, are you asking about the schedule?",
        "Let me know if you need any clarification.",
        "I see what you mean. What would you like to do next?",
        "Understood. Let's move on to the next step.",
    ]
    main(test_sentences)
