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
    # print("==================================================================")
    # print("目前處理的語句是 : ", text)
    # print(f"預測類別: {result['labels'][0]} (信心度: {result['scores'][0]:.4f})\n")
    # print(f"預測類別: {result['labels'][1]} (信心度: {result['scores'][1]:.4f})\n")
    # print(f"預測類別: {result['labels'][2]} (信心度: {result['scores'][2]:.4f})\n")
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


def main(texts: list):
    model_path = "./deberta_mnli_finetuned_6"
    # model_path = "./deberta_mnli_finetuned_5"
    # model_path = "./deberta_mnli_finetuned_4"
    # model_path = "./deberta_mnli_finetuned_3"
    # model_path = "./deberta-v3-large-zeroshot-v1"

    # 載入模型
    nlp, classifier = load_models(model_path)

    # motion_labels = [
    #     "Interaction Openers & Closers: This category includes statements that mark the beginning or conclusion of an interaction, such as greetings, introductions, and farewells. Example: 'Hello! How can I assist you today?' or 'Goodbye! Hope to see you soon!'",
    #     "Emotion Set: This category covers expressions of emotional states, including happiness, sadness, excitement, frustration, and empathy. It includes statements that convey understanding or emotional support. Example: 'I’m so excited about this!' or 'That must have been really challenging, I completely understand.'",
    #     "Idle Animations: This category consists of statements related to moments of inactivity or waiting, often indicating a lack of immediate engagement. Example: 'Please hold on while I look that up.' or 'I’m waiting for your input.'",
    #     "Product Showcase: This category involves descriptions, promotions, or demonstrations of products, often highlighting key features or benefits. Example: 'This phone has an amazing camera and a long-lasting battery.' or 'Check out this sleek new smartwatch!'",
    #     "Navigation: This category includes statements related to movement, direction, or transitioning from one place to another. Example: 'Let me guide you to the help section.' or 'Click here to go to your profile.'",
    #     "Error Handling: This category contains statements addressing system failures, misunderstandings, or unexpected issues, often providing explanations or corrective guidance. Example: 'Sorry, something went wrong. Let me fix that for you.' or 'There was an issue with your request, please try again.'",
    #     "Speaking & Listening Mode: This category includes statements that indicate an ongoing conversation state, such as the system actively responding, prompting for input, or acknowledging a request. It does NOT include expressions of empathy or emotions. Example: 'I’m listening, please continue.' or 'Could you clarify that for me?'"
    # ]
    motion_labels = [
        "Interaction Openers & Closers: This category includes greetings and farewells.",
        "Emotion Set: Expressions of emotions such as happiness, sadness, and empathy.",
        "Idle Animations: Statements related to waiting or inactivity.",
        "Product Showcase: Descriptions and demonstrations of products.",
        "Navigation: Statements related to movement or direction.",
        "Error Handling: Statements addressing failures or misunderstandings.",
        "Speaking & Listening Mode: Indicating conversation state (listening or talking)."
    ]
    sentiment_labels = ["positive", "negative", "neutral"]

    # **標準答案對照表**
    ground_truth = {
        "I’m sorry, but it seems there’s a system issue at the moment. Let me quickly resolve that for you.": "Error Handling",
        "Apologies, we ran into an unexpected issue. Please try again later.": "Error Handling",
        "An error has occurred. Let me fix that for you right away.": "Error Handling",
        "We’re experiencing technical difficulties. Please be patient as we resolve the issue.": "Error Handling",
        "Something isn’t working as expected. I’ll look into it now.": "Error Handling",

        "I truly appreciate your patience, thank you so much!": "Emotion Set",
        "That must have been really challenging, I completely understand.": "Emotion Set",
        "Wow, that’s incredible news! I’m thrilled for you!": "Emotion Set",
        "I can see how much this means to you, and I’m here to support you.": "Emotion Set",
        "That’s such a wonderful achievement, you should be proud!": "Emotion Set",

        "Here’s our brand-new smartwatch, packed with cutting-edge health tracking features.": "Product Showcase",
        "Check out this high-performance gaming laptop, built for speed and power.": "Product Showcase",
        "This stylish wireless headset offers crystal-clear sound and all-day comfort.": "Product Showcase",
        "I’d be happy to help you explore our latest products! Can I show you something specific today?": "Product Showcase",
        "This model offers amazing features that can really improve your daily experience. Would you like to know more?": "Product Showcase",

        "If you have any questions, feel free to ask. Thank you for visiting, and have a great day!": "Interaction Openers & Closers",
        "Thanks for stopping by! Hope to see you again soon!": "Interaction Openers & Closers",
        "Take care and have a fantastic day!": "Interaction Openers & Closers",
        "Goodbye! Let me know if you ever need any help.": "Interaction Openers & Closers",
        "It was great assisting you! Wishing you all the best!": "Interaction Openers & Closers",

        "Hey there! It’s great to see you!": "Interaction Openers & Closers",
        "Welcome! Let me know if there’s anything I can assist you with.": "Interaction Openers & Closers",
        "Good to have you here! How can I make your day better?": "Interaction Openers & Closers",
        "Hi! I’m here to help. What can I do for you today?": "Interaction Openers & Closers",
        "Hello and welcome! Feel free to ask me anything.": "Interaction Openers & Closers",

        "Let me guide you to the support section for further assistance.": "Navigation",
        "Click here to move to your account settings.": "Navigation",
        "Let me show you how to get to the product details.": "Navigation",
        "You can find the section you need by following this link.": "Navigation",
        "Let me direct you to the help center for more information.": "Navigation",

        "I'm here. You can continue whenever you're ready.": "Speaking & Listening Mode",
        "Just to confirm, are you asking about the schedule?": "Speaking & Listening Mode",
        "Let me know if you need any clarification.": "Speaking & Listening Mode",
        "I see what you mean. What would you like to do next?": "Speaking & Listening Mode",
        "Understood. Let's move on to the next step.": "Speaking & Listening Mode",
    }
    ground_truth2 = {
        "I apologize for the inconvenience. I’m working on a solution right now.": "Error Handling",
        "Oops! Something went wrong on our end. I’ll get that sorted for you.": "Error Handling",
        "We’re facing a temporary glitch. I appreciate your patience as we fix it.": "Error Handling",
        "It looks like there’s a system hiccup. Let me handle it for you.": "Error Handling",
        "I’ll take care of this issue immediately. Thanks for waiting!": "Error Handling",

        "I truly appreciate your patience—it means a lot!": "Emotion Set",
        "That must have been tough. I completely understand how you feel.": "Emotion Set",
        "Wow, that’s fantastic news! I’m so happy for you!": "Emotion Set",
        "I can tell this is important to you. I’m here to assist however I can.": "Emotion Set",
        "That’s an amazing milestone! You should be really proud!": "Emotion Set",

        "Introducing our latest smartwatch, designed to keep you active and healthy!": "Product Showcase",
        "Discover our high-performance gaming laptop, built for ultimate speed.": "Product Showcase",
        "This premium wireless headset provides immersive sound and comfort.": "Product Showcase",
        "Looking for something specific? I’d love to help you explore our products!": "Product Showcase",
        "This model comes with fantastic features that enhance your daily life. Interested in learning more?": "Product Showcase",

        "If you ever need assistance, don’t hesitate to ask. Have a great day!": "Interaction Openers & Closers",
        "Thanks for visiting! I hope to assist you again soon!": "Interaction Openers & Closers",
        "Take care and have an amazing day ahead!": "Interaction Openers & Closers",
        "Goodbye! Reach out anytime if you need assistance.": "Interaction Openers & Closers",
        "It was a pleasure helping you! Wishing you the best!": "Interaction Openers & Closers",

        "Hey there! So happy to see you!": "Interaction Openers & Closers",
        "Welcome! Let me know how I can assist you today.": "Interaction Openers & Closers",
        "Glad to have you here! How can I make your experience better?": "Interaction Openers & Closers",
        "Hi there! I’m here to help. What do you need today?": "Interaction Openers & Closers",
        "Hello and welcome! Ask me anything you’d like!": "Interaction Openers & Closers",

        "Let me guide you to the right section for more details.": "Navigation",
        "Click here to access your account settings instantly.": "Navigation",
        "I’ll show you how to navigate to the product details page.": "Navigation",
        "You can find what you need by following this link.": "Navigation",
        "Let me direct you to our help center for more information.": "Navigation",

        "I’m here whenever you’re ready to continue.": "Speaking & Listening Mode",
        "Just confirming—are you asking about the schedule?": "Speaking & Listening Mode",
        "Let me know if you need me to clarify anything.": "Speaking & Listening Mode",
        "I see what you mean! What’s your next step?": "Speaking & Listening Mode",
        "Got it. Let’s proceed to the next step.": "Speaking & Listening Mode",
    }

    for text in texts:
        start_time = time.time()

        # **執行分類**
        motion_label = classify_text(classifier, text, motion_labels)
        motion = extract_motion(motion_label)
        intention = classify_text(classifier, text, sentiment_labels)
        modified_text = insert_timestamps(nlp, text)

        # **查找標準答案**
        ground_truth_motion = ground_truth.get(text, "Unknown")

        # **構建輸出 JSON**
        output = {
            "labeled_text": modified_text,
            "intention": intention,
            "motion_tags": [{
                "ID": 1,
                "motion": f"{motion}  #chatGPT答案: {ground_truth_motion}"
            }]
        }

        json_string = json.dumps(output, ensure_ascii=False, indent=2)
        print("測試問題 : ",text)
        print(json_string)
        print("====================================================================")

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    test_sentences = [
        "I’m sorry, but it seems there’s a system issue at the moment. Let me quickly resolve that for you.",
        "Apologies, we ran into an unexpected issue. Please try again later.",
        "An error has occurred. Let me fix that for you right away.",
        "We’re experiencing technical difficulties. Please be patient as we resolve the issue.",
        "Something isn’t working as expected. I’ll look into it now.",

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

        "Let me guide you to the support section for further assistance.",
        "Click here to move to your account settings.",
        "Let me show you how to get to the product details.",
        "You can find the section you need by following this link.",
        "Let me direct you to the help center for more information.",

        "I'm here. You can continue whenever you're ready.",
        "Just to confirm, are you asking about the schedule?",
        "Let me know if you need any clarification.",
        "I see what you mean. What would you like to do next?",
        "Understood. Let's move on to the next step.",
    ]
    test_sentences2 = [
        "I apologize for the inconvenience. I’m working on a solution right now.",
        "Oops! Something went wrong on our end. I’ll get that sorted for you.",
        "We’re facing a temporary glitch. I appreciate your patience as we fix it.",
        "It looks like there’s a system hiccup. Let me handle it for you.",
        "I’ll take care of this issue immediately. Thanks for waiting!",

        "I truly appreciate your patience—it means a lot!",
        "That must have been tough. I completely understand how you feel.",
        "Wow, that’s fantastic news! I’m so happy for you!",
        "I can tell this is important to you. I’m here to assist however I can.",
        "That’s an amazing milestone! You should be really proud!",

        "Introducing our latest smartwatch, designed to keep you active and healthy!",
        "Discover our high-performance gaming laptop, built for ultimate speed.",
        "This premium wireless headset provides immersive sound and comfort.",
        "Looking for something specific? I’d love to help you explore our products!",
        "This model comes with fantastic features that enhance your daily life. Interested in learning more?",

        "If you ever need assistance, don’t hesitate to ask. Have a great day!",
        "Thanks for visiting! I hope to assist you again soon!",
        "Take care and have an amazing day ahead!",
        "Goodbye! Reach out anytime if you need assistance.",
        "It was a pleasure helping you! Wishing you the best!",

        "Hey there! So happy to see you!",
        "Welcome! Let me know how I can assist you today.",
        "Glad to have you here! How can I make your experience better?",
        "Hi there! I’m here to help. What do you need today?",
        "Hello and welcome! Ask me anything you’d like!",

        "Let me guide you to the right section for more details.",
        "Click here to access your account settings instantly.",
        "I’ll show you how to navigate to the product details page.",
        "You can find what you need by following this link.",
        "Let me direct you to our help center for more information.",

        "I’m here whenever you’re ready to continue.",
        "Just confirming—are you asking about the schedule?",
        "Let me know if you need me to clarify anything.",
        "I see what you mean! What’s your next step?",
        "Got it. Let’s proceed to the next step.",
    ]
    test_sentences3 = [
        "I can tell this is important to you.",
        "Hello and welcome to our mobile section!",
    ]

    main(test_sentences3)
