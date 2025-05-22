import requests
import json
import time
import re
import random
import google.generativeai as genai

# ========== 設定 ==========
# LLaMA 本機 API
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

# Gemini API 金鑰
genai.configure(api_key="AIzaSyAlWLZBKWQdRt29odex1kwAJnn1ZBsaU00")


# ========== 主函式 ==========
def main(sentence: str):
    output_lines = []

    # Step 2：將此問題丟給本地 LLaMA 處理
    response_text = get_llm_response(sentence)

    if response_text:
        cleaned = response_text.strip().replace('\n', '')
        print("=================================")
        print(f"[LLaMA 回覆] {cleaned}")
    else:
        cleaned = "Failed to get response"

    # Step 3：儲存結果
    matches = re.findall(r'"(.*?)"', cleaned)
    print(f"[清理過後] {matches}")
    with open("KD_train_from_llama_label_Confidence_Score.txt", "a", encoding="utf-8") as f:
        f.write(sentence + "\t" + ", ".join(matches) + "\n")

# ========== 發送請求至 LLaMA ==========
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": """
You are a professional actor skilled in stage presence, body language, and emotional expression.
Your task is to annotate key parts of a spoken sentence with physical gestures that would enhance the delivery if it were performed live on stage or in a video.

For each key moment, add annotations next to the most suitable word for the action, in square brackets with the following format:
[Gesture name, physical movement description, emotional tone]

Only annotate the most important words or phrases. Do not overuse gestures — they should highlight meaning, not clutter the message.

After completing the annotation, provide a confidence score (between 0.0 and 1.0) indicating how confident you are in your annotation quality. Format your output as:

Annotated Sentence: "<your full annotated version of the sentence>"
Confidence Score: <confidence between 0.0 and 1.0>

Example:
Input: "As we continue to enhance our virtual assistant's capabilities, I'm pleased to announce the launch of our new AI-powered language generator."
Output: As we continue to enhance our virtual assistant's capabilities [Open Hands, palms facing up and moving outward slightly, confident], I'm pleased [Hand to heart, gentle nod, warm] to announce the launch of our new AI-powered language generator [Presenting Gesture, one hand extended forward like revealing something new, enthusiastic].
Confidence Score: 0.92

"""},
        {"role": "user", "content": f"""
user question: {sentence}
"""}
    ]

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "messages": messages, "temperature": 1.9},
            timeout=15,
            stream=True
        )

        response.raise_for_status()
        response_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    data = json.loads(line)
                    message_content = data.get("message", {}).get("content", "")
                    if message_content:
                        response_text += message_content
                except json.JSONDecodeError as e:
                    print(f"JSON 解析錯誤: {e}")

        return response_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"API 請求失敗: {e}")
        return ""


# ========== 執行程式 ==========
if __name__ == "__main__":
    # 檔案路徑
    file_path = "KD_train_from_llama_5sentence_1000.txt"

    # 設定最大處理筆數（可選）
    max_samples = 1000

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            sentence = line.strip()
            print(f"\n--- 第 {idx + 1} 筆 ---")
            print(sentence)
            print("================================")
            main(sentence)