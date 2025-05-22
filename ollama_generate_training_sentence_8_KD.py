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
    with open("KD_train_from_llama_5sentence_label_thinking_1.txt", "a", encoding="utf-8") as f:
        f.write(sentence + "\t" + ", ".join(matches) + "\n")


# ========== 發送請求至 LLaMA ==========
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": """
You are a helpful assistant that analyzes professional communication for embodied AI use.
Your task is to:
1. Summarize the thinking process of the text.
2. Identify the intention (positive/neutral/negative).
3. Identify the dominant current emotion.
4. Add motion annotation in the format [@timestamp_begin][ID] before parts of the text that would involve relevant physical expression or gestures.
5. List a motion_tag array describing each gesture for each [ID].

example:
input : "As we continue to enhance our virtual assistant's capabilities, I'm pleased to announce the launch of our new AI-powered language generator. This innovative tool has been integrated into our system to improve the quality and efficiency of our responses. "
output : 
{
  "thinking_process": "The speaker announces a new feature, explains its purpose, acknowledges a temporary issue, and reassures users with ongoing improvements.",
  "intention": "positive",
  "current_emotion": "optimism",
  "labeled_text": "As we continue to enhance our virtual assistant's capabilities, I'm [@timestamp_begin][1]pleased to announce the launch of our new AI-powered language generator. This [@timestamp_begin][2]innovative tool has been integrated into our system to improve the quality and efficiency of our responses.",
  "motion_tag": [
    {
      "ID": "1",
      "category": "Emotion Set",
      "description": "Smile and open one hand forward to express enthusiasm about the announcement"
    },
    {
      "ID": "2",
      "category": "Product Showcase",
      "description": "Present with both hands toward an imaginary product to highlight the new tool"
    }
  ]
}

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
    file_path = "KD_train_from_llama_5sentence_1000_3.txt"

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