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
def main():
    output_lines = []

    # Step 1：先用 Gemini 生成一句顧客的問題（根據情境）
    scenario = """
    Please randomly generate a question.
    This question should be something a customer would ask when they enter a store.
    In addition to the question itself, also describe the scenario in which the question is asked.
    """
    prompt = get_customer_question_from_gemini(scenario)

    print(f"[Gemini 問句] {prompt}")

    # Step 2：將此問題丟給本地 LLaMA 處理
    response_text = get_llm_response(prompt)

    if response_text:
        cleaned = response_text.strip().replace('\n', '')
        print("=================================")
        print(f"[LLaMA 回覆] {cleaned}")
    else:
        cleaned = "Failed to get response"

    # Step 3：儲存結果
    with open("KD_train_from_llama_5sentence_3_1000_2.txt", "a", encoding="utf-8") as f:
        f.write(cleaned + "\n")


# ========== Gemini 生成顧客問題句子 ==========
def get_customer_question_from_gemini(scenario: str) -> str:
    model = genai.GenerativeModel("models/gemma-3-27b-it")

    response = model.generate_content(scenario)
    return response.text.strip()


# ========== 發送請求至 LLaMA ==========
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": "You are an AI Agent."},
        {"role": "user", "content": f"""
user question: {sentence}

Please respond to the user's question based on the given scenario, without writing 'Response:' or using colon.
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
    # for model in genai.list_models():
    #     print(model.name)

    n = 0
    while n < 1000:
        print(f"\n--- 第 {n + 1} 筆 ---")
        main()
        n += 1