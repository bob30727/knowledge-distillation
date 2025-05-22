import requests
import json
import time
import re
import random

# 設定 Ollama API 端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

# **主函式**
def main():
    output_lines = []
    prompt_parts = []

    prompt = "abc"
    response_text = get_llm_response(prompt)
    if response_text:
        cleaned = response_text.strip().replace('\n', '')
        print(f"{cleaned}")
    else:
        output_lines.append("Failed to get response")

    with open("KD_train_from_llama_5sentence_2.txt", "a", encoding="utf-8") as f:
        f.write(cleaned + "\n")



# **發送請求至 LLM**
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": "You are an article generator."},
        {"role": "user", "content": """
Please write me a paragraph in English.
This paragraph should consist of five connected sentences with logical flow.
It should be something that an AI agent might say.
It could be introducing a product, reporting a system error, or any other possible scenario—please help me create various contexts.
Do not output phrases like "Here are five different paragraphs."
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


# **執行程式**
if __name__ == "__main__":
    n = 0
    while n < 600:
        print(n ," : ")
        main()
        n += 1