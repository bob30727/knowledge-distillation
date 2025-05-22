import requests
import json
import re
import time

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

# ===== 抓 Urban Dictionary 詞意與例句 =====
def fetch_urban_definition(term):
    url = f"https://api.urbandictionary.com/v0/define?term={term}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    if not data["list"]:
        return None
    top = data["list"][0]
    # 清理 [brackets]
    definition = re.sub(r'\[([^\]]+)\]', r'\1', top["definition"])
    example = re.sub(r'\[([^\]]+)\]', r'\1', top["example"])
    return definition.strip(), example.strip()

# ===== 組 prompt 給 LLaMA =====
def build_prompt(word, definition, example):
    return f"""
Word: {word}
Definition: {definition}
Example: {example}

Please generate 5 example sentences using the word "{word}" in the context of casual, helpful customer service replies.
Make sure your responses sound friendly, natural, and stay within a customer support scenario.

Respond in this format:
1. ...
2. ...
3. ...
4. ...
5. ...
"""

# ===== 發送 prompt 給 LLaMA-3 本地模型 =====
def get_llm_response(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You're an expert in English slang and sentence generation."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "messages": messages, "temperature": 1.2},
            timeout=15,
            stream=True
        )
        response.raise_for_status()
        response_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    data = json.loads(line)
                    msg = data.get("message", {}).get("content", "")
                    if msg:
                        response_text += msg
                except json.JSONDecodeError:
                    continue
        return response_text.strip()
    except requests.exceptions.RequestException as e:
        print(f"[LLaMA Error] {e}")
        return ""

def fetch_random_slang_terms(n=10):
    terms = set()
    while len(terms) < n:
        response = requests.get("https://api.urbandictionary.com/v0/random")
        if response.status_code != 200:
            continue
        data = response.json()
        for entry in data["list"]:
            term = entry["word"]
            if term and len(terms) < n:
                terms.add(term)
    return list(terms)

# ===== 主流程：每個詞都處理一次 =====
def main():
    words = fetch_random_slang_terms(n=100)  # 抓 10 個隨機詞

    for idx, word in enumerate(words):
        print(f"\n--- 第 {idx + 1} 個詞：{word} ---")
        result = fetch_urban_definition(word)
        if not result:
            print(f"[跳過] 找不到 {word}")
            continue

        definition, example = result
        prompt = build_prompt(word, definition, example)
        print(prompt)
        print("=============================")
        response = get_llm_response(prompt)
        print(response)
        print("==========================================================")

        if not response:
            print(f"[失敗] LLaMA 無回應：{word}")
            continue

        matches = re.findall(r'\d+\.\s*(.*)', response.strip())
        if not matches:
            print(f"[警告] 找不到有效例句格式：{word}")
            continue

        # 每句一行儲存
        with open("llama_generated_examples.tsv", "a", encoding="utf-8") as f:
            for sentence in matches:
                f.write(f"{word}\t{sentence.strip()}\n")

        print(f"[完成] {word} → {len(matches)} 句")
        time.sleep(1.5)

if __name__ == "__main__":
    main()
