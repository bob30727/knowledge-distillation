import requests
import json
import time
import re

# 設定 Ollama API 端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

# 七大分類
categories = [
    "Greeting Group : welcome a user when they approach",
    "Farewell Set : bid farewell at the end of an interaction",
    "Emotion Set : express the user's emotions",
    "Idle Animations : Waiting for the user to provide the next instruction",
    "Product Showcase : showcase or display a product",
    "Navigation : guide the user through a process or moving from one location to another",
    "Error Handling : the system encounters an error",
    "Listening State : the system is listening to the user's input",
    "Talking State : transitional phrases used to introduce an explanation."
]

# **主函式**
def main():
    output_lines = []

    for category in categories:
        prompt = f"Please refer to the above passage to generate an English sentence corresponding to the category 「{category}」"
        match = re.match(r"^(.*?)(?=:)", category)
        response_text = get_llm_response(prompt)
        if response_text:
            cleaned = response_text.strip()
            # output_lines.append(cleaned)
            print(f"[{match.group(1).strip()}] {cleaned}")
        else:
            output_lines.append(f"[{match.group(1).strip()}] Failed to get response")

        # with open("KD_train_from_llama.txt", "a", encoding="utf-8") as f:
        #     f.write("[" + match.group(1).strip() + "] " + cleaned + "\n")


# **發送請求至 LLM**
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": """
Please generate an English sentence that matches the meaning according to the following category, as if it were spoken by an AI customer service assistant.
Refer to the sentences below to help you generate your own, but do not copy them, Mimic the way it speaks:

Hey there, champion! Glad you dropped by—this machine’s ready to change your game!
It’s got digital resistance, guided programs, and real-time feedback to push your limits.
Let me show you how to fire it up and start sweating smart.
Need help navigating the modes? I got you! Just follow me step by step.
No rush—I'll be standing by until you're pumped and ready to go.
Something not working as expected? Boom—I’m on it. We’ll fix it together.
You can talk to me anytime—I’m all ears and ready to roll.
Let me explain how this beast helps you crush your fitness goals.
You did great today—can’t wait to see you back in action!

Just respond with one English sentence, do not include quotation marks or any other special characters. Do not include phrases like 'Here is a sentence'.

"""},
        {"role": "user", "content": sentence}
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
    while n < 50:
        main()
        n += 1






# Please generate an English sentence that matches the meaning according to the following category, as if it were spoken by an AI customer service assistant.
# The theme is about someone entering a home appliance store and communicating with the customer service assistant.
#
# - Greeting Group: welcome a user when they approach.
# Example sentences:
# "Hello! How can I help you today?"
# "Welcome to our store, we're glad to have you here!"
# "Hi there, nice to meet you!"
#
# - Farewell Set: bid farewell at the end of an interaction.
# Example sentences:
# "Thanks for visiting, take care!"
# "Goodbye! Have a great day ahead!"
# "It was a pleasure helping you, see you next time!"
#
# - Emotion Set: express the user's emotions.
# Example sentences:
# "I’m so happy to see you!"
# "I understand how frustrating that must be."
# "That sounds amazing, I’m so excited for you!"
#
# - Idle Animations: periods of no user interaction.
# Example sentences:
# "The system is waiting for your input."
# "Please hold on, we're processing your request."
# "We're just getting things ready for you."
#
# - Product Showcase: showcase or display a product.
# Example sentences:
# "Let me show you our latest smartphone, featuring a 48MP camera."
# "This is our newest model, designed for long battery life."
# "Take a look at this sleek laptop with advanced features."
#
# - Navigation: moving from one location to another.
# Example sentences:
# "Let me take you to the checkout page."
# "I’ll guide you to the product you’re looking for."
# "Click here to navigate to your account settings."
#
# - Error Handling: the system encounters an error.
# Example sentences:
# "Sorry, something went wrong. Please try again."
# "Oops! It looks like we encountered an error."
# "There was an issue processing your request, please check again."
#
# - Listening State: the system is listening to the user's input.
# Example sentences:
# "I’m listening, please go ahead."
# "I’m waiting for your response."
# "Please tell me more, I’m here."
#
# - Talking State: the system's spoken response.
# Example sentences:
# "I’m explaining how it works."
# "Here’s what you need to know about this product."
# "Let me walk you through the process."
#
# Just respond with one English sentence, do not include quotation marks or any other special characters. Do not include phrases like 'Here is a sentence'.
# example:
# Welcome to our service!
# It was nice chatting with you, goodbye.
# I'm glad you're excited to use our service!