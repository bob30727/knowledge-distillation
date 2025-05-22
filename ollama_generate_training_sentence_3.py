import requests
import json
import time
import re
import random

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
    selected_categories = random.sample(categories, 5)
    prompt_parts = []

    for category in selected_categories:
        prompt_parts.append(f"「{category}」")

    prompt = f"Please refer to the above passage to generate English sentences corresponding to each of the following five categories: {', '.join(prompt_parts)}. combine the five generated sentences together and do not return them separately. Do not include phrases like 'Here are the five English sentences:'."
    response_text = get_llm_response(prompt)
    if response_text:
        cleaned = response_text.strip().replace('\n', '')
        print(f"{cleaned}")
    else:
        output_lines.append("Failed to get response")

    with open("KD_train_from_llama_5sentence.txt", "a", encoding="utf-8") as f:
        f.write(cleaned + "\n")



# **發送請求至 LLM**
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": """
Please generate an English sentence that matches the meaning according to the following category, as if it were spoken by an AI customer service assistant.
Refer to the sentences below to help you generate your own, but do not copy them, Mimic the way it speaks:

I'd be happy to help you find what you're looking for! We actually carry a limited selection of left-handed gardening tools, including hand pruners. Let me check in the back if we have any left-handed shears that might suit your needs. I've had customers ask about this before and it's been harder to find than you'd think, so I'm not surprised you're having trouble elsewhere.
I think I have some good news for you! We actually carry a few different brands of left-handed gardening shears that might interest you. Let me show you what we have in stock. (rummages through nearby shelves) Ah, yes... here are the Felco brand ones, and over here we have the Corona brand. Both of these are high-quality options that are designed with ergonomics in mind. The Felcos have a bit more curve to them, which might be more comfortable for left-handed gardeners, but the Coronas have an adjustable handle that can accommodate different hand sizes. Would you like to take a look at either of these?
We actually do carry left-handed gardening shears in our specialty tool section over by the plant care supplies. Let me show you exactly where they are so you can take a look and see if that feels more comfortable for you to use. We also have some adjustable-handled pruning tools that might be worth considering, as those can sometimes work well for gardeners who need extra flexibility with their tool grip. Would you like me to grab those for you?
We actually carry a limited selection of left-handed gardening shears from a specialty brand that caters specifically to gardeners with unique needs. I'd be happy to show you what we have! Let me just check in the back real quick to see if we have any available, and I can also check our inventory online to make sure we're not missing any that might be on order.In the meantime, would you like to take a look at some of our other tools? We have a few ergonomic handles and ambidextrous shears that might work for you too. It's just that the left-handed ones are a bit more specialized, so I want to make sure we're showing you the best options possible.Oh, and just to confirm, would you be interested in trying out some of our demo shears? We have a few different models available for customers to test out before making a purchase. That way, you can get a feel for which one works best for your hand and gardening style!
I'd be happy to help you find what you're looking for! Let me check our inventory real quick. We actually do carry left-handed gardening shears from a reputable brand that specializes in ergonomic tools for gardeners with specific needs like yours. May I show them to you? They're not as commonly stocked as right-handed ones, but we try to keep a selection on hand for customers who need them. Would you like me to grab them for you so you can take a look?

Just respond with English sentences, do not include quotation marks or any other special characters. 
Do not include phrases like "Here are the five English sentences" and do not label which category each sentence belongs to.

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
    while n < 100:
        print(n ," : ")
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