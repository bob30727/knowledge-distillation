# from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
#
# model_path = "./deberta-v3-large"  # 本地模型路徑
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
#
# # 手動設定 label2id
# model.config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
# model.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
#
# classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
#
# text = "Can you help me with this task?"
# labels = [
#     "Greeting", "Question", "Statement", "Opinion/Comment",
#     "Suggestion/Invitation", "Response/Reply", "Emotion/Feeling",
#     "Request", "Apology/Gratitude", "Command/Instruction"
# ]
#
# result = classifier(text, labels)
#
# # 取得最有可能的分類
# best_label = result["labels"][0]  # 最高機率的分類
# best_score = result["scores"][0]  # 最高機率的分數
#
# # 輸出結果
# print(f"Predicted Category: {best_label} (Confidence: {best_score:.4f})")



# from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
# import time
#
# # 指定本地模型路徑
# model_path = "./deberta-v3-large"  # 確保這個資料夾存在並包含模型權重
#
# # 加載模型與 tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# # model = AutoModelForSequenceClassification.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path)
#
# # # 建立 Zero-Shot Classification Pipeline
# # classifier = pipeline(
# #     "zero-shot-classification",
# #     model=model,
# #     tokenizer=tokenizer,
# #     device=0,  # 若有 GPU 可用 device=0
# #     truncation=True
# # )
#
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#
# test_sentences = [
#     "Hey! How are you today?",  # Greeting
#     "Do you know where the nearest bus stop is?",  # Question
#     "I'm sorry, I can't make it to the meeting.",  # Apology/Gratitude
#     "Let's go grab some coffee later.",  # Suggestion/Invitation
#     "No, I don’t want to go out today.",  # Response/Reply
# ]
# # 測試文本
# text = "No, I don’t want to go out today."
#
# # 定義分類標籤
# labels = [
#     "Greeting", "Question", "Statement", "Opinion/Comment",
#     "Suggestion/Invitation", "Response/Reply", "Emotion/Feeling",
#     "Request", "Apology/Gratitude", "Command/Instruction"
# ]
#
# start_time = time.time()
# # 執行分類
# result = classifier(text, labels)
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
#
# # 取得最有可能的分類
# best_label = result["labels"][0]  # 最高機率的分類
# best_score = result["scores"][0]  # 最高機率的分數
#
# # 輸出結果
# print(f"Predicted Category: {best_label} (Confidence: {best_score:.4f})")


from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time

# question = input("Question: ")

# 指定本地模型路徑
model_path = "./DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# bart-large-mnli
# deberta-v3-large-zeroshot-v1
# nli-deberta-v3-large 檔案很大
# xlm-roberta-large-xnli
# DeBERTa-v3-large-mnli-fever-anli-ling-wanli

# 加載本地模型
classifier = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path)

test_sentences = [
    "Hey! How are you today?",  # Social Obligations Management
    "Do you know where the nearest bus stop is?",  #Task Management
    "Let's go grab some coffee later.",  #Task Management
    "No, I don’t want to go out today.",  #Own Communication Management
    "I'm sorry, I can't make it to the meeting.",  #Social Obligations Management
    "It is 3 PM now.", #Time Management
    "I think this movie is great.", #Auto-Feedback
    "Do you like coffee?", #Task Management
    "Do you want this?", #Task Management
    "Do you want to go out? No, I don’t want to go out today.",
]


text = "I’d be happy to help you explore our latest products!"


# 定義分類標籤
# labels = [
#     "Greeting", "Question", "Statement", "Opinion/Comment",
#     "Suggestion/Invitation", "Response/Reply", "Emotion/Feeling",
#     "Request", "Apology/Gratitude", "Command/Instruction"
# ]

# labels = [
#     "Statements", "Questions", "Responses", "Directives",
#     "Commissives", "Expressives", "Social Obligations",
#     "Auto-Feedback", "Allo-Feedback"
# ]

# 36 個細分類別
# labels = [
#     "Fact-stating", "Opinion-stating",  # 陳述
#     "Yes/No Question", "Choice Question", "Wh-Question", "Declarative Question", "Check Question",  # 問題
#     "Confirm", "Disconfirm", "Answer", "Partial Answer",  # 回應
#     "Command", "Request", "Suggestion", "Offer", "Warning",  # 指示
#     "Promise", "Threat", "Offer to Commit", "Refusal",  # 承諾
#     "Happy", "Sad", "Angry", "Surprised",  # 情感
#     "Apology", "Thanks", "Greeting", "Farewell", "Welcome",  # 社交義務
#     "Understanding", "Not Understanding", "Acknowledgment",  # 自我回饋
#     "Encouragement", "Disapproval", "Clarification Request", "Repetition Request"  # 他人回饋
# ]

# labels = [
#     "Task Management - Involves tasks related to the main conversation, such as asking for information or giving advice.",
#     "Turn Management - Refers to the control of turns in a conversation, such as requesting to speak or ending a statement.",
#     "Time Management - Pertains to the management of time in a conversation, like setting a time or asking about a schedule.",
#     "Discourse Structuring - Organizes the structure of the conversation, such as introducing new topics or summarizing content.",
#     "Own Communication Management - Managing your own communication, such as correcting a mistake or rephrasing.",
#     "Partner Communication Management - Influencing the way your conversation partner communicates, like asking them to slow down or speak louder.",
#     "Auto-Feedback - Reflects the speaker's understanding of their own output, such as confirming if they've expressed themselves clearly.",
#     "Allo-Feedback - Reflects the understanding of the other participant in the conversation, such as asking for repetition or expressing understanding.",
#     "Social Obligations Management - Relates to social niceties in conversation, like greetings, apologies, or thanks."
# ]

# labels = [
#     "Task Management - Involves exchanging information related to the main task of the conversation, such as asking questions, giving advice, or responding to task-related queries. Examples:\n1. 'Can you help me find a good restaurant nearby?'\n2. 'You should try restarting your computer to fix the issue.'\n3. 'What’s the best way to learn a new language?'\n4. 'I think you should double-check your work before submitting it.'\n5. 'Do you know how to use this software?'",
#     "Turn Management - Refers to regulating the flow of conversation, such as requesting to speak, yielding the floor, interrupting, or signaling the end of a turn. Examples:\n1. 'Can I say something about that?'\n2. 'Go ahead, I’ll let you finish first.'\n3. 'Sorry to interrupt, but I have an important update.'\n4. 'Hold on, let me finish my point.'\n5. 'I think it's your turn to speak now.'",
#     "Time Management - Pertains to managing time-related aspects of a conversation, like scheduling meetings, asking about availability, or reminding about time. Examples:\n1. 'What time are we meeting tomorrow?'\n2. 'Let’s schedule the call for 3 PM.'\n3. 'How long will this meeting take?'\n4. 'I need to leave in 10 minutes.'\n5. 'Could you remind me about our appointment later?'",
#     "Discourse Structuring - Organizing and shaping the conversation, such as introducing new topics, transitioning between ideas, or summarizing discussions. Examples:\n1. 'Speaking of food, have you tried the new café downtown?'\n2. 'Before we move on, let me summarize what we’ve discussed.'\n3. 'That reminds me of something I wanted to ask you.'\n4. 'Let's switch topics and talk about the upcoming trip.'\n5. 'Just to clarify, are we all on the same page?'",
#     "Partner Communication Management - Influencing the way the conversation partner communicates, such as asking them to speak slower, louder, or clarify their statements. Examples:\n1. 'Could you speak a little slower?'\n2. 'Can you say that again, but more clearly?'\n3. 'Please lower your voice, we are in a library.'\n4. 'Can you explain that in simpler terms?'\n5. 'Could you repeat that last part? I didn’t catch it.'",
#     "Auto-Feedback - Reflects the speaker’s understanding of their own output, like confirming whether their statement is clear or acknowledging uncertainty. Examples:\n1. 'Does that make sense?'\n2. 'I hope I explained that well enough.'\n3. 'Let me know if that was confusing.'\n4. 'Am I being clear so far?'\n5. 'I think I got that right, but let me double-check.'",
#     "Allo-Feedback - Reflects the understanding of the other participant in the conversation, such as asking for repetition or confirming comprehension. Examples:\n1. 'Did you mean that we should meet at 5 PM?'\n2. 'Can you clarify what you just said?'\n3. 'So, if I understand correctly, you want to cancel the order?'\n4. 'Are you saying that the project is delayed?'\n5. 'Let me repeat that to make sure I got it right.'",
#     "Social Obligations Management - Relates to managing social niceties in the conversation, like greetings, apologies, expressing gratitude, or making polite requests. Examples:\n1. 'Hello! How have you been?'\n2. 'I’m really sorry for being late.'\n3. 'Thank you so much for your help!'\n4. 'It was great talking to you. Have a nice day!'\n5. 'Excuse me, could I ask you a question?'"
# ]

# labels = [
#     "Greeting Group : Actions used to welcome a user when they approach.",
#     "Farewell Set : Actions used to bid farewell at the end of an interaction.",
#     "Emotion Set : Actions that express the user's emotions.",
#     "Product Showcase : Actions used to showcase or display a product.",
#     "Navigation : Actions performed when moving from one location to another.",
#     "Error Handling : Actions shown when the system encounters an error.",
#     "Listening State : Actions indicating the system is listening to the user's input.",
#     "Talking State : Gestures that accompany the system's spoken response.",
#     "Idle Animations : Actions displayed during periods of no user interaction."
# ]

labels = [
    "positive",
    "negative",
    "neutral",
]

# for sentence in test_sentences:
#     result = classifier(sentence, labels)
#     print(f"句子: {sentence}")
#     print(f"預測類別: {result['labels'][0]} (信心度: {result['scores'][0]:.4f})\n")
#     print(f"預測類別: {result['labels'][1]} (信心度: {result['scores'][1]:.4f})\n")
#     print(f"預測類別: {result['labels'][2]} (信心度: {result['scores'][2]:.4f})\n")
#     print("==========================================================================")

start_time = time.time()
result = classifier(text, labels)
print(f"句子: {text}")
print(f"預測類別: {result['labels'][0]} (信心度: {result['scores'][0]:.4f})\n")
print(f"預測類別: {result['labels'][1]} (信心度: {result['scores'][1]:.4f})\n")
print(f"預測類別: {result['labels'][2]} (信心度: {result['scores'][2]:.4f})\n")
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# start_time = time.time()
# # 執行分類
# result = classifier(text, labels)
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
#
# # 取得分類結果並排序
# sorted_results = sorted(zip(result["labels"], result["scores"]), key=lambda x: x[1], reverse=True)
#
# # 顯示完整排名
# print("Classification Results:")
# for rank, (label, score) in enumerate(sorted_results, 1):
#     print(f"{rank}. {label} -  {score:.4f}")