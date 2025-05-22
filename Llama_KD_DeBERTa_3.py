import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests
import json
import re

# 模型與路徑設定
TEACHER_MODEL_PATH = "llama3.1"
STUDENT_MODEL_PATH = "./deberta-v3-large-zeroshot-v1"
NUM_LABELS = 7
TXT_FILE_PATH = "./KD_train.txt"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 載入學生模型
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
student_model = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL_PATH,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)
torch.nn.init.xavier_uniform_(student_model.classifier.weight)
torch.nn.init.zeros_(student_model.classifier.bias)

# 類別標籤
CATEGORIES = [
    "Emotion Set",
    "Interaction Openers & Closers",
    "Idle Animations",
    "Product Showcase",
    "Navigation",
    "Error Handling",
    "Speaking & Listening Mode"
]

# 讀取資料
def load_sentences_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = set(line.strip() for line in f.readlines() if line.strip())
    return list(sentences)

# 產生 prompt
def format_prompt(sentence: str) -> str:
    return f"""
the sentence is : "{sentence}"
"""

# 呼叫 Ollama API 取得 LLaMA 回應
def get_llm_response(prompt: str) -> str:
    messages = [
        {"role": "system", "content": """
        Assist in identifying the similarity of the input sentence according to the instructions in these seven categories.
        predict its probability distribution over the following categories:
        - Emotion Set
        - Interaction Openers & Closers
        - Idle Animations
        - Product Showcase
        - Navigation
        - Error Handling
        - Speaking & Listening Mode
        
        Output probabilities as a space-separated list of 7 numbers summing to 1.
        Only return the probability number.
        such as:
        [0.7, 0.2, 0.0, 0.05, 0.15, 0.05, 0.05]
        """},
        {"role": "user", "content": prompt}
    ]

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": TEACHER_MODEL_PATH, "messages": messages, "temperature": 0},
            timeout=30,
            stream=True
        )
        full_response = ""
        for line in response.iter_lines():
            if line:
                line_json = json.loads(line.decode("utf-8"))
                full_response += line_json.get("message", {}).get("content", "")
        return full_response.strip()

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return ""

# 教師模型 soft labels
def get_teacher_soft_labels(sentence: str) -> torch.Tensor:
    prompt = format_prompt(sentence)
    response_text = get_llm_response(prompt)
    print(f"🧠 LLaMA Response: \n{response_text}")
    print("===========================================")

    try:
        probs = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", response_text)))
        print(probs)
        print(abs(sum(probs) - 1.0))
        print(len(probs))
        if len(probs) == NUM_LABELS and abs(sum(probs) - 1.0) < 0.2:
            tensor_probs = torch.tensor(probs)
            print(f"✅ Soft Labels: {tensor_probs}")
            return tensor_probs
    except ValueError:
        pass

    print("⚠️ 解析失敗，回傳均勻分布")
    return torch.ones(NUM_LABELS) / NUM_LABELS

# 知識蒸餾訓練
def knowledge_distillation(sentences, use_soft_labels=True):
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-5)

    for step, premise in enumerate(sentences):
        inputs = student_tokenizer(premise, return_tensors="pt")
        student_logits = student_model(**inputs).logits

        if use_soft_labels:
            teacher_probs = get_teacher_soft_labels(premise).unsqueeze(0)
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction="batchmean"
            )
        else:
            teacher_label = torch.argmax(get_teacher_soft_labels(premise))
            loss = F.cross_entropy(student_logits, teacher_label.unsqueeze(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {step}: Loss = {loss.item()}")

    print("✅ 蒸餾訓練完成！")

# 開始訓練
if __name__ == "__main__":
    train_sentences = load_sentences_from_txt(TXT_FILE_PATH)
    knowledge_distillation(train_sentences, use_soft_labels=True)
    student_model.save_pretrained("./trained_student_model")
    student_tokenizer.save_pretrained("./trained_student_model")
    print("💾 學生模型已儲存至 ./trained_student_model")