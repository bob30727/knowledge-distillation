import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModelForCausalLM

# 設定模型路徑
TEACHER_MODEL_PATH = "./Llama-3.1-8B"
STUDENT_MODEL_PATH = "./deberta-v3-large-zeroshot-v1"
NUM_LABELS = 7  # 你的分類標籤數量
TXT_FILE_PATH = "./KD_train.txt"

# 載入教師模型 (Llama)
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)
teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_PATH, torch_dtype=torch.float16).cuda()

# 載入學生模型 (DeBERTa)
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
student_model = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL_PATH,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)

# **重新初始化分類層**
torch.nn.init.xavier_uniform_(student_model.classifier.weight)
torch.nn.init.zeros_(student_model.classifier.bias)

# 讀取 TXT 檔案，去除重複的句子
def load_sentences_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = set(line.strip() for line in f.readlines() if line.strip())
    return list(sentences)

# 訓練資料（從 TXT 讀取）
train_sentences = load_sentences_from_txt(TXT_FILE_PATH)

# 7 種分類標籤
hypothesis_options = [
    "Emotion Set",
    "Interaction Openers & Closers",
    "Idle Animations",
    "Product Showcase",
    "Navigation",
    "Error Handling",
    "Speaking & Listening Mode"
]

### **Llama 產生 7 類別的 soft label**
def get_teacher_soft_labels(premise):
    prompt = f"""Given the sentence: "{premise}", predict its probability distribution over the following categories:
- Emotion Set
- Interaction Openers & Closers
- Idle Animations
- Product Showcase
- Navigation
- Error Handling
- Speaking & Listening Mode

Output probabilities as a space-separated list of 7 numbers summing to 1.
"""
    inputs = teacher_tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = teacher_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
    generated_text = teacher_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

    # 從 prompt 中截取生成部分
    if prompt in generated_text:
        response = generated_text.split(prompt)[-1].strip()
    else:
        response = generated_text.strip()

    try:
        probs = list(map(float, response.strip().split()))
        if len(probs) == NUM_LABELS and sum(probs) > 0.99:
            return torch.tensor(probs)
    except ValueError:
        pass

    return torch.ones(NUM_LABELS) / NUM_LABELS

### **開始蒸餾訓練**
def knowledge_distillation(sentences, use_soft_labels=True):
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-5)

    for step, premise in enumerate(sentences):
        inputs = student_tokenizer(premise, return_tensors="pt")

        # **學生模型輸出**
        student_logits = student_model(**inputs).logits

        if use_soft_labels:
            # **方式 1：KL 散度蒸餾**
            teacher_probs = get_teacher_soft_labels(premise)  # 7 維 soft label
            print(teacher_probs)
            teacher_probs = teacher_probs.unsqueeze(0)  # (1, 7) batch 維度
            print(teacher_probs)
            print("==============================================")

            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction="batchmean"
            )
        else:
            # **方式 2：交叉熵分類蒸餾**
            teacher_label = torch.argmax(get_teacher_soft_labels(premise))
            teacher_label_tensor = torch.tensor([teacher_label])

            loss = F.cross_entropy(student_logits, teacher_label_tensor)

        # 反向傳播更新權重
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {step}: Loss = {loss.item()}")

    print("✅ 蒸餾訓練完成！")

# 進行蒸餾
knowledge_distillation(train_sentences, use_soft_labels=True)

# student_model.save_pretrained("./student_kd_model")
# student_tokenizer.save_pretrained("./student_kd_model")