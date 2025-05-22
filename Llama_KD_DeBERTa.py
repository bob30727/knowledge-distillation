import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    Trainer, TrainingArguments
)
from datasets import load_dataset
from transformers import AutoTokenizer

# **1. 設定本地端模型路徑**
TEACHER_MODEL_PATH = "./Llama-3.1-8B"  # LLaMA 3.1 教師模型（本地）
STUDENT_MODEL_PATH = "./deberta-v3-large-zeroshot-v1"  # DeBERTa 學生模型（本地）
NUM_LABELS = 7  # 你的分類標籤數量

# **2. 載入本地 Tokenizer**
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
# teacher_tokenizer = LlamaTokenizer.from_pretrained(TEACHER_MODEL_PATH)
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, legacy=False)

# **3. 載入本地模型**
teacher_model = LlamaForCausalLM.from_pretrained(TEACHER_MODEL_PATH, device_map="auto")
teacher_model.eval()  # 設為推理模式

student_model = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL_PATH, num_labels=NUM_LABELS)
student_model.to("cuda")

# **4. 載入資料集**
dataset = load_dataset("json", data_files={"train": "MNLI_train.json", "test": "MNLI_eval.json"})

# **5. 資料預處理**
def preprocess_function(examples):
    return student_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# **6. 產生教師模型的 Soft Label**
def generate_soft_labels(examples):
    inputs = teacher_tokenizer(examples["text"], return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(teacher_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = teacher_model(**inputs).logits[:, -1, :]  # 取最後一個 Token 的 logits
        soft_labels = torch.nn.functional.softmax(logits / 3.0, dim=-1)  # 設定溫度 T=3.0

    return {"soft_labels": soft_labels.cpu().tolist()}

dataset_with_soft_labels = encoded_dataset["train"].map(generate_soft_labels, batched=True)

# **7. 定義蒸餾 Loss（KL 散度 + Cross Entropy）**
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # KL Loss 比重
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")  # KL Loss
        self.ce_loss = nn.CrossEntropyLoss()  # 交叉熵 Loss

    def forward(self, student_logits, teacher_probs, labels):
        student_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = self.kl_loss(student_probs, teacher_probs)  # KL Loss
        ce_loss = self.ce_loss(student_logits, labels)  # 交叉熵 Loss
        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss  # 混合 Loss

# **8. 自訂 Trainer（加入蒸餾 Loss）**
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        soft_labels = inputs.pop("soft_labels")

        outputs = model(**inputs)
        logits = outputs.logits

        labels = labels.to(logits.device)
        soft_labels = torch.tensor(soft_labels).to(logits.device)

        loss_fct = DistillationLoss(alpha=0.7, temperature=3.0)
        loss = loss_fct(logits, soft_labels, labels)

        return (loss, outputs) if return_outputs else loss

# **9. 設定訓練參數**
training_args = TrainingArguments(
    output_dir="./distilled_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_total_limit=2,
)

trainer = CustomTrainer(
    model=student_model,
    args=training_args,
    train_dataset=dataset_with_soft_labels,
    eval_dataset=encoded_dataset["test"],
)

# **10. 開始訓練**
trainer.train()