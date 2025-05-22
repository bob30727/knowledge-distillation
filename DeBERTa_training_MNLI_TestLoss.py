import json
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# ✅ 讀取 JSON 訓練集
with open("MNLI_train_6.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

# ✅ 讀取 JSON 測試集
with open("MNLI_eval.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# ✅ 定義標籤對應
label_mapping = {"entailment": 0, "contradiction": 1}

# ✅ 轉換標籤格式，確保大小寫一致
for dataset in [train_data, eval_data]:
    for item in dataset:
        if "label" in item:
            label_lower = item["label"].lower()  # 轉換為小寫
            if label_lower in label_mapping:
                item["label"] = label_mapping[label_lower]
            else:
                raise ValueError(f"❌ 發現無效的標籤值: {item['label']}")

# ✅ 轉換成 Hugging Face Dataset
dataset_train = Dataset.from_list(train_data)
dataset_eval = Dataset.from_list(eval_data)

# ✅ 加載 Tokenizer
model_name = "./deberta-v3-large-zeroshot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Tokenize 數據
def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length")

tokenized_train_dataset = dataset_train.map(preprocess_function, batched=True)
tokenized_eval_dataset = dataset_eval.map(preprocess_function, batched=True)

# ✅ 重新命名標籤欄位
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_eval_dataset = tokenized_eval_dataset.rename_column("label", "labels")
tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

# ✅ 加載 DeBERTa 模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# ✅ 訓練參數
training_args = TrainingArguments(
    output_dir="./deberta_mnli_finetuned_6",
    evaluation_strategy="epoch",  # 每個 epoch 評估一次
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=False,
    logging_dir="./logs",
    logging_steps=1,
    push_to_hub=False
)

# ✅ 確保批次資料能夠被正常處理
data_collator = DataCollatorWithPadding(tokenizer)

# ✅ 設定 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # 加入測試集
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ 開始微調
trainer.train()

# ✅ 訓練完成後，執行測試集評估
eval_results = trainer.evaluate()

# ✅ 顯示測試結果
test_loss = eval_results.get("eval_loss", "N/A")
print("📊 測試集評估結果:", eval_results)
print(f"📉 Test Loss: {test_loss}")

# ✅ 儲存模型
model.save_pretrained("./deberta_mnli_finetuned_6")
tokenizer.save_pretrained("./deberta_mnli_finetuned_6")
