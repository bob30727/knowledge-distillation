from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


# 標籤對應字典 (共 7 類)
label_mapping = {
    "Interaction Openers & Closers": 0,
    "Emotion Set": 1,
    "Idle Animations": 2,
    "Product Showcase": 3,
    "Navigation": 4,
    "Error Handling": 5,
    "Speaking & Listening Mode": 6
}

# 加載 JSON 訓練數據
dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})

# model_name = "microsoft/deberta-v3-base"
model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v1"
# model_name = "./DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# model_name = "./deberta-v3-large-zeroshot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 轉換標籤為數字
def convert_labels(example):
    if example["label"] in label_mapping:  # 確保標籤存在於映射中
        example["label"] = label_mapping[example["label"]]
    else:
        raise ValueError(f"未知標籤: {example['label']}")  # 增加錯誤檢查
    return example

dataset = dataset.map(convert_labels)

# Tokenize 數據
def preprocess_function(examples):
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True)
    encoding["label"] = examples["label"]  # 保留標籤
    return encoding

dataset = dataset.map(preprocess_function, batched=True)

# 指定標籤數量 (7 類)
num_labels = 7
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True  # 忽略 classifier 層大小不匹配
)

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./deberta-intent-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 加入 data_collator 以確保 batch 正確處理
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 建立 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator  # 確保 batch 正確處理
)

# 開始訓練
trainer.train()

# 保存模型與 tokenizer
model.save_pretrained("./deberta-intent-model_v3_zero")
tokenizer.save_pretrained("./deberta-intent-model_v3_zero")