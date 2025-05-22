import json

# 讀取 JSON 檔案
with open('MNLI_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 過濾掉 label 為 "neutral" 的項目
filtered_data = [item for item in data if item["label"] != "neutral"]

# 寫回新的 JSON 檔案
with open('filtered_data.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print("已成功過濾 neutral 項目，結果存入 filtered_data.json")