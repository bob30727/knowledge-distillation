import json


def extract_unique_premises(json_file, output_txt):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_premises = set(item["premise"] for item in data)  # 去除重複的 premise

    with open(output_txt, "w", encoding="utf-8") as f:
        for premise in unique_premises:
            f.write(premise + "\n")

    print(f"✅ 已存入 {output_txt}，共 {len(unique_premises)} 條句子。")


# 執行轉換
extract_unique_premises("MNLI_train.json", "KD_train.txt")