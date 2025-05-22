import requests


def fetch_urban_definition(term):
    url = f"https://api.urbandictionary.com/v0/define?term={term}"
    response = requests.get(url)
    data = response.json()

    results = []
    for item in data.get("list", []):
        definition = item.get("definition", "").replace('\r', '').replace('\n', ' ')
        example = item.get("example", "").replace('\r', '').replace('\n', ' ')
        results.append({
            "word": term,
            "definition": definition.strip(),
            "example": example.strip()
        })
    return results


# 測試
slang = "rizz"
definitions = fetch_urban_definition(slang)
for d in definitions:
    print(f"Word: {d['word']}\nDefinition: {d['definition']}\nExample: {d['example']}\n---")
