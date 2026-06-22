from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_TARGET_DIR = BASE_DIR / "data" / "raw" / "01.news_r" / "2~3sent"

json_files = list(RAW_TARGET_DIR.rglob("*.json"))

print(f"찾은 JSON 파일 개수: {len(json_files)}")

if len(json_files) == 0:
    print("JSON 파일이 없습니다")
    exit()

sample_file = json_files[0]

print("\n샘플 파일:")
print(sample_file)

with open(sample_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n최상위 타입:")
print(type(data))

if isinstance(data, dict):
    print("\n최상위 key:")
    print(data.keys())

    print("\n전체 구조 일부:")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:3000])

elif isinstance(data, list):
    print("\n리스트 길이:")
    print(len(data))

    if len(data) > 0:
        print("\n첫 번째 원소 타입:")
        print(type(data[0]))

        if isinstance(data[0], dict):
            print("\n첫 번째 원소 key:")
            print(data[0].keys())

        print("\n첫 번째 원소 구조 일부:")
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:3000])