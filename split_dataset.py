from pathlib import Path
import json
import random
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "output" / "train_data.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "output" / "dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALID_FILE = OUTPUT_DIR / "valid.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

SEED = 42


def load_jsonl(file_path):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_type_count(name, data):
    counter = Counter()

    for item in data:
        counter[item.get("doc_type", "")] += 1

    print(f"\n{name} 문서 유형별 개수")
    for doc_type, count in counter.items():
        print(f"- {doc_type}: {count}")


def main():
    print("=== 데이터셋 분리 시작 ===")
    print(f"입력 파일: {INPUT_FILE}")
    print(f"입력 파일 존재 여부: {INPUT_FILE.exists()}")

    if not INPUT_FILE.exists():
        print("[오류] train_data.jsonl 파일이 없습니다.")
        return

    data = load_jsonl(INPUT_FILE)

    print(f"전체 데이터 수: {len(data)}")

    random.seed(SEED)
    random.shuffle(data)

    total = len(data)

    train_size = int(total * 0.8)
    valid_size = int(total * 0.1)

    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(valid_data, VALID_FILE)
    save_jsonl(test_data, TEST_FILE)

    print("\n=== 데이터셋 분리 완료 ===")
    print(f"train: {len(train_data)}개 -> {TRAIN_FILE}")
    print(f"valid: {len(valid_data)}개 -> {VALID_FILE}")
    print(f"test : {len(test_data)}개 -> {TEST_FILE}")

    print_type_count("train", train_data)
    print_type_count("valid", valid_data)
    print_type_count("test", test_data)


if __name__ == "__main__":
    main()