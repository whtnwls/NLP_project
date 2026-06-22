from pathlib import Path
import json
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_FILE = BASE_DIR / "data" / "output" / "train_data.jsonl"


def main():
    counter = Counter()
    total = 0

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            counter[item.get("doc_type", "")] += 1
            total += 1

    print(f"전체 데이터 수: {total}")
    print("\n문서 유형별 데이터 수:")
    for doc_type, count in counter.items():
        print(f"- {doc_type}: {count}")


if __name__ == "__main__":
    main()