from pathlib import Path
import json
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "train_data.jsonl"


def main():
    print("=== 전체 학습 데이터 생성 시작 ===")
    print(f"입력 폴더: {PROCESSED_DIR}")
    print(f"입력 폴더 존재 여부: {PROCESSED_DIR.exists()}")
    print(f"출력 파일: {OUTPUT_FILE}")

    if not PROCESSED_DIR.exists():
        print("[오류] processed 폴더가 없습니다.")
        return

    json_files = list(PROCESSED_DIR.rglob("*.json"))

    print(f"찾은 JSON 파일 개수: {len(json_files)}")

    if len(json_files) == 0:
        print("[주의] 처리할 JSON 파일이 없습니다.")
        return

    total_count = 0
    skip_count = 0
    type_counter = Counter()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    data = [data]

                for doc in data:
                    input_text = doc.get("input_text", "").strip()
                    target_summary = doc.get("target_summary", "").strip()

                    if not input_text or not target_summary:
                        skip_count += 1
                        continue

                    doc_type = doc.get("doc_type", "")

                    item = {
                        "id": doc.get("passage_id", doc.get("doc_id", "")),
                        "doc_id": doc.get("doc_id", ""),
                        "doc_type": doc_type,
                        "title": doc.get("doc_name", ""),
                        "input": input_text,
                        "output": target_summary,
                        "input_char_count": doc.get("input_char_count", 0),
                        "summary_char_count": doc.get("summary_char_count", 0),
                        "input_sentence_count": doc.get("input_sentence_count", 0),
                        "summary_sentence_count": doc.get("summary_sentence_count", 0)
                    }

                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    total_count += 1
                    type_counter[doc_type] += 1

            except Exception as e:
                print(f"[오류] {json_file.name}: {e}")

    print("=== 전체 학습 데이터 생성 완료 ===")
    print(f"저장된 데이터 수: {total_count}")
    print(f"건너뛴 데이터 수: {skip_count}")

    print("\n문서 유형별 데이터 수:")
    for doc_type, count in type_counter.items():
        print(f"- {doc_type}: {count}")


if __name__ == "__main__":
    main()