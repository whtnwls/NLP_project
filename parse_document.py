from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PARSED_DIR = BASE_DIR / "data" / "parsed"

SUMMARY_FOLDER = "2~3sent"

TARGET_CATEGORIES = [
    "01.news_r",
    "02.briefing",
    "03.his_cul",
    "04.paper",
    "05.minute",
    "06.edit",
    "07.public",
    "08.speech",
    "09.literature",
    "10.narration",
]


def extract_text_from_json(data: dict):
    if not isinstance(data, dict):
        return None

    meta_acq = data.get("Meta(Acqusition)", {})
    meta_refine = data.get("Meta(Refine)", {})
    annotation = data.get("Annotation", {})

    return {
        "doc_id": meta_acq.get("doc_id", ""),
        "doc_category": meta_acq.get("doc_category", ""),
        "doc_type": meta_acq.get("doc_type", ""),
        "doc_name": meta_acq.get("doc_name", ""),
        "author": meta_acq.get("author", ""),
        "publisher": meta_acq.get("publisher", ""),
        "publisher_year": meta_acq.get("publisher_year", ""),
        "doc_origin": meta_acq.get("doc_origin", ""),

        "passage_id": meta_refine.get("passage_id", ""),
        "passage": meta_refine.get("passage", ""),
        "passage_cnt": meta_refine.get("passage_Cnt", ""),

        # 2~3sent 기준 요약문
        "summary": annotation.get("summary2", ""),
        "summary1": annotation.get("summary1", ""),
        "summary2": annotation.get("summary2", ""),
        "summary3": annotation.get("summary3", "")
    }


def parse_category(category_name: str):
    raw_target_dir = RAW_DIR / category_name / SUMMARY_FOLDER
    parsed_target_dir = PARSED_DIR / category_name / SUMMARY_FOLDER
    parsed_target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 파싱 시작: {category_name} ===")
    print(f"입력 폴더: {raw_target_dir}")
    print(f"출력 폴더: {parsed_target_dir}")

    if not raw_target_dir.exists():
        print(f"[건너뜀] 입력 폴더 없음: {raw_target_dir}")
        return 0, 0

    json_files = list(raw_target_dir.rglob("*.json"))
    print(f"찾은 JSON 파일 개수: {len(json_files)}")

    success_count = 0
    empty_count = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            parsed = extract_text_from_json(data)

            if parsed is None:
                continue

            if not parsed["passage"] or not parsed["summary"]:
                empty_count += 1

            relative_path = json_file.relative_to(raw_target_dir)
            output_path = parsed_target_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([parsed], f, ensure_ascii=False, indent=2)

            success_count += 1

        except Exception as e:
            print(f"[오류] {json_file.name}: {e}")

    print(f"[완료] {category_name}")
    print(f"성공 파일 수: {success_count}")
    print(f"본문/요약 빈 파일 수: {empty_count}")

    return success_count, empty_count


def main():
    print("=== 전체 문서 파싱 시작 ===")

    total_success = 0
    total_empty = 0

    for category in TARGET_CATEGORIES:
        success_count, empty_count = parse_category(category)
        total_success += success_count
        total_empty += empty_count

    print("\n=== 전체 문서 파싱 완료 ===")
    print(f"전체 성공 파일 수: {total_success}")
    print(f"전체 본문/요약 빈 파일 수: {total_empty}")


if __name__ == "__main__":
    main()