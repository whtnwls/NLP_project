from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).resolve().parents[1]

PARSED_DIR = BASE_DIR / "data" / "parsed"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

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


def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    # 줄바꿈, 탭 제거
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # 여러 공백을 한 칸으로
    text = re.sub(r"\s+", " ", text)

    # 앞뒤 공백 제거
    text = text.strip()

    return text


def split_sentences(text):
    if not text:
        return []

    # 마침표, 물음표, 느낌표 뒤 공백 기준 분리
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def preprocess_doc(doc):
    passage = doc.get("passage", "")
    summary = doc.get("summary", "")

    cleaned_passage = clean_text(passage)
    cleaned_summary = clean_text(summary)

    passage_sentences = split_sentences(cleaned_passage)
    summary_sentences = split_sentences(cleaned_summary)

    return {
        "doc_id": doc.get("doc_id", ""),
        "doc_category": doc.get("doc_category", ""),
        "doc_type": doc.get("doc_type", ""),
        "doc_name": doc.get("doc_name", ""),
        "passage_id": doc.get("passage_id", ""),

        # 모델 입력
        "input_text": cleaned_passage,

        # 모델 정답
        "target_summary": cleaned_summary,

        # 확인용
        "passage_sentences": passage_sentences,
        "summary_sentences": summary_sentences,
        "input_char_count": len(cleaned_passage),
        "summary_char_count": len(cleaned_summary),
        "input_sentence_count": len(passage_sentences),
        "summary_sentence_count": len(summary_sentences),
    }


def preprocess_category(category_name):
    parsed_target_dir = PARSED_DIR / category_name / SUMMARY_FOLDER
    processed_target_dir = PROCESSED_DIR / category_name / SUMMARY_FOLDER
    processed_target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 전처리 시작: {category_name} ===")
    print(f"입력 폴더: {parsed_target_dir}")
    print(f"출력 폴더: {processed_target_dir}")

    if not parsed_target_dir.exists():
        print(f"[건너뜀] 입력 폴더 없음: {parsed_target_dir}")
        return 0, 0

    json_files = list(parsed_target_dir.rglob("*.json"))
    print(f"찾은 JSON 파일 개수: {len(json_files)}")

    success_count = 0
    skip_count = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            processed_docs = []

            for doc in data:
                if not isinstance(doc, dict):
                    skip_count += 1
                    continue

                processed = preprocess_doc(doc)

                if not processed["input_text"] or not processed["target_summary"]:
                    skip_count += 1
                    continue

                processed_docs.append(processed)

            if not processed_docs:
                continue

            relative_path = json_file.relative_to(parsed_target_dir)
            output_path = processed_target_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_docs, f, ensure_ascii=False, indent=2)

            success_count += 1

        except Exception as e:
            print(f"[오류] {json_file.name}: {e}")

    print(f"[완료] {category_name}")
    print(f"성공 파일 수: {success_count}")
    print(f"건너뛴 문서 수: {skip_count}")

    return success_count, skip_count


def main():
    print("=== 전체 텍스트 전처리 시작 ===")

    total_success = 0
    total_skip = 0

    for category in TARGET_CATEGORIES:
        success_count, skip_count = preprocess_category(category)
        total_success += success_count
        total_skip += skip_count

    print("\n=== 전체 텍스트 전처리 완료 ===")
    print(f"전체 성공 파일 수: {total_success}")
    print(f"전체 건너뛴 문서 수: {total_skip}")


if __name__ == "__main__":
    main()