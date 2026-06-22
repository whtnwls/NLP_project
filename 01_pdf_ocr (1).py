import os
from datetime import datetime

import fitz
import numpy as np
import easyocr

# =========================================================
# PDF 경로
# =========================================================

pdf_path = "자연어처리.pdf"

# =========================================================
# 결과 폴더 생성
# =========================================================

timestamp = datetime.now().strftime(
    "%y%m%d_%H%M"
)

output_dir = f"ocr_output_{timestamp}"

os.makedirs(
    output_dir,
    exist_ok=True
)

# =========================================================
# EasyOCR 모델
# =========================================================

reader = easyocr.Reader(
    ["ko", "en"],
    gpu=False
)

# =========================================================
# PDF 열기
# =========================================================

doc = fitz.open(pdf_path)

all_text = ""

# =========================================================
# 페이지별 OCR
# =========================================================

for page_idx in range(len(doc)):

    print("=" * 80)
    print(f"{page_idx+1}페이지 OCR 시작")
    print("=" * 80)

    page = doc[page_idx]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(5, 5),
        alpha=False
    )

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    result = reader.readtext(
        img,
        detail=0
    )

    page_text = []

    if result:

        for text in result:

            text = text.strip()

            if len(text) < 2:
                continue

            page_text.append(text)

    page_result = " ".join(page_text)

    all_text += (
        f"\n\n===== PAGE {page_idx+1} =====\n\n"
    )

    all_text += page_result

    print(
        f"{page_idx+1}페이지 완료"
    )

    print(
        f"추출 문장 수 : {len(page_text)}"
    )

# =========================================================
# OCR 결과 저장
# =========================================================

ocr_text_path = os.path.join(
    output_dir,
    "ocr_result.txt"
)

with open(
    ocr_text_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(all_text)

# =========================================================
# 완료
# =========================================================

print("\n" + "=" * 80)
print("OCR 완료")
print("=" * 80)
print(f"OCR 결과 : {ocr_text_path}")
print("=" * 80)