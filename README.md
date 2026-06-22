# NLP Study Note System

## 프로젝트 소개

본 프로젝트는 PDF 문서로부터 텍스트와 이미지를 추출하고, OCR(광학 문자 인식) 기술을 활용하여 이미지 내 텍스트까지 분석한 후 학습 노트를 생성하는 자연어처리 기반 시스템이다.

## 주요 기능

* PDF 문서 텍스트 추출
* PDF 이미지 추출
* OCR을 통한 이미지 내 텍스트 인식
* 텍스트 전처리
* 학습 데이터 생성
* KoELECTRA 기반 자연어 처리
* 학습 노트 자동 생성

## 사용 기술

* Python
* KoELECTRA
* PaddleOCR
* PyMuPDF
* Pandas
* Jupyter Notebook

## 프로젝트 구조

* `parse_document.py` : PDF 텍스트 및 이미지 추출
* `extract_image.py` : PDF 이미지 추출
* `ocr.py` : OCR 수행
* `preprocess_text.py` : 텍스트 전처리
* `make_dataset.py` : 학습 데이터 생성
* `train_koelectra.py` : KoELECTRA 모델 학습
* `NLP_StudyNote_System.ipynb` : 실험 및 개발 과정

## 시스템 동작 과정

1. PDF 문서 입력
2. 텍스트 및 이미지 추출
3. OCR을 통한 이미지 내 텍스트 인식
4. 텍스트 전처리
5. 데이터셋 생성
6. KoELECTRA 모델 학습
7. 학습 노트 생성

## 개발 환경

* Python 3.x
* VS Code
* GitHub
* PaddleOCR
* KoELECTRA

