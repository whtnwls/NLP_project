# NLP Study Note System

## 프로젝트 소개

본 프로젝트는 PDF 문서의 텍스트를 추출하고 전처리한 후, NLP 모델을 활용하여 학습 노트를 생성하는 시스템이다.

## 주요 기능

* PDF 문서 텍스트 추출
* 텍스트 전처리
* 학습 데이터 생성
* KoELECTRA 기반 학습
* 학습 노트 생성 및 저장

## 사용 기술

* Python
* KoELECTRA
* Jupyter Notebook
* NLP (Natural Language Processing)

## 프로젝트 구조

* `parse_document.py` : PDF 문서 분석
* `preprocess_text.py` : 텍스트 전처리
* `make_dataset.py` : 학습 데이터 생성
* `train_koelectra.py` : KoELECTRA 모델 학습
* `NLP_StudyNote_System.ipynb` : 전체 실험 과정

## 실행 방법

1. 텍스트 전처리

```bash
python preprocess_text.py
```

2. 데이터셋 생성

```bash
python make_dataset.py
```

3. 모델 학습

```bash
python train_koelectra.py
```

## 개발 환경

* Python 3.x
* VS Code
* GitHub
