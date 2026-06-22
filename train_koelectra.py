# =========================================================
# VSCode 최종 안정화 버전
# Kernel Crash 방지 + GPU 학습 가능
# 실행 방법:
# python train_koelectra.py
# =========================================================

import json
import re
import gc
import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from sentence_transformers import (
    SentenceTransformer,
    util
)

# =========================================================
# GPU 확인 및 작업 경로 체크
# =========================================================

print("=" * 80)
print("환경 검증 및 GPU 확인")
print("=" * 80)

print("현재 파이썬 작업 경로(PWD):", os.getcwd())
print("CUDA 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
print("=" * 80)

# =========================================================
# device 설정
# SBERT는 CPU
# KoELECTRA는 GPU
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# 메모리 정리
# =========================================================

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =========================================================
# 모델 설정
# =========================================================

MODEL_NAME = "monologg/koelectra-small-v3-discriminator"

# =========================================================
# SBERT 모델
# CPU 사용 (중요)
# =========================================================

similarity_model = SentenceTransformer(
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    device="cpu"
)

# =========================================================
# 중요 패턴
# =========================================================

important_patterns = [
    "중요", "핵심", "반드시", "시험", "출제", "필수", 
    "정의", "의미", "란", "특징", "역할", "구조", 
    "원리", "과정", "개념", "주의"
]

# =========================================================
# hard negative 패턴
# =========================================================

hard_negative_patterns = [
    "최근", "설명하였다", "발표하였다", "소개하였다", 
    "진행하였다", "언급하였다", "보고하였다"
]

# =========================================================
# 문장 분리
# =========================================================

def split_sentences(text):
    if text is None:
        return []

    text = str(text)
    sentences = re.split(r'(?<=[.!?다])\s+', text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 5
    ]

# =========================================================
# 핵심 개념 추출
# =========================================================

def extract_keywords(sentence):
    keywords = []
    patterns = [
        r"[가-힣A-Za-z0-9]+ 모델",
        r"[가-힣A-Za-z0-9]+ 알고리즘",
        r"[가-힣A-Za-z0-9]+ 함수",
        r"[가-힣A-Za-z0-9]+ 구조",
        r"[가-힣A-Za-z0-9]+ 기법",
        r"[가-힣A-Za-z0-9]+ 시스템",
        r"[가-힣A-Za-z0-9]+ 네트워크",
        r"[가-힣A-Za-z0-9]+ 학습",
        r"[가-힣A-Za-z0-9]+ 분석",
        r"[A-Z]{2,}"
    ]

    for pattern in patterns:
        found = re.findall(pattern, sentence)
        for word in found:
            if word not in keywords:
                keywords.append(word)

    return keywords

# =========================================================
# 데이터셋 생성
# =========================================================

def build_dataset(jsonl_path, max_data=300):
    # 파일이 진짜 존재하는지 체크하는 안전장치
    if not os.path.exists(jsonl_path):
        print(f"❌ 에러: [{jsonl_path}] 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")
        sys.exit(1)

    dataset = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_data:
                break

            item = json.loads(line)
            input_text = item["input"]
            summary_text = item["output"]

            sentences = split_sentences(input_text)
            if len(sentences) == 0:
                continue

            # -------------------------------------------------
            # SBERT 임베딩
            # -------------------------------------------------
            sentence_embeddings = similarity_model.encode(
                sentences,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            summary_embedding = similarity_model.encode(
                summary_text,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # -------------------------------------------------
            # 문장별 라벨 생성
            # -------------------------------------------------
            for s_idx, sentence in enumerate(sentences):
                similarity = util.cos_sim(
                    sentence_embeddings[s_idx],
                    summary_embedding
                ).item()

                rule_based = any(p in sentence for p in important_patterns)
                hard_negative = any(p in sentence for p in hard_negative_patterns)

                label = 1 if (
                    (similarity >= 0.35 or rule_based)
                    and not hard_negative
                ) else 0

                dataset.append({
                    "text": sentence,
                    "label": label
                })

            # -------------------------------------------------
            # 메모리 정리
            # -------------------------------------------------
            del sentence_embeddings
            del summary_embedding
            gc.collect()

    return pd.DataFrame(dataset)

# =========================================================
# 데이터셋 생성 시작 (★ 경로 수정: koelectra_project/ 추가)
# =========================================================

print("\ntrain 데이터 생성 중...")
train_df = build_dataset("koelectra_project/train.jsonl", max_data=300)

print("valid 데이터 생성 중...")
valid_df = build_dataset("koelectra_project/valid.jsonl", max_data=100)

print("test 데이터 생성 중...")
test_df = build_dataset("koelectra_project/test.jsonl", max_data=100)

print("\n데이터 생성 완료")
print(f"train 개수: {len(train_df)}")
print(f"valid 개수: {len(valid_df)}")
print(f"test 개수 : {len(test_df)}")

# =========================================================
# 클래스 균형 맞추기
# =========================================================

positive_df = train_df[train_df["label"] == 1]
negative_df = train_df[train_df["label"] == 0]

negative_df = negative_df.sample(
    n=min(len(negative_df), len(positive_df) * 2),
    random_state=42
)

train_df = pd.concat([positive_df, negative_df])
train_df = train_df.sample(frac=1, random_state=42)

print("\n라벨 분포")
print(train_df["label"].value_counts())

# =========================================================
# Dataset 변환
# =========================================================

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

# =========================================================
# 토크나이저
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================================================
# 토큰화
# =========================================================

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# =========================================================
# torch format
# =========================================================

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =========================================================
# 모델 로드
# =========================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(device)

# =========================================================
# 평가 함수
# =========================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1
    }

# =========================================================
# 학습 옵션
# =========================================================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=20,
    fp16=False,
    load_best_model_at_end=True
)

# =========================================================
# Trainer
# =========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================================================
# 학습 시작
# =========================================================

print("\n")
print("=" * 80)
print("KoELECTRA 학습 시작")
print("=" * 80)

trainer.train()

# =========================================================
# 최종 평가
# =========================================================

results = trainer.evaluate(test_dataset)

print("\n")
print("=" * 80)
print("최종 성능")
print("=" * 80)

print(f"Accuracy : {results['eval_accuracy']:.4f}")
print(f"F1 Score : {results['eval_f1']:.4f}")
print(f"Loss      : {results['eval_loss']:.4f}")

# =========================================================
# 모델 저장
# =========================================================

SAVE_PATH = "./koelectra_keysentence_model"
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\n모델 저장 완료")

# =========================================================
# 예측 함수
# =========================================================

def predict_sentence(sentence):
    model.eval()
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][1].item()

    return pred, confidence

# =========================================================
# 긴 문장 테스트
# =========================================================

print("\n")
print("=" * 80)
print("긴 문장 기반 모델 성능 테스트")
print("=" * 80)

sample_text = """
딥러닝이란 인간의 신경망 구조를 모방하여 데이터를 학습하는 머신러닝 기법으로,
대량의 데이터를 활용해 이미지 분류, 자연어 처리, 음성 인식 등의 다양한 문제를 해결할 수 있다.

CNN 모델은 이미지의 공간적 특징을 효과적으로 추출할 수 있기 때문에
컴퓨터 비전 분야에서 매우 중요한 역할을 수행하며,
시험에서도 자주 출제되는 핵심 개념 중 하나이다.

Transformer 구조는 Self-Attention 메커니즘을 기반으로 문장 내 단어 간의 관계를 계산하며,
기존 RNN 기반 모델보다 병렬 처리가 가능하다는 장점이 있다.

오늘은 날씨가 좋아서 친구들과 함께 카페에 가서 커피를 마셨고,
저녁에는 영화를 보며 시간을 보냈다.

GPU는 대규모 행렬 연산을 병렬로 처리할 수 있기 때문에
딥러닝 모델 학습 속도를 크게 향상시킬 수 있으며,
CUDA 환경 설정은 실습에서 매우 중요하게 다뤄진다.
"""

sentences = split_sentences(sample_text)

for idx, sentence in enumerate(sentences):
    pred, confidence = predict_sentence(sentence)

    print(f"\n[{idx+1}] 문장")
    print(sentence)
    print(f"\n핵심 문장 확률: {confidence:.4f}")

    if confidence >= 0.40:
        print("\n예측 결과:")
        print("핵심 문장")
        keywords = extract_keywords(sentence)
        print("\n핵심 개념:")
        print(keywords)
    else:
        print("\n예측 결과:")
        print("일반 문장")

    print("-" * 80)