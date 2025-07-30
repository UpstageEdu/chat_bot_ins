# Chat Bot (Insurance)

한국어 보험 QA 데이터를 기반으로 **GPT‑2**를 LoRA 방식으로 파인튜닝하고  
4‑bit 양자화까지 수행해 경량‑GPU 환경에서도 실시간 상담 챗봇을 돌릴 수 있게 한 프로젝트입니다.

## 사전 요구사항

- **Python**: 3.11.8 이상
- **GPU**: CUDA 지원 GPU (권장, 최소 4GB VRAM)

### 운영체제
- Windows 10/11
- macOS 10.15 이상
- Ubuntu 18.04 이상

## 주요 특징
* **LoRA 파인튜닝** – `trl.SFTTrainer` 활용으로 손쉽게 미세조정  
* **지침형(Alpaca) 프롬프트** 자동 생성 파이프라인  
* **4‑bit 양자화** – `BitsAndBytesConfig` 지원, VRAM 절감

## 프로젝트 구조
```text
chat_bot_ins/
├── train.py              # 모델 학습 스크립트
├── quantization.py       # 모델 양자화 스크립트
├── inference.py          # 학습된 모델로 추론을 실행하는 스크립트
├── requirements.txt      # 프로젝트에 필요한 패키지 및 버전 명시
├── setup.py              # 환경 설정 및 패키지 설치 스크립트
├── utils/                  # 학습 및 추론에 사용되는 유틸리티 모듈
│   ├── data.py           # 데이터셋 로딩 및 전처리
│   ├── metric.py         # 모델 성능 평가 지표
│   └── prompts.py        # 챗봇 프롬프트 형식 관리
└── data/                   # 학습용 데이터
    └── train.csv         # 실제 학습에 사용될 CSV 데이터 파일
```


## 빠른 시작
```bash
# 1. 저장소 클론 & 의존성 설치
$ git clone <repository-url>
$ cd tox_ko_classification
$ python setup.py

# 2. 기본 설정으로 학습 (GPU 권장)
$ python train.py  # 약 30분–2시간

# 3. 4‑bit 양자화 (선택)
$ python quantization.py 

# 4. 단일 문장 추론
$ python inference.py
```

## 모델 학습
```bash
# data/train.csv (columns: instruction,input,output) 준비 후
python train.py
```
결과 LoRA 어댑터 → `checkpoints/gpt2-lora/`

## 4‑bit 양자화
```bash
python quantization.py   # LoRA → 4-bit 모델 저장
```

경로 오류 발생 시, 아래 경로를 확인 후 `quantization.py` 파일에서 경로를 수정해주세요!
```
%ls model-checkpoints/gpt2-lora/
```

```
def main():
    # 경로 설정
    base_model_name = "gpt2"
    adapter_path = "checkpoints/gpt2-lora/checkpoint-100" # 이 부분에서 경로 설정!
    ...
```

## 추론
```bash
python inference.py      # 샘플 질문에 대한 응답 출력
```
## Dataset

| 파일 | 컬럼 | 설명 |
|------|------|------|
| `data/train.csv`  | `instruction`, `input`, `output` | 보험 관련 질문·문맥·모범답변 |

* **instruction** – 사용자의 핵심 질문 (예: “실손보험과 종합보험의 차이는?”)  
* **input** – 선택적 추가 문맥 (대부분 공란)  
* **output** – 약관/법령을 반영한 정답형 답변  


---

## Code Walk‑through

### `utils/`

| 파일 | 역할 |
|------|------|
| **`data.py`** | • CSV 로드 → Alpaca 프롬프트화 → `datasets.Dataset` <br>• 95 / 5 train‑eval 분할 (고정 seed = 42) |
| **`prompts.py`** | Alpaca 템플릿 문자열 두 가지(입력 유무) |
| **`metric.py`** | BLEU + Perplexity 계산 함수 (HuggingFace `evaluate`) |

```python
# 예시: 프롬프트 생성 (축약)
prompt = f"""아래에는 작업을 설명하는 지시문과 입력이 주어집니다.
### 지시문: {instruction}
### 입력: {input}
### 응답: {output}"""
```

### 루트 스크립트

| 파일 | 주요 포인트 |
|------|------------|
| **`train.py`** | *Base* = `gpt2`; LoRA on `c_attn`, r = 16, α = 32 <br>20 epochs, batch 2 × grad acc 16, FP16 지원 |
| **`quantization.py`** | LoRA 병합 → `BitsAndBytesConfig(nf4)` 4‑bit 양자화 → 저장 |
| **`inference.py`** | 템플릿 채운 뒤 `model.generate()` (temp 0.7, top‑p 0.9, max 150) |

---

## License & Acknowledgement

* 코드: MIT (unless stated otherwise)  
* 데이터: 공개 보험 약관을 재가공한 2차 저작물 – **비영리 연구·교육 목적** 사용을 권장합니다.  

---
