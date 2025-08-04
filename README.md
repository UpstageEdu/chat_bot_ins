# Chat Bot (Insurance)

한국어 보험 QA 데이터를 기반으로 **GPT‑2**를 LoRA 방식으로 파인튜닝하고  
4‑bit 양자화까지 수행해 경량‑GPU 환경에서도 실시간 상담 챗봇을 돌릴 수 있게 한 프로젝트입니다.

## Features
* **LoRA 파인튜닝** – `trl.SFTTrainer` 활용으로 손쉽게 미세조정  
* **지침형(Alpaca) 프롬프트** 자동 생성 파이프라인  
* **4‑bit 양자화** – `BitsAndBytesConfig` 지원, VRAM 절감  


## Quick Start
```bash
# 깃허브 레포 다운
git clone https://github.com/DopeorNope-Lee/chat_bot_ins.git
cd chat_bot_ins

# 콘다 가상환경 생성 및 활성화
conda create -n chatbot_ins python=3.11.8 -y
conda create chatbot_ins

pip install -r requirements.txt
# 의존성 & 하드웨어 세팅
python setup.py
```

## Training
```bash
# data/train.csv (columns: instruction,input,output) 준비 후
python train.py
```
결과 LoRA 어댑터 → `checkpoints/gpt2-lora/`

## Quantization (선택)
```bash
python quantization.py   # LoRA → 4-bit 모델 저장
```

## Inference
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
