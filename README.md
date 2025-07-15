# Chat Bot (Insurance)

한국어 보험 QA 데이터를 기반으로 **GPT‑2**를 LoRA 방식으로 파인튜닝하고  
4‑bit 양자화까지 수행해 경량‑GPU 환경에서도 실시간 상담 챗봇을 돌릴 수 있게 한 프로젝트입니다.

## Features
* **LoRA 파인튜닝** – `trl.SFTTrainer` 활용으로 손쉽게 미세조정  
* **지침형(Alpaca) 프롬프트** 자동 생성 파이프라인  
* **4‑bit 양자화** – `BitsAndBytesConfig` 지원, VRAM 절감  


## Quick Start
```bash
# 의존성 & 하드웨어 세팅
python setup.py
```

## 🏋️‍♂️ Training
```bash
# data/train.csv (columns: instruction,input,output) 준비 후
python train.py
```
결과 LoRA 어댑터 → `checkpoints/gpt2-lora/`

## 🗜 Quantization (선택)
```bash
python quantization.py   # LoRA → 4-bit 모델 저장
```

## 🔮 Inference
```bash
python inference.py      # 샘플 질문에 대한 응답 출력
```

## 📁 Project Structure
```
chat_bot_ins/
├─ data/                # train.csv (↖︎사용자 준비)
├─ utils/
│  ├─ data.py, prompts.py, metric.py
├─ train.py             # LoRA 학습
├─ quantization.py      # 4-bit 양자화
├─ inference.py         # 추론 예시
├─ setup.py             # ← 원클릭 환경 구축
└─ README.md            # (YOU ARE HERE)
```
