# inference.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.prompts import ALPACA_PROMPT_TEMPLATE, ALPACA_PROMPT_WITHOUT_INPUT_TEMPLATE

def run_inference(model, tokenizer, instruction):
    """
    주어진 instruction과 input으로 추론을 실행합니다.
    """
    # 1. 프롬프트 생성
    prompt = ALPACA_PROMPT_WITHOUT_INPUT_TEMPLATE.format(instruction=instruction)

    # 2. 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 3. 모델 추론
    print("추론 중...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # 4. 결과 디코딩 및 출력
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    for tag in ["### 응답:", "### Response:"]:
        if tag in response:
            return response.split(tag, 1)[1].strip()
    # 
    return response[len(prompt):].strip()


def main():
    # 1. 학습 모델 및 토크나이저 로드
    model_path = "checkpoints/gpt2-lora/checkpoint-100"
    print(f"'{model_path}'에서 모델을 로드합니다...")
    
    model = AutoModelForCausalLM.from_pretrained('gpt2', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = PeftModel.from_pretrained(model, model_path)
    
    print("모델 로드 완료!")

    # 2. 추론할 샘플 데이터
    sample_instruction = "실손 보험과 종합 보험의 가장 큰 차이점은 무엇인가요?"
    
    # 3. 추론 실행
    generated_answer = run_inference(model, tokenizer, sample_instruction)
    
    # 4. 결과 출력
    print("\n" + "="*50)
    print(f"📜 질문: {sample_instruction}")
    print(f"🤖 생성된 답변: {generated_answer}")
    print("="*50)


if __name__ == "__main__":
    main()