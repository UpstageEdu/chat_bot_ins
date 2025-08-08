# inference.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# PeftModel is no longer needed, but BitsAndBytesConfig is required
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
   # A small fix here to ensure we only get the generated part
   prompt_decoded = tokenizer.decode(inputs[0], skip_special_tokens=True)
   return response[len(prompt_decoded):].strip()


def main():
   # 1. 최종 모델이 저장된 경로를 지정합니다.
   model_path = "checkpoints/gpt2-lora/checkpoint-100" # 추론에 사용할 모델의 경로를 정확하게 입력해야 합니다.
   print(f"'{model_path}'에서 병합 및 양자화된 모델을 로드합니다...")
  
   # 2. 모델을 양자화할 때 사용했던 것과 동일한 BitsAndBytesConfig를 정의합니다.
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16
   )
  
   # 3. AutoModelForCausalLM을 사용하여 최종 모델을 직접 로드합니다.
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       quantization_config=bnb_config,
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained(model_path)
  
   print("모델 로드 완료!")

   # 4. 추론할 샘플 데이터
   sample_instruction = "실손 보험과 종합 보험의 가장 큰 차이점은 무엇인가요?"
  
   # 5. 추론 실행
   generated_answer = run_inference(model, tokenizer, sample_instruction)
  
   # 6. 결과 출력
   print("\n" + "="*50)
   print(f"질문: {sample_instruction}")
   print(f"생성된 답변: {generated_answer}")
   print("="*50)


if __name__ == "__main__":
   main()

