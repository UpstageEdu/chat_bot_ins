# inference.py
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_inference(model, tokenizer, instruction):
    """
    주어진 instruction과 input으로 추론을 실행합니다.
    """

    # 2. 토크나이징
    inputs = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": instruction,
            },
        ],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("mps")

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
            top_p=0.9,
        )

    # 4. 결과 디코딩 및 출력
    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    return response.strip()


def main():
    # 1. 학습 모델 및 토크나이저 로드
    model_path = "checkpoints/SmolLM2-360M-Instruct-lora"
    print(f"'{model_path}'에서 모델을 로드합니다...")
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # model = PeftModel.from_pretrained(model, model_path)

    print("모델 로드 완료!")

    # 2. 추론할 샘플 데이터
    sample_instruction = "Does Medicare Cover Co-Pays?"

    # 3. 추론 실행
    generated_answer = run_inference(model, tokenizer, sample_instruction)

    # 4. 결과 출력
    print("\n" + "=" * 50)
    print(f"질문: {sample_instruction}")
    print(f"생성된 답변: {generated_answer}")
    print("=" * 50)


if __name__ == "__main__":
    main()
