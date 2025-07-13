# train.py
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer,SFTConfig
from utils.data import load_and_prepare_data
from utils.metric import build_compute_metrics
from transformers import DataCollatorForLanguageModeling


def main():
    # 기본 설정
    model_name = "gpt2"
    output_dir = "checkpoints/gpt2-lora"
    
    # 1. 데이터 로드 및 전처리
    print("데이터 로딩 및 전처리를 시작합니다...")
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data(model_name)
    print("데이터 준비 완료!")

    # 2. 기본 모델 로드
    print(f"'{model_name}' 모델을 로드합니다...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id # pad_token_id 설정

    # 3. LoRA 설정
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"] # GPT-2의 어텐션 레이어 타겟
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
        return_tensors = "pt",
    )
    
    # 4. 학습 인자 설정
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=20, 
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        logging_steps=1,
        fp16=torch.cuda.is_available(), # FP16 학습 활성화 (GPU 사용 시)
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=10,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    # 5. SFTTrainer를 사용한 학습
    trainer = SFTTrainer(
        model=model,    
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
        processing_class=tokenizer,
        args=training_args,
        compute_metrics=build_compute_metrics(tokenizer) 
    )
    print("LoRA 파인튜닝을 시작합니다...")
    trainer.train()
    print("학습 완료!")

    # 6. 학습된 LoRA 어댑터 저장
    trainer.save_model(output_dir)
    print(f"학습된 모델이 '{output_dir}'에 저장되었습니다.")


if __name__ == "__main__":
    main()