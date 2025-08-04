# utils/data.py

import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from utils.prompts import ALPACA_PROMPT_TEMPLATE, ALPACA_PROMPT_WITHOUT_INPUT_TEMPLATE

def load_and_prepare_data(model_name="gpt2", data_path="data/train.csv", test_size=0.05):
    """
    로컬 CSV 파일을 로드하고, 학습/평가용으로 분리한 뒤,
    Alpaca 프롬프트 형식으로 변환합니다.
    """
    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 계열엔 pad 토큰이 없으므로 eos 토큰을 재사용 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 판다스로 데이터셋 로드
    df = pd.read_csv(data_path)
    df['input'] = df['input'].fillna('')

    # 3. 프롬프트 생성 함수
    def create_prompt(row):
        if row['input']:
            full_prompt = ALPACA_PROMPT_TEMPLATE.format(
                instruction=row['instruction'],
                input=row['input']
            )
        else:
            full_prompt = ALPACA_PROMPT_WITHOUT_INPUT_TEMPLATE.format(
                instruction=row['instruction']
            )
        return f"{full_prompt}{row['output']}{tokenizer.eos_token}"

    df['text'] = df.apply(create_prompt, axis=1)
    
    # 4. 판다스 DataFrame을 datasets.Dataset 객체로 변환
    full_dataset = Dataset.from_pandas(df)

    # ✅ 5. 데이터셋을 학습용과 평가용으로 분리
    split_dataset = full_dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"데이터 분리 완료: 학습용 {len(train_dataset)}개, 평가용 {len(eval_dataset)}개")

    # ✅ 6. 분리된 데이터셋과 토크나이저 반환
    return train_dataset, eval_dataset, tokenizer
