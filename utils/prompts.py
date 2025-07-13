# utils/prompts.py

ALPACA_PROMPT_TEMPLATE = """아래에는 작업을 설명하는 지시문과 추가적인 문맥을 제공하는 입력이 주어집니다. 요청을 적절히 완료하는 응답을 작성하세요.

### 지시문:
{instruction}

### 입력:
{input}

### 응답:
"""

ALPACA_PROMPT_WITHOUT_INPUT_TEMPLATE = """아래에는 작업을 설명하는 지시문이 주어집니다. 요청을 적절히 완료하는 응답을 작성하세요.

### 지시문:
{instruction}

### 응답:
"""
