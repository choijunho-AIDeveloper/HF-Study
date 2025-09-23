## Processing Data

#### MRPC 데이터셋

- 로드
    - `load_dataset(”glue”, “mrpc”)`
- 데이터 특징
    - Train,Test, Valid 세트로 구성
    - Sentence1: 첫 번째 문장
    - Sentence2: 두 번째 문장
    - label: 두 문장이 의미상 동일한지 아닌지
    - index: 데이터셋 인덱스

#### 토큰화

- 문장 쌍 토큰화
    - MRPC 데이터셋은 Sentence1, Sentence2로 구성된 쌍 데이터로, 모델이 두 문장을 한 쌍으로 처리하기 위해 입력 형식을 맞춰야 함
    - 방법
        - `tokenizer(My First Sentence, My Second Sentence)`
    - 결과
        - input_ids: 문장의 각 토큰을 고유한 숫자로 변환
        - attention_mask: 실제 토큰과 패딩 토큰을 구분
        - token_type_ids: 입력의 첫 번째 문장과 두 번째 문장을 구분
            - Raw: [CLS] My First Sentence [SEP] My Second Sentence [SEP]
                - 0: [CLS] My First Sentence [SEP]
                - 1: My Second Sentence [SEP]
            - Next Sentence Prediction Task에서 활용
- 효율화
    - 이전 방식은 전체 데이터셋을 한 번에 토큰화함으로써 메모리 효율성 문제, 딕셔너리로 반환되어 데이터셋 객체를 사용할 수 없는 문제가 발생
    - `dataset.map()`
        - e.g., `raw_dataset.map(tokenize_fn, batched = True)`
            - `batched = True`
                - 데이터를 개별 샘플이 아닌 배치 단위로 처리
            - Dynamic Padding
                - `tokenize_fn`에서는 전체 데이터셋 길이에 맞춘 비효율적인 패딩 대신 배치 단위 길이에 맞춘 동적 패딩을 활용

## Fine-tuning a model with the Trainer API

#### Trainer 클래스

- 정의
    - Trainer 클래스는 사전 훈련된 모델을 사용자 데이터셋에 맞춰 미세 조정할 때, 그 훈련 과정을 파이프라인화 함
- 종류
    - TrainingArguments
        - 정의
            - 학습 및 평가에 사용되는 모든 하이퍼파라미터를 정의
        - 필수 Args
            - 학습된 모델과 체크포인트가 저장될 경로
    - Trainer
        - 정의
            - 자동화된 학습을 제공
        - 필수 Args
            - Model
            - Train Arguments
            - Dataset
            - Data Collator
                - 동적 패딩
            - Processing Class
                - 어떤 토크나이저를 사용할지
