## Token Classification
- 정의
    - 텍스트를 구성하는 각 토큰에 레이블을 할당하는 작업
- 종류
    - 개채명 인식(NER)
        - 문장에서 사람, 장소, 조직과 같은 개체(Entity)를 찾는 작업
        - 각 토큰은 개체 클래스(e.g., 사람, 장소) 또는 개체 아님에 대한 레이블을 부여
    - 품사 태깅(POS)
        - 문장의 각 단어에 품사(e.g., 명사, 동사, 형용사 등)를 부여하는 작업
    - 청킹(Chunking)
        - 같은 개체에 속하는 토큰을 찾는 작업
        - e.g., 시작 토큰: ‘B-’, 내부 토큰: ‘I-’, 어떤 청크에도 속하지 않는 토큰: ‘O’
- LAB
    - 데이터셋
        - NER 특화 CoNNL-2003 데이터셋 로드
        
        ```python
        from datasets import load_dataset
        
        raw_datasets = load_dataset("conll2003")
        ```
        
        - 데이터 & 라벨
        
        ```python
        raw_datasets["train"][0]["tokens"]
        # Return: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
        raw_datasets["train"][0]["ner_tags"]
        # Return: [3, 0, 7, 0, 0, 0, 7, 0, 0]
        ```
        
    - 전처리
        - 토큰화(is_split_into_words= True)
        
        ```python
        inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
        inputs.tokens()
        # Return: ['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']
        ```
        
        - 정렬
            - 원본 단어 리스트와 토크나이저에 의해 생성된 토큰 리스트의 길이가 달라짐
        
        ```python
        # 개별 토큰에 대해 전처리
        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
                    # If the label is B-XXX we change it to I-XXX
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)
        
            return new_labels
        ```
        
        ```python
        # 전체 데이터에 대해 배치 단위 처리
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            all_labels = examples["ner_tags"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))
        
            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs
            
        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        ```
        
    - 평가
        - seqeval 평가 지표
            - 토큰 분류 작업에서 사용되는 평가 프레임워크
            - 예측값과 참조 레이블에 대해 숫자 형태가 아닌 문자열 리스트로 받음
            - 개체 유형별 Precision, Recall, F1 Score과 Overall 점수를 반환
        ```python
        # !pip install seqeval
        import evaluate

        metric = evaluate.load("seqeval")
        ```

## Fine-Tuning Masked Language Model
- 정의
    - 마스크드 언어 모델링을 미세 조정하는 과정
- 도메인 적응
    - 법률 계약서와 과학 논문처럼 특정 도메인의 전문 용어가 많은 데이터셋을 다룰 때는 사전 훈련된 모델이 해당 용어들을 희귀 토큰으로 취급하여 성능이 저하될 수 있음
    - 이 때, 사전 훈련된 모델을 먼저 특정 도메인의 데이터로 미세 조정하는 과정을 도메인 적응이라고 함
    - 이 과정을 통해 여러 하위 작업(Downstream Task)의 성능을 향상시킬 수 있음
- LAB
    - 사전 훈련 모델로 DistilBERT를 선택
        - 지식 증류를 활용
            - BERT(교사 모델)보다 적은 수의 매개변수를 가지도록 훈련된 학생 모델로 훈련 속도가 빠름
    - 데이터셋
        - 영화 리뷰 데이터셋 IMDb 로드
        
        ```python
        from datasets import load_dataset
        
        imdb_dataset = load_dataset("imdb")
        imdb_dataset
        ```
        
    - 전처리
        - Concatenation
            - 개별 예시를 자르면 유용한 정보가 손실될 수 있으므로, 모든 텍스트 예시를 하나로 합친 다음 일정한 크기의 덩어리(Chunk)로 나눔
        - 토크나이징(without truncation= True)
        
        ```python
        def tokenize_function(examples):
            result = tokenizer(examples["text"])
            if tokenizer.is_fast:
                result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
            return result
        
        # Use batched=True to activate fast multithreading!
        tokenized_datasets = imdb_dataset.map(
            tokenize_function, batched=True, remove_columns=["text", "label"]
        )
        tokenized_datasets
        ```
        
        - Chunk 크기 정의
        
        ```python
        chunk_size = tokenizer.model_max_length # Return 512
        chunk_size = 128 # For limit memory, like colab environment
        ```
        
        - 데이터 청크화
        
        ```python
        # Data concatenate
        concatenated_examples = {
            k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
        }
        total_length = len(concatenated_examples["input_ids"])
        
        # Chunk
        chunks = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        
        # Overall process function
        def group_texts(examples):
            # Concatenate all texts
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            # Compute length of concatenated texts
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the last chunk if it's smaller than chunk_size
            total_length = (total_length // chunk_size) * chunk_size
            # Split by chunks of max_len
            result = {
                k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
                for k, t in concatenated_examples.items()
            }
            # Create a new labels column
            result["labels"] = result["input_ids"].copy()
            return result
        ```
        
    - 마스킹
        - Datacollator 마스킹(토큰 단위)
        
        ```python
        from transformers import DataCollatorForLanguageModeling
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        ```
        
        - 단어 단위 마스킹
        
        ```python
        import collections
        import numpy as np
        
        from transformers import default_data_collator
        
        wwm_probability = 0.2
        
        def whole_word_masking_data_collator(features):
            for feature in features:
                word_ids = feature.pop("word_ids")
        
                # Create a map between words and corresponding token indices
                mapping = collections.defaultdict(list)
                current_word_index = -1
                current_word = None
                for idx, word_id in enumerate(word_ids):
                    if word_id is not None:
                        if word_id != current_word:
                            current_word = word_id
                            current_word_index += 1
                        mapping[current_word_index].append(idx)
        
                # Randomly mask words
                mask = np.random.binomial(1, wwm_probability, (len(mapping),))
                input_ids = feature["input_ids"]
                labels = feature["labels"]
                new_labels = [-100] * len(labels)
                for word_id in np.where(mask)[0]:
                    word_id = word_id.item()
                    for idx in mapping[word_id]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = tokenizer.mask_token_id
                feature["labels"] = new_labels
        
            return default_data_collator(features)
        ```
        
    - 평가
        - 퍼플렉시티
            - 언어 모델이 새로운 텍스트를 얼마나 잘 예측하는지 측정하는 지표
            - 모델이 훈련 데이터에 없는 문장을 접했을 때, 얼마나 당황하는지(Perlexed)에 대한 정도를 나타냄
            - 낮은 퍼플렉시티 값은 모델이 새로운 데이터에 대해 덜 당황한다는 것을 의미하므로 더 좋은 언어 모델을 의미
            - Cross-Entropy Loss에 대한 지수 값으로 정의
        ```python
        import math

        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        ```

## Translation
- 정의
    - 한 시퀀스(문장)를 다른 시퀀스로 변환하는 시퀀스-투-시퀀스 작업
- LAB
    - 데이터셋
        - 영어와 프랑스어 문장 쌍을 담은 KDE4 데이터셋
        
        ```python
        from datasets import load_dataset
        
        raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
        ```
        
    - 토큰화
        - 영어 문장을 입력, 프랑스어 문장을 대상으로 토큰화
        
        ```python
        max_length = 128
        
        def preprocess_function(examples):
            inputs = [ex["en"] for ex in examples["translation"]]
            targets = [ex["fr"] for ex in examples["translation"]]
            model_inputs = tokenizer(
                inputs, text_target=targets, max_length=max_length, truncation=True
            )
            return model_inputs
            
        tokenized_datasets = split_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=split_datasets["train"].column_names,
        )
        ```
        
    - 평가
        - SacreBLEU
            - BLEU
                - 생성된 번역이 정답과 얼마나 비슷한지를 측정
                - 문장의 이해도나 문법적 정확성보다는 통계적인 규칙을 기반
                - 생성된 문장의 모든 단어가 정답에도 나타나는지 확인
            - 토큰화 단계의 차이를 표준화하여 서로 다른 모델 간 점수를 공정히 비교
        
        ```python
        # !pip install sacrebleu
        import evaluate
        
        metric = evaluate.load("sacrebleu")
        ```
        
        - 평가 함수
        ```python
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # In case the model returns more than the prediction logits
            if isinstance(preds, tuple):
                preds = preds[0]

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Replace -100s in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [[label.strip()] for label in decoded_labels]

            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"bleu": result["score"]}
        ```

##  Summarization Task
- 정의
    - 긴 글을 이해하고 문서의 핵심 내용을 담은 일관성 있는 텍스트를 생성하는 작업
- LAB
    - 일반적으로 요약 모델은 영어에 의존함
    - 해당 Lab에서는 영어와 스페인어를 다루는 고객 후기 요약 모델을 훈련하는 것이 목표
    - 데이터셋
        - 다국어 아마존 후기 코퍼스(Multilingual Amazon Reviews Corpus) 데이터셋 로드
        
        ```python
        from datasetsimport load_dataset
        
        spanish_dataset = load_dataset("amazon_reviews_multi", "es")
        english_dataset = load_dataset("amazon_reviews_multi", "en")
        ```
        
        - 무작위 샘플링(Optional)
        
        ```python
        def show_samples(dataset, num_samples=3, seed=42):
            sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
            for example in sample:
                print(f"\n'>> Title: {example['review_title']}'")
                print(f"'>> Review: {example['review_body']}'")
        ```
        
        - 카테고리 별 데이터 수(Optional)
        
        ```python
        english_dataset.set_format("pandas")
        english_df = english_dataset["train"][:]
        # Show counts for top 20 products
        english_df["product_category"].value_counts()[:20]
        ```
        
        - 필터링
            - 특정 카테고리 데이터만 추출
        
        ```python
        def filter_books(example):
            return (
                example["product_category"] == "book"
                or example["product_category"] == "digital_ebook_purchase"
            )
        ```
        
    - 전처리
        - 스페인어 데이터셋과 영어 데이터셋을 Concatenate
            - Datasets Lib에서는 `concatenate_datasets`를 제공
        
        ```python
        from datasets import concatenate_datasets, DatasetDict
        
        books_dataset = DatasetDict()for split in english_books.keys():
            books_dataset[split] = concatenate_datasets(
                [english_books[split], spanish_books[split]]
            )
            books_dataset[split] = books_dataset[split].shuffle(seed=42)# Peek at a few examples
        show_samples(books_dataset)
        ```
        
        - 길이가 짧은 데이터를 Drop
            - 띄워쓰기를 기준으로 제목이 세 단어 이상인 것만 추출
            - 길이가 짧은 데이터 학습에 치우치면 편향이 생길 수 있음
        
        ```python
        books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
        ```
        
    - 모델 선정
        - mT5, mBART-50(모두 multilingual을 지원) 중 mT5를 선정
        
        ```python
        from transformers import AutoTokenizer
        
        model_checkpoint = "google/mt5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        ```
        
        - 토크나이징 테스트
        
        ```python
        inputs = tokenizer("I loved reading the Hunger Games!")
        inputs
        
        tokenizer.convert_ids_to_tokens(inputs.input_ids)
        '''
        Return [' I', ' ', 'loved', ' reading', ' the', ' Hung', 'er', ' Games', '</s>']
        '''
        ```
        
        - 문장 길이 규제
        
        ```python
        max_input_length = 512
        max_target_length = 30
        
        def preprocess_function(examples):
            model_inputs = tokenizer(
                examples["review_body"],
                max_length=max_input_length,
                truncation=True,
            )
            labels = tokenizer(
                examples["review_title"], max_length=max_target_length, truncation=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
            
        tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
        ```
        
    - 모델 평가
        - ROUGE 점수
            - 요약 작업의 경우, “잘” 요약했다라고 판단하는 평가 지표가 어려움
            - ROUGE 점수는 $y_{pred}$와 $y_{true}$의 겹치는 부분에 대한 Precision과 Recall을 활용
                - Precision = 겹치는 총 단어 수/생성된 문장의 총 단어 수
                - Recall = 겹치는 총 단어 수/참조된 문장의 총 단어 수
                - 이 둘의 조화 평균인 F1-Score을 통해 결과를 측정
        
        ```python
        import evaluate
        
        rouge_score = evaluate.load("rouge")
        
        scores = rouge_score.compute(
            predictions=[generated_summary], references=[reference_summary]
        )
        scores
        ```
        
    - 미세 조정
        ```python
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

        from transformers import Seq2SeqTrainingArguments

        batch_size = 8
        num_train_epochs = 8

        # Show the training loss with every epoch
        logging_steps = len(tokenized_datasets["train"]) // batch_size
        model_name = model_checkpoint.split("/")[-1]

        args = Seq2SeqTrainingArguments(
            output_dir=f"{model_name}-finetuned-amazon-en-es",
            evaluation_strategy="epoch",
            learning_rate=5.6e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            logging_steps=logging_steps,
            push_to_hub=True,
        )

        import numpy as np
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # Decode generated summaries into text
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            # Decode reference summaries into text
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # ROUGE expects a newline after each sentence
            decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
            # Compute ROUGE scores
            result = rouge_score.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
            # Extract the median scores
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            return {k: round(v, 4) for k, v in result.items()}
            
        from transformers import DataCollatorForSeq2Seq
        # For Batch calculation
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        from transformers import Seq2SeqTrainer

        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.evaluate()
        ```

## Causal Language Model
- 정의
    - 전이학습이 아닌 End to End 모델 구축
- LAB
    - 파이썬 코드 생성 델 구축이 목표
    - 필터링
        - filters를 포함하는 데이터만 남김
            - e.g., `import numpy as np`는 filter
        
        ```python
        from collections import defaultdict
        from tqdm import tqdm
        from datasets import Dataset
        
        def filter_streaming_dataset(dataset, filters):
            filtered_dict = defaultdict(list)
            total = 0
            for sample in tqdm(iter(dataset)):
                total += 1
                if any_keyword_in_string(sample["content"], filters):
                    for k, v in sample.items():
                        filtered_dict[k].append(v)
            print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
            return Dataset.from_dict(filtered_dict)
        
        from datasets import load_dataset
        
        split = "train"  # "valid"
        filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
        
        data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
        filtered_data = filter_streaming_dataset(data, filters)
        ```
        
    - 토크나이즈
        - 사전 학습된 모델이 아니므로 자체적으로 토크나이징을 수행
        
        ```python
        def tokenize(element):
            outputs = tokenizer(
                element["content"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == context_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}
        
        tokenized_datasets = raw_datasets.map(
            tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
        )
        tokenized_datasets
        ```
        
    - 모델 초기화
        - GPT-2의 Config만 불러옴
        
        ```python
        from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
        
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        model = GPT2LMHeadModel(config)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
        '''
        return 12.4m
        '''
        ```
        
    - 데이터 콜렉터 초기화
        - 마스크 언어 모델링, 인과적 언어 모델링을 모두 지원
        - 해당 작업에서는 인과적 언어 모델링이므로 `mlm= False`
        
        ```python
        from transformers import DataCollatorForLanguageModeling
        
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        ```
        
    - 학습
        
        ```python
        from transformers import Trainer, TrainingArguments
        
        args = TrainingArguments(
            output_dir="codeparrot-ds",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            eval_steps=5_000,
            logging_steps=5_000,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            weight_decay=0.1,
            warmup_steps=1_000,
            lr_scheduler_type="cosine",
            learning_rate=5e-4,
            save_steps=5_000,
            fp16=True,
            push_to_hub=True,
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
        )
        
        trainer.train()
        ```
        
    - 결과 확인
        ```python
        txt = """\
        # create some data
        x = np.random.randn(100)
        y = np.random.randn(100)

        # create dataframe from x and y
        """
        print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
        '''
        # create some data
        x = np.random.randn(100)
        y = np.random.randn(100)

        # create dataframe from x and y
        df = pd.DataFrame({'x': x, 'y': y})
        df.insert(0,'x', x)for
        '''
        ```

## Question Answering
- 정의
    - 질문-답변 방식 중 문서에 대해 질문하고, 그 질문에 대한 답을 문서 자체 텍스트 범위에서 찾는 방법
- 한계
    - “Transformer는 누가 개발했나요?”와 같은 사실적 질문에는 잘 답변하는 반면, “하늘은 왜 파랗나요?”와 같은 개방형 질문에는 잘 답변하지 못
- LAB
    - 데이터 로드
        - SQuAD 데이터셋
            - 위키피디아 기사 세트에 대해 크라우드워커들이 던진 질문들로 구성
    
    ```python
    from datasets import load_dataset
    
    raw_datasets = load_dataset("squad")
    ```
    
    - 토크나이징
    
    ```python
    from transformers import AutoTokenizer
    
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    ```
    
    - 전처리
        - Raw Data는 Dictionary 형태로 되어있으므로 이를 BERT 모델에 맞춰 특수 토큰 처리를 해줘야 함
    
    ```python
    '''
    Raw Data Format: Context: 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
    Question: 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
    '''
    context = raw_datasets["train"][0]["context"]
    question = raw_datasets["train"][0]["question"]
    
    inputs = tokenizer(question, context)
    tokenizer.decode(inputs["input_ids"])
    '''
    Return '[CLS] To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? [SEP] Architecturally, ''the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin ''Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms ''upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred ''Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a ''replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette ''Soubirous in 1858. At the end of the main drive ( and in a direct line that connects through 3 statues ''and the Gold Dome ), is a simple, modern stone statue of Mary. [SEP]'
    '''
    
    ```
    
    - 긴 컨텍스트 처리 기술
        - 컨텍스트 길이가 설정한 최대 길이를 초과할 경우, 여러 작은 조각으로 나눠서 학습 데이터로 사용하는 방법
        - 슬라이딩 윈도우
            - 긴 컨텍스트를 여러 슬라이드로 나누고, 각 조각에 겹치는 부분을 둬 정보 손실을 최소화
            - 답변이 특정 위치에 포함되므로 답변이 포함되지 않은 조각들은 별도 처리
                - 입력에 답변이 포함되는 경우
                    - 시작 위치와 끝 위치를 인덱스의 레이블로 설정
                        - e.g. “해운대 바닷가”라는 답변이 26:28에 있다면 start= 26, end= 27
                - 입력에 답변이 포함되지 않은 경우
                    - 시작 위치와 끝 위치를 0으로 설정하여 [CLS] 토큰을 예측하도록 유도
                    - 여기에 답변이 포함되지 않았다는 것을 명시하기 위함
            - 오프셋 매핑
                - 각 토큰의 위치 정보를 매핑
                - 정확한 컨텍스트를 출력하기 위함
            - 오버플로우 매핑
                - 하나의 긴 문서가 여러 짧은 조각으로 분할되었을 경우, 어떤 원본 문서에서 왔는지 매핑
            - 오프셋을 각 서브 조각의 특정 열이고, 오버플로우를 전체 데이터에 대한 각각의 행으로 생각하면 쉬움
    - 미세 조정
    
    ```python
    from tqdm.auto import tqdm
    
    def compute_metrics(start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)
    
        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []
    
            # 해당 예제와 관련된 모든 피처를 반복합니다
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]
    
                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # 컨텍스트 내에 완전히 포함되지 않는 답변은 건너뜁니다
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # 길이가 0보다 작거나 max_answer_length보다 긴 답변은 건너뜁니다
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
    
                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)
    
            # 가장 좋은 점수를 가진 답변을 선택합니다
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})
    
        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    from transformers import Trainer, TrainingArguments
    
    args = TrainingArguments(
        "bert-finetuned-squad",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=True,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    ```
    
    - 테스트
    ```python
    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
    '''
    Return {'exact_match': 81.18259224219489, 'f1': 88.67381321905516}
    '''
    ```