import pandas as pd

train_processed_news_csv_path = "../data/processed/train/train_merged.csv"
validation_processed_news_csv_path = "../data/processed/validation/valid_merged.csv"

# Dataframe을 데이터셋으로 사용
df_train = pd.read_csv(train_processed_news_csv_path).dropna()
df_valid = pd.read_csv(validation_processed_news_csv_path).dropna()

df_train.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 683335 entries, 0 to 683334
Data columns (total 3 columns):
 #   Column           Non-Null Count   Dtype 
---  ------           --------------   ----- 
 0   Unnamed: 0       683335 non-null  int64 
 1   original_text    683335 non-null  object
 2   summarized_text  683335 non-null  object
dtypes: int64(1), object(2)
memory usage: 15.6+ MB
"""

df_valid.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 73916 entries, 0 to 73915
Data columns (total 3 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   Unnamed: 0       73916 non-null  int64 
 1   original_text    73916 non-null  object
 2   summarized_text  73916 non-null  object
dtypes: int64(1), object(2)
memory usage: 1.7+ MB
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import wandb
from transformers.integrations import WandbCallback

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, tokenizer, dataframe, max_length=512):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        original_text = row['original_text']
        summarized_text = row['summarized_text']
        
        # 인코딩
        encoding = self.tokenizer.encode_plus(
            original_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = self.tokenizer.encode(
            summarized_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).squeeze()

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

# CSV 파일에서 데이터셋 로드
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df['original_text'].tolist(), df['summarized_text'].tolist()

# 데이터셋 준비
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

# 데이터셋 생성
train_dataset = CustomDataset(tokenizer, df_train)
valid_dataset = CustomDataset(tokenizer, df_valid)

# 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")

# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 매개변수 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./results/korean-total-merged",
    num_train_epochs=3,
    per_device_train_batch_size=60,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    learning_rate=5e-5
)

# WandB 통합 설정
wandb.init(project="kobart-text-summ", name="231124-korean-total-merged") 
wandb_callback = WandbCallback()

# 트레이너 초기화 및 WandB 콜백 추가
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[wandb_callback]
)

# 모델 파인튜닝 시작
trainer.train()

# WandB 세션 종료
wandb.finish()
