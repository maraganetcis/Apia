import torch
from transformers import Trainer, TrainingArguments
from model import ProgrammingAI
from dataset import CodeDataset

def train_model():
    # 모델 및 토크나이저 초기화
    model = ProgrammingAI()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
    
    # 데이터셋 로드
    train_dataset = CodeDataset("data/processed/train.jsonl", tokenizer)
    val_dataset = CodeDataset("data/processed/val.jsonl", tokenizer)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_dir="./logs",
    )
    
    # 학습 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    trainer.save_model("./final_model")

if __name__ == "__main__":
    train_model()
