import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import yaml
from model import WorldClassProgrammingAI
from dataset import CodeDataset
import os
from dotenv import load_dotenv

load_dotenv()

class WorldClassTrainer:
    def __init__(self, config_path="train_config.yaml"):
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_config(self, config_path):
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        print("🚀 세계 1등 AI 모델 초기화 중...")
        
        model_config = self.config['model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['base_model'])
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['base_model'],
            load_in_8bit=model_config.get('load_in_8bit', True),
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ 모델 초기화 완료!")
    
    def setup_training(self):
        """학습 설정"""
        print("🎯 세계 최고 수준 학습 설정 중...")
        
        training_config = self.config['training']
        
        # 데이터셋 로드
        self.train_dataset = CodeDataset(
            self.config['data']['train_file'], 
            self.tokenizer,
            max_length=4096
        )
        self.val_dataset = CodeDataset(
            self.config['data']['val_file'],
            self.tokenizer, 
            max_length=4096
        )
        
        # 학습 인자 설정
        self.training_args = TrainingArguments(
            output_dir="./world_class_checkpoints",
            overwrite_output_dir=True,
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            warmup_steps=training_config['warmup_steps'],
            learning_rate=training_config['learning_rate'],
            fp16=training_config['fp16'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            save_total_limit=training_config['save_total_limit'],
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            eval_accumulation_steps=1,
        )
        
        # 데이터 콜레이터
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        print("✅ 학습 설정 완료!")
    
    def train(self):
        """세계 최고 수준 학습 실행"""
        print("🔥 세계 1등 AI 모델 학습 시작!")
        
        self.setup_model_and_tokenizer()
        self.setup_training()
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        
        # 학습 시작
        trainer.train()
        
        # 모델 저장
        trainer.save_model("./world_class_final_model")
        self.tokenizer.save_pretrained("./world_class_final_model")
        
        print("🎉 세계 최고 수준 AI 모델 학습 완료!")
        return trainer

if __name__ == "__main__":
    trainer = WorldClassTrainer()
    trainer.train()
