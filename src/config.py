import yaml
import json
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class WorldClassConfig:
    """세계 최고 수준 설정 관리"""
    
    # 모델 설정
    model_name: str = "microsoft/CodeGPT-small-py"
    max_length: int = 4096
    vocab_size: int = 50257
    
    # 학습 설정
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    
    # 데이터 설정
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    test_file: str = "data/processed/test.jsonl"
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """YAML 파일에서 설정 로드"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            model_name=config_data['model']['base_model'],
            max_length=config_data['model']['max_length'],
            batch_size=config_data['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=config_data['training']['gradient_accumulation_steps'],
            learning_rate=config_data['training']['learning_rate'],
            num_epochs=config_data['training']['num_train_epochs'],
            warmup_steps=config_data['training']['warmup_steps'],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }

class AdvancedConfig:
    """향상된 설정 관리"""
    
    def __init__(self):
        self.config = self.load_all_configs()
    
    def load_all_configs(self):
        """모든 설정 파일 로드"""
        configs = {}
        
        # YAML 설정 로드
        try:
            with open('train_config.yaml', 'r') as f:
                configs['training'] = yaml.safe_load(f)
        except FileNotFoundError:
            print("⚠️ train_config.yaml 파일을 찾을 수 없습니다.")
            
        # JSON 설정 로드
        try:
            with open('model_config.json', 'r') as f:
                configs['model'] = json.load(f)
        except FileNotFoundError:
            print("⚠️ model_config.json 파일을 찾을 수 없습니다.")
            
        return configs
    
    def get(self, key, default=None):
        """설정 값 가져오기"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
