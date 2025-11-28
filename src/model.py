import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import json

class WorldClassProgrammingAI:
    def __init__(self, model_name="microsoft/CodeGPT-small-py", use_lora=True):
        self.model_name = model_name
        self.use_lora = use_lora
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """세계 최고 수준의 모델 로드"""
        print("🔮 세계 최고 수준 프로그래밍 AI 모델 로딩 중...")
        
        # 모델 구성 로드
        try:
            with open('model_config.json', 'r') as f:
                model_config = json.load(f)
        except:
            model_config = {}
            
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **model_config
        )
        
        # LoRA 적용 (효율적인 파인튜닝)
        if self.use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            
        print("✅ 모델 로딩 완료!")
        return self.model
    
    def generate_code(self, prompt, max_length=512, temperature=0.7):
        """코드 생성 메인 함수"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code

class AdvancedCodeModel(nn.Module):
    """향상된 코드 생성 모델"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.code_quality_head = nn.Linear(768, 1)  # 코드 품질 예측 헤드
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
