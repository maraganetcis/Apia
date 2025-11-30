# core_engine/model_loader.py
import torch
import os
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import json

class ApiaModelLoader:
    """Apia 모델 로드 및 관리 클래스"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_configs = {}
        
    def load_model_with_quantization(self, 
                                   model_name: str,
                                   use_4bit: bool = True,
                                   use_8bit: bool = False,
                                   device_map: str = "auto") -> Dict[str, Any]:
        """양자화를 사용한 모델 로드"""
        
        print(f"🔧 모델 로드 중: {model_name}")
        
        try:
            # 양자화 설정
            quantization_config = None
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.float16 if not use_4bit else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            model_info = {
                "model": model,
                "tokenizer": tokenizer,
                "name": model_name,
                "loaded_at": self._get_timestamp(),
                "quantization": "4bit" if use_4bit else "8bit" if use_8bit else "none"
            }
            
            self.loaded_models[model_name] = model_info
            print(f"✅ 모델 로드 완료: {model_name}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {model_name} - {e}")
            raise
    
    def load_peft_model(self, base_model_name: str, peft_model_path: str) -> Dict[str, Any]:
        """PEFT 모델 로드"""
        print(f"🔧 PEFT 모델 로드: {peft_model_path}")
        
        try:
            # 기본 모델 로드
            base_model_info = self.load_model_with_quantization(base_model_name)
            base_model = base_model_info["model"]
            tokenizer = base_model_info["tokenizer"]
            
            # PEFT 모델 로드
            model = PeftModel.from_pretrained(base_model, peft_model_path)
            
            model_info = {
                "model": model,
                "tokenizer": tokenizer,
                "name": f"{base_model_name}-peft",
                "peft_path": peft_model_path,
                "loaded_at": self._get_timestamp()
            }
            
            self.loaded_models[model_info["name"]] = model_info
            print(f"✅ PEFT 모델 로드 완료: {peft_model_path}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ PEFT 모델 로드 실패: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """모델 언로드 및 메모리 해제"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"🗑️ 모델 언로드: {model_name}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """모델 정보 조회"""
        return self.loaded_models.get(model_name)
    
    def list_loaded_models(self) -> list:
        """로드된 모델 목록 반환"""
        return list(self.loaded_models.keys())
    
    def _get_timestamp(self) -> str:
        """타임스탬프 생성"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def optimize_for_inference(self, model_info: Dict[str, Any]):
        """추론 최적화"""
        model = model_info["model"]
        
        # 평가 모드로 전환
        model.eval()
        
        # 그래프 최적화
        if hasattr(model, "config"):
            model.config.use_cache = True
        
        print("⚡ 추론 최적화 완료")

# 전역 모델 로더 인스턴스
_model_loader = None

def get_model_loader():
    """전역 모델 로더 인스턴스 얻기"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ApiaModelLoader()
    return _model_loader

if __name__ == "__main__":
    # 테스트 코드
    loader = ApiaModelLoader()
    print("Apia 모델 로더 준비 완료!")
