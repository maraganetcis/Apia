import torch
import numpy as np
import random
import os
from datetime import datetime
import json
import logging

def setup_world_class_environment(seed=42):
    """세계 최고 수준 실험 환경 설정"""
    # 재현성 보장
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # GPU 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'world_class_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    print("🌍 세계 최고 수준 AI 환경 설정 완료!")

def save_world_class_checkpoint(model, tokenizer, epoch, metrics, path):
    """세계 최고 수준 체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'world_class_version': '1.0.0'
    }
    
    # 체크포인트 저장
    torch.save(checkpoint, f"{path}/checkpoint_epoch_{epoch}.pt")
    
    # 모델과 토크나이저 저장
    model.save_pretrained(f"{path}/model_epoch_{epoch}")
    tokenizer.save_pretrained(f"{path}/model_epoch_{epoch}")
    
    # 메타데이터 저장
    metadata = {
        'epoch': epoch,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'WorldClassProgrammingAI'
    }
    
    with open(f"{path}/metadata_epoch_{epoch}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 세계 최고 수준 체크포인트 저장 완료: 에포크 {epoch}")

def load_world_class_checkpoint(model, path, epoch=None):
    """체크포인트 로드"""
    if epoch is None:
        # 가장 최근 체크포인트 찾기
        checkpoints = [f for f in os.listdir(path) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            raise FileNotFoundError("체크포인트를 찾을 수 없습니다.")
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = f"{path}/{latest_checkpoint}"
    else:
        checkpoint_path = f"{path}/checkpoint_epoch_{epoch}.pt"
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📥 체크포인트 로드 완료: {checkpoint_path}")
    return model, checkpoint['metrics']

def calculate_code_quality_score(generated_code, original_code=None):
    """코드 품질 점수 계산 (세계 최고 수준 평가)"""
    score = 0
    
    # 1. 문법 검사 (기본 점수)
    try:
        compile(generated_code, '<string>', 'exec')
        score += 30
    except:
        pass
    
    # 2. 코드 길이 효율성
    lines = generated_code.split('\n')
    if len(lines) < 50:  # 너무 길지 않음
        score += 20
    
    # 3. 주석 존재 여부
    if '#' in generated_code or '//' in generated_code or '/*' in generated_code:
        score += 15
    
    # 4. 함수 정의 존재 여부
    if 'def ' in generated_code or 'function ' in generated_code:
        score += 20
    
    # 5. 에러 처리 존재 여부
    if 'try:' in generated_code or 'catch' in generated_code or 'except' in generated_code:
        score += 15
    
    return min(score, 100)

class WorldClassMetrics:
    """세계 최고 수준 메트릭 계산"""
    
    @staticmethod
    def calculate_pass_rate(generated_codes, test_cases):
        """코드 통과율 계산"""
        passed = 0
        total = len(generated_codes)
        
        for code, test_case in zip(generated_codes, test_cases):
            try:
                # 실제 실행 환경에서는 더 정교한 테스트 필요
                exec(code)
                exec(test_case)
                passed += 1
            except:
                continue
        
        return passed / total if total > 0 else 0
    
    @staticmethod
    def calculate_bleu_score(references, candidates):
        """BLEU 스코어 계산 (코드 유사도)"""
        from nltk.translate.bleu_score import sentence_bleu
        
        scores = []
        for ref, cand in zip(references, candidates):
            ref_tokens = ref.split()
            cand_tokens = cand.split()
            
            score = sentence_bleu([ref_tokens], cand_tokens)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0
