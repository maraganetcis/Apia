# core_engine/code_generator.py
import torch
import re
from typing import Dict, List, Optional, Any
from transformers import pipeline, TextStreamer
from .model_loader import get_model_loader

class ApiaCodeGenerator:
    """Apia 코드 생성 엔진"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B"):
        self.model_loader = get_model_loader()
        self.model_name = model_name
        self.model_info = None
        self.generation_config = {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None
        }
        
    def load_model(self, use_4bit: bool = True):
        """모델 로드"""
        self.model_info = self.model_loader.load_model_with_quantization(
            self.model_name, 
            use_4bit=use_4bit
        )
        self.model_loader.optimize_for_inference(self.model_info)
        
    def generate_code(self, 
                     prompt: str,
                     language: str = "python",
                     style: str = "clean",
                     max_length: int = 1024,
                     temperature: float = 0.7) -> Dict[str, Any]:
        """코드 생성 메인 함수"""
        
        if self.model_info is None:
            self.load_model()
        
        model = self.model_info["model"]
        tokenizer = self.model_info["tokenizer"]
        
        # 언어별 시스템 프롬프트
        system_prompts = {
            "python": self._get_python_system_prompt(),
            "javascript": self._get_javascript_system_prompt(),
            "java": self._get_java_system_prompt(),
            "cpp": self._get_cpp_system_prompt()
        }
        
        system_prompt = system_prompts.get(language, self._get_python_system_prompt())
        full_prompt = f"{system_prompt}\n\n사용자 요청: {prompt}\n\n생성된 코드:"
        
        try:
            # 입력 토크나이징
            inputs = tokenizer.encode(full_prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # 코드 생성
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    top_p=self.generation_config["top_p"],
                    do_sample=self.generation_config["do_sample"],
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # 결과 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 생성된 코드 부분만 추출
            generated_code = self._extract_generated_code(generated_text, full_prompt)
            
            # 코드 정제
            cleaned_code = self._clean_code(generated_code, language)
            
            return {
                "success": True,
                "code": cleaned_code,
                "full_generation": generated_text,
                "language": language,
                "model": self.model_name,
                "prompt": prompt
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": "",
                "model": self.model_name
            }
    
    def generate_multiple_versions(self, 
                                 prompt: str, 
                                 language: str = "python",
                                 num_versions: int = 3) -> List[Dict[str, Any]]:
        """다중 버전 코드 생성"""
        versions = []
        
        # 다양한 temperature로 여러 버전 생성
        temperatures = [0.3, 0.7, 1.0][:num_versions]
        
        for temp in temperatures:
            result = self.generate_code(
                prompt=prompt,
                language=language,
                temperature=temp,
                max_length=512
            )
            result["temperature"] = temp
            versions.append(result)
        
        return versions
    
    def complete_function(self, partial_code: str, language: str = "python") -> Dict[str, Any]:
        """부분 코드 완성"""
        prompt = f"다음 코드를 완성해주세요:\n```{language}\n{partial_code}\n```"
        return self.generate_code(prompt, language)
    
    def debug_code(self, code: str, error_message: str = "") -> Dict[str, Any]:
        """코드 디버깅"""
        prompt = f"다음 코드를 디버깅하고 수정해주세요:"
        if error_message:
            prompt += f"\n에러 메시지: {error_message}"
        prompt += f"\n```python\n{code}\n```"
        
        return self.generate_code(prompt)
    
    def _extract_generated_code(self, full_text: str, original_prompt: str) -> str:
        """생성된 코드 부분 추출"""
        # 원본 프롬프트 제거
        if original_prompt in full_text:
            code_part = full_text.split(original_prompt)[-1].strip()
        else:
            code_part = full_text
        
        # 코드 블록 추출 시도
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', code_part, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # 코드 블록이 없으면 첫 번째 함수/클래스부터 추출
        lines = code_part.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'import ', 'from ']):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else code_part
    
    def _clean_code(self, code: str, language: str) -> str:
        """코드 정제"""
        # 불필요한 설명 제거
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 주석이 아닌 코드 라인만 필터링 (너무 aggressive하지 않게)
            if not line.strip().startswith('#') or 'TODO' in line or 'FIXME' in line:
                cleaned_lines.append(line)
        
        cleaned_code = '\n'.join(cleaned_lines)
        
        # 언어별 추가 정제
        if language == "python":
            cleaned_code = self._clean_python_code(cleaned_code)
        elif language == "javascript":
            cleaned_code = self._clean_javascript_code(cleaned_code)
        
        return cleaned_code.strip()
    
    def _clean_python_code(self, code: str) -> str:
        """Python 코드 정제"""
        # 중복 import 제거 등 Python 특화 정제
        return code
    
    def _clean_javascript_code(self, code: str) -> str:
        """JavaScript 코드 정제"""
        # JavaScript 특화 정제
        return code
    
    def _get_python_system_prompt(self) -> str:
        return """당신은 세계 최고 수준의 Python 프로그래머입니다. 다음 지침을 따르세요:

1. Pythonic하고 효율적인 코드 작성
2. 타입 힌트 추가
3. 적절한 예외 처리 포함
4. 상세한 한국어 주석 추가
5. PEP 8 스타일 가이드 준수
6. 완전한 실행 가능한 코드 제공"""

    def _get_javascript_system_prompt(self) -> str:
        return """당신은 세계 최고 수준의 JavaScript 프로그래머입니다. 다음 지침을 따르세요:

1. Modern ES6+ 문법 사용
2. 적절한 에러 처리
3. Async/Await 적절히 사용
4. 상세한 한국어 주석 추가
5. Clean code 원칙 준수"""

    def _get_java_system_prompt(self) -> str:
        return """당신은 세계 최고 수준의 Java 프로그래머입니다. 다음 지침을 따르세요:

1. 객체지향 원칙 준수
2. 적절한 예외 처리
3. 의미 있는 변수/메소드명 사용
4. 상세한 한국어 주석 추가
5. Java 코딩 컨벤션 준수"""

    def _get_cpp_system_prompt(self) -> str:
        return """당신은 세계 최고 수준의 C++ 프로그래머입니다. 다음 지침을 따르세요:

1. Modern C++ (C++17/20) 문법 사용
2. 메모리 안전성 보장
3. RAII 원칙 준수
4. 상세한 한국어 주석 추가
5. 표준 라이브러리 적극 활용"""

# 빠른 사용을 위한 헬퍼 함수
def create_code_generator(model_name: str = "Qwen/Qwen2.5-Coder-7B") -> ApiaCodeGenerator:
    """코드 생성기 빠른 생성"""
    return ApiaCodeGenerator(model_name)

if __name__ == "__main__":
    # 테스트
    generator = ApiaCodeGenerator()
    result = generator.generate_code("리스트에서 중복을 제거하는 함수를 작성하세요.")
    
    if result["success"]:
        print("✅ 코드 생성 성공!")
        print(result["code"])
    else:
        print("❌ 오류:", result["error"])
