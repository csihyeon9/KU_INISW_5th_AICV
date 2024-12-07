import os
import tensorflow as tf
from preprocess import tokenize_data
from model import build_model
import re
import pandas as pd

# Constants
INPUT_SIZE = 500
VOCAB_SIZE = 10000
MODEL_PATH = '../saved_models/model.weights.h5'
TARGET_FOLDER = '../example.c/'  # 분석할 C 파일들이 있는 폴더 경로

# CWE 수정 권장사항
CWE_FIXES = {
    "CWE-120": "버퍼 오버플로를 방지하려면 strcpy 대신 strncpy 또는 snprintf를 사용하세요.",
    "CWE-119": "버퍼 경계를 초과하지 않도록 배열 인덱스를 검사하세요.",
    "CWE-469": "포인터 계산에 정확한 데이터 크기를 사용하세요.",
    "CWE-476": "NULL 포인터를 사용하기 전에 항상 NULL 여부를 확인하세요.",
    "CWE-other": "코드 전체를 검토하고 입력 검증 또는 예외 처리를 추가하세요."
}

def load_model(input_size, vocab_size):
    """학습된 모델 로드"""
    model = build_model(input_size, vocab_size)
    model.load_weights(MODEL_PATH)
    return model

def remove_comments(code):
    """
    C 코드에서 주석을 제거합니다.
    Args:
        code (str): 입력된 C 코드.
    Returns:
        str: 주석이 제거된 C 코드.
    """
    # 정규식을 사용하여 주석 제거 (/* */ 또는 // 스타일)
    pattern = r'(\/\*[\s\S]*?\*\/)|(\/\/.*)'
    return re.sub(pattern, '', code)

def analyze_code(code, tokenizer, model):
    """
    입력된 코드의 취약점을 분석합니다.
    Args:
        code (str): 분석할 C 코드.
        tokenizer (Tokenizer): Keras 토크나이저.
        model (tf.keras.Model): 학습된 모델.
    Returns:
        dict: 취약점 탐지 결과 및 추천 수정 방법.
    """
    # 전처리 및 토큰화
    tokenized_code = tokenize_data(tokenizer, pd.DataFrame({"functionSource": [code]}), INPUT_SIZE)

    # 예측
    prediction = model.predict(tokenized_code)[0][0]
    print(f"  🔍 예측 확률: {prediction}")  # 예측 확률 출력
    is_vulnerable = prediction > 0.1

    if not is_vulnerable:
        return {"vulnerable": False, "recommendations": []}

    # CWE 타입 추정 및 수정 권장사항
    recommendations = []
    for cwe_id, fix in CWE_FIXES.items():
        if re.search(cwe_id, code, re.IGNORECASE):  # 코드 내 CWE 패턴 확인
            recommendations.append({"CWE": cwe_id, "fix": fix})

    return {"vulnerable": True, "recommendations": recommendations}


def analyze_folder(folder_path, tokenizer, model):
    """
    지정된 폴더 내 모든 `.c` 파일을 분석합니다.
    Args:
        folder_path (str): 분석할 폴더 경로.
        tokenizer (Tokenizer): Keras 토크나이저.
        model (tf.keras.Model): 학습된 모델.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.c'):
                file_path = os.path.abspath(os.path.join(root, file))
                print(f"\n분석 중: {file_path}")

                # 파일 읽기
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                except Exception as e:
                    print(f"  ⚠️ 파일을 읽는 중 오류 발생: {e}")
                    continue

                # 코드 분석
                result = analyze_code(code, tokenizer, model)

                # 결과 출력
                if not result["vulnerable"]:
                    print("  ✅ 이 코드는 취약하지 않은 것으로 보입니다.")
                else:
                    print("  ⚠️ 취약점이 발견되었습니다:")
                    for rec in result["recommendations"]:
                        print(f"  - CWE: {rec['CWE']}")
                        print(f"    권장 수정 사항: {rec['fix']}")


def main():
    # 토크나이저 및 모델 로드
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts([""])  # 빈 토크나이저 초기화
    model = load_model(INPUT_SIZE, VOCAB_SIZE)

    # 폴더 분석 실행
    if not os.path.exists(TARGET_FOLDER):
        print(f"폴더가 존재하지 않습니다: {TARGET_FOLDER}")
    else:
        analyze_folder(TARGET_FOLDER, tokenizer, model)

if __name__ == "__main__":
    main()
