```markdown
# **C 언어 소스 코드 취약점 탐지 (CNN 기반) - Artificial Intelligence C language Vulnerability**

이 프로젝트는 C 언어 소스 코드의 취약점을 탐지하고 수정 권장 사항을 제공하기 위한 엔드-투-엔드 솔루션을 제공합니다. 소스 코드를 토큰화하고, CNN(Convolutional Neural Network) 모델을 학습시키며, 탐지된 취약점에 대한 수정을 제안합니다.

---

## **프로젝트 개요**

소스 코드 내의 취약점 탐지는 현대 사이버 보안에서 매우 중요한 과제입니다. 본 프로젝트는 소스 코드의 패턴을 분석하여, 수작업 규칙 없이 효율적으로 취약점을 탐지하는 것을 목표로 합니다. 사전에 레이블링된 데이터를 활용해 C 언어 소스 코드를 학습하고, 취약 여부를 분류하며, 탐지된 취약점에 대한 수정 권장 사항을 제공합니다.

AI 전문가의 관점에서 **모델 구조와 가중치를 분리**하는 접근법을 채택했습니다. 이는 실험 과정에서의 유연성과 모듈화를 강화하며, 가중치만 저장하는 방식은 저장 용량을 효율적으로 사용하고 모델 재사용성을 높입니다.

---

## **프로젝트 구조**

```plaintext
AISS/
├── data/                          # HDF5 데이터 파일 디렉토리
│   ├── VDISC_train.hdf5           # 학습 데이터
│   ├── VDISC_validate.hdf5        # 검증 데이터
│   ├── VDISC_test.hdf5            # 테스트 데이터
├── logs/                          # TensorBoard 로그 디렉토리
├── saved_models/                  # 모델 가중치 저장 디렉토리
│   └── model.weights.h5           # 최적 모델 가중치
├── example.c/                     # 분석 대상 C 코드 디렉토리
│   ├── example1.c
│   ├── example2.c
│   ├── ...
├── src/                           # 소스 코드 디렉토리
│   ├── preprocess.py              # 데이터 전처리 유틸리티
│   ├── model.py                   # CNN 모델 정의
│   ├── train.py                   # 학습 스크립트
│   ├── evaluate.py                # 평가 스크립트
│   ├── detect_and_fix.py          # 코드 분석 및 수정 권장 스크립트
│   ├── visualization.py           # 학습 결과 시각화 스크립트
├── README.md                      # 프로젝트 문서
```

---

## **설치**

### **필요한 라이브러리**

- Python 3.8
- TensorFlow 2.11
- CUDA 11.2 (GPU를 사용할 경우 cuDNN 8.1 필요)
- 필수 Python 라이브러리 설치:
  ```bash
  pip install tensorflow==2.11.* pandas scikit-learn matplotlib h5py
  ```

### **환경 설정**

1. 가상환경 생성:
   ```bash
   py -3.8 -m venv ./venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. 필수 라이브러리 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. GPU 지원 확인:
   - CUDA와 cuDNN이 제대로 설치되었는지 확인한 뒤, GPU를 사용 가능한지 테스트:
     ```bash
     python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
     ```

---

## **데이터셋 설명**

데이터셋은 C/C++ 함수 수준의 코드 예제와 취약점 레이블을 포함한 HDF5 파일로 제공됩니다. 각 파일에는 다음과 같은 데이터가 포함됩니다:

1. **`functionSource`**: C 코드 함수의 소스 문자열.
2. **`CWE-120`, `CWE-119`, `CWE-469`, `CWE-476`, `CWE-other`**: 취약점 유형에 대한 바이너리 레이블 (1: 취약, 0: 안전).
3. **`combine`**: 모든 CWE에 대해 OR 연산을 수행한 결과로, 이진 분류 학습을 위한 목표 변수입니다.

데이터셋 파일:
- `VDISC_train.hdf5`: 학습 데이터.
- `VDISC_validate.hdf5`: 검증 데이터.
- `VDISC_test.hdf5`: 테스트 데이터.

---

## **코드 설명**

### **1. 데이터 전처리 (`preprocess.py`)**

#### 주요 기능:
- **`load_data(file_path)`**: HDF5 파일에서 데이터를 로드하여 pandas DataFrame으로 변환.
- **`create_combine_label(data)`**: CWE 레이블을 OR 연산으로 결합하여 `combine` 컬럼 생성 (이진 분류용).
- **`tokenize_data(tokenizer, data, input_size)`**: 소스 코드를 토큰화하고 패딩 처리.

---

### **2. CNN 모델 정의 (`model.py`)**

#### 모델 구조:
- **Embedding 레이어**: 토큰을 고정된 크기의 벡터로 매핑.
- **Conv1D 레이어**: 토큰 시퀀스에서 지역 패턴을 학습.
- **MaxPooling 레이어**: 차원 축소와 중요한 패턴 캡처.
- **Dense 레이어**: 완전 연결 레이어로 이진 분류 수행.

---

### **3. 학습 스크립트 (`train.py`)**

#### 실행:
```bash
python src/train.py
```

#### 주요 기능:
1. 학습 및 검증 데이터 로드.
2. Keras 토크나이저로 소스 코드 토큰화.
3. CNN 모델 학습.
4. 최적의 모델 가중치를 `model.weights.h5`로 저장.

#### 학습 결과 지표:
- **Loss**: 학습 데이터의 손실 값.
- **Accuracy**: 학습 데이터의 정확도.
- **Val_loss**: 검증 데이터의 손실 값.
- **Val_accuracy**: 검증 데이터의 정확도.

#### 지표의 의미:
- **Loss/Val_loss**: 손실 값은 모델 예측과 실제 값 간의 차이를 수치화. 검증 손실은 모델이 학습 데이터 외부에서도 잘 동작하는지 평가.
- **Accuracy/Val_accuracy**: 정확도는 올바른 예측 비율을 나타내며, 검증 정확도는 일반화 성능을 나타냄.

---

### **4. 평가 스크립트 (`evaluate.py`)**

#### 실행:
```bash
python src/evaluate.py
```

#### 주요 기능:
- 테스트 데이터 로드 및 전처리.
- 학습된 모델 가중치를 로드(`model.weights.h5`).
- 모델 성능 평가:
  - 정확도 출력.
  - 혼동 행렬 및 분류 보고서 생성.

---

### **5. 코드 분석 및 수정 권장 스크립트 (`detect_and_fix.py`)**

#### 실행:
```bash
python src/detect_and_fix.py
```

#### 주요 기능:
1. `example.c/` 디렉토리 내의 모든 `.c` 파일을 자동으로 분석.
2. C 코드의 **주석 제거** 후 코드 내용을 분석하여 취약점 탐지.
3. 탐지된 CWE에 대해 수정 권장 사항 출력.

---

### **6. 학습 결과 시각화 스크립트 (`visualization.py`)**

#### 실행:
```bash
python src/visualization.py
```

#### 주요 기능:
- TensorBoard 로그 데이터를 기반으로 학습/검증 손실 및 정확도를 시각화.
- 학습 과정에서 모델이 어떻게 개선되었는지 확인 가능.

---

## **확장 가능성**

1. **성능 최적화**:
   - 더 많은 학습 데이터 사용.
   - 모델 구조 개선 (예: RNN 추가, 더 깊은 CNN).

2. **다중 레이블 분류**:
   - CWE별 다중 레이블 분류로 확장 가능.

3. **추가 평가 지표**:
   - Precision-Recall AUC, ROC AUC 분석 포함.

---

### **결론**

이 프로젝트는 취약점 탐지 및 수정 권장 사항을 제공하는 완성형 워크플로를 제공합니다. AI 전문가의 관점에서, 주석 제거 및 가중치 분리 접근법은 모델 성능을 최적화하고 실질적인 활용도를 높이는 데 기여합니다. 추가적인 지원이나 질문이 있다면 언제든 환영합니다! 😊
```