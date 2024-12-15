```markdown
# **C 언어 소스 코드 취약점 탐지 (CNN 기반) - Artificial Intelligence C Language Vulnerability**

이 프로젝트는 C 언어 소스 코드의 취약점을 탐지하고 수정 권장 사항을 제공하기 위한 엔드-투-엔드 솔루션을 제공합니다. 소스 코드를 토큰화하고, CNN(Convolutional Neural Network) 모델을 학습시키며, 탐지된 취약점에 대한 수정을 제안합니다.

---

## **프로젝트 개요**

소스 코드 내의 취약점 탐지는 현대 사이버 보안에서 매우 중요한 과제입니다. 본 프로젝트는 소스 코드의 패턴을 분석하여, 수작업 규칙 없이 효율적으로 취약점을 탐지하는 것을 목표로 합니다. 사전에 레이블링된 데이터를 활용해 C 언어 소스 코드를 학습하고, 취약 여부를 분류하며, 탐지된 취약점에 대한 수정 권장 사항을 제공합니다.

이번 프로젝트에서는 **모델 구조와 가중치를 분리**하는 접근법을 채택했습니다. 이는 실험 과정에서의 유연성과 모듈화를 강화하며, 가중치만 저장하는 방식은 저장 용량을 효율적으로 사용하고 모델 재사용성을 높입니다.

---

## **프로젝트 구조**

```plaintext
AICV/
├── data/                          # HDF5 데이터 파일 디렉토리
│   ├── VDISC_train.hdf5           # 학습 데이터 (https://osf.io/d45bw/)
│   ├── VDISC_validate.hdf5        # 검증 데이터 (https://osf.io/d45bw/)
│   ├── VDISC_test.hdf5            # 테스트 데이터 (https://osf.io/d45bw/)
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

## **프로젝트 결과물**

```bash
Epoch 1/20
2024-12-08 01:08:23.260413: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8101
7965/7965 [==============================] - 430s 51ms/step - loss: 0.5336 - accuracy: 0.8995 - val_loss: 0.2630 - val_accuracy: 0.8927
Epoch 2/20
7965/7965 [==============================] - 448s 56ms/step - loss: 0.5012 - accuracy: 0.9049 - val_loss: 0.2525 - val_accuracy: 0.9065
...
Epoch 20/20
7965/7965 [==============================] - 470s 59ms/step - loss: 0.4001 - accuracy: 0.8713 - val_loss: 0.4065 - val_accuracy: 0.8712
```
```bash
분석 중: C:\AISS\example.c\example.c
1/1 [==============================] - 2s 2s/step
  🔍 예측 확률: 0.20407727360725403
  ⚠️ 취약점이 발견되었습니다:
  - CWE: CWE-120
    권장 수정 사항: 버퍼 오버플로를 방지하려면 strcpy 대신 strncpy 또는 snprintf를 사용하세요.
```
---

## **설치**

### **필요한 라이브러리**

- Python 3.8
- TensorFlow 2.11
- CUDA 11.2 및 cuDNN 8.1 (GPU 사용 시)
- 필수 라이브러리 설치:
  ```bash
  pip install tensorflow==2.11.* pandas scikit-learn matplotlib h5py
  ```

### **환경 설정**

1. 가상환경 생성:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Windows: myenv\Scripts\activate
   ```

2. 필수 라이브러리 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. GPU 지원 확인:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

---

## **데이터셋 설명**

### 데이터셋의 구성
데이터셋은 오픈 소스 소프트웨어에서 채굴된 약 127만 개의 함수로 구성된 소스 코드와 레이블 데이터입니다. 데이터셋은 다음과 같은 정보를 포함합니다:
1. **functionSource**: 함수 수준의 소스 코드 문자열.
2. **CWE-120, CWE-119, CWE-469, CWE-476, CWE-other**: CWE별 취약점 여부 (1: 취약, 0: 안전).
3. **combine**: 모든 CWE에 대해 OR 연산한 결과값으로, 이진 분류 학습의 목표값입니다.

---

## **코드 설명**

### **1. 데이터 전처리 (`preprocess.py`)**

#### 주요 기능:
1. **`load_data(file_path)`**: HDF5 데이터를 pandas DataFrame으로 로드.
2. **`create_combine_label(data)`**: CWE 레이블을 OR 연산하여 `combine` 컬럼 생성.
3. **`tokenize_data(tokenizer, data, input_size)`**: 코드를 토큰화하고 고정 크기로 패딩.

---

### **2. CNN 모델 정의 (`model.py`)**

#### 모델 구조:
- **Embedding 레이어**: 입력을 고정된 차원의 벡터로 매핑.
- **Conv1D 레이어**: 지역적 관계를 학습.
- **MaxPooling 레이어**: 차원 축소 및 중요 특징 캡처.
- **Dense 레이어**: 최종 이진 분류 수행.

---

### **3. 학습 스크립트 (`train.py`)**

#### 주요 기능:
- 데이터 로드 및 전처리.
- 모델 학습 및 가중치 저장.

---

### **4. 평가 스크립트 (`evaluate.py`)**

#### 주요 기능:
- 학습된 모델 성능 평가 (정확도, 혼동 행렬 등).

---

### **5. 코드 분석 및 수정 권장 (`detect_and_fix.py`)**

#### 주요 기능:
- C 코드 파일 분석 및 취약점 탐지.
- CWE에 대한 수정 권장 사항 출력.

---

## **확장 가능성**

1. **다중 레이블 분류로 확장 가능**:
   - CWE별 상세 취약점 분류.
2. **정적 분석 도구와의 통합**:
   - IDE 플러그인으로 개발 가능.
3. **모델 구조 최적화**:
   - 더 깊은 네트워크 또는 Transformer 기반 구조로 확장.

---

이 프로젝트는 효율적이고 실질적인 취약점 탐지를 목표로 설계되었습니다. 더 나은 보안성을 위해 새로운 기능이나 개선 사항이 필요하다면 언제든지 기여를 환영합니다! 😊
```
