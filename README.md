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
## **참고 수학식**
---

## **수학적 접근 및 모델 구성**

본 프로젝트에서는 C 코드 내의 취약점 탐지를 위해 **Convolutional Neural Network (CNN)** 기반 접근 방식을 사용합니다. 이를 통해 소스 코드의 시퀀스 데이터를 벡터화하고, CNN을 통해 공간적 패턴을 학습합니다.

### **1. 데이터 전처리**

#### (1) 레이블 생성 (`create_combine_label`)
CWE 레이블을 기반으로 OR 연산을 통해 새로운 `combine` 레이블을 생성합니다. 이진 분류를 위해 아래와 같이 정의됩니다:

\[
combine_i = \max \{ CWE_{i,j} \mid j = 1, 2, ..., n \}
\]

여기서:
- \( combine_i \): \(i\)-번째 샘플의 레이블 값 (0 또는 1).
- \( CWE_{i,j} \): \(i\)-번째 샘플의 \(j\)-번째 CWE 레이블 값.

#### (2) 토큰화 및 패딩
코드 문자열 \( x \)는 정수 시퀀스 \( s \)로 변환되며, 고정된 입력 길이 \( L \)로 패딩됩니다:

1. **토큰화**:
   \[
   s = \text{Tokenizer}(x)
   \]
   여기서 \( s = [t_1, t_2, ..., t_k] \), \( t_i \)는 \(i\)-번째 단어의 정수 인덱스입니다.

2. **패딩**:
   \[
   s' = [t_1, t_2, ..., t_k, 0, 0, ..., 0] \quad \text{(길이 \( L \))}
   \]

---

### **2. CNN 모델 구성**

#### (1) 임베딩 레이어
입력 데이터 \( s' \)는 임베딩 레이어를 통해 고정된 차원의 벡터로 매핑됩니다:

\[
E = \text{Embedding}(s'; W_E)
\]

여기서:
- \( W_E \): 학습 가능한 임베딩 행렬 (\( |V| \times d \)), \( |V| \)는 어휘 크기, \( d \)는 임베딩 차원.

#### (2) 컨볼루션 연산
컨볼루션 레이어는 지역적 특징을 추출합니다:

\[
f_k = \sigma(W_k \cdot E + b_k)
\]

여기서:
- \( W_k \): \(k\)-번째 필터의 가중치.
- \( \sigma \): 활성화 함수 (\(\text{ReLU}\)).
- \( b_k \): 필터의 편향.

#### (3) 맥스 풀링
컨볼루션 결과를 차원 축소합니다:

\[
p_k = \max(f_k)
\]

#### (4) 완전 연결 레이어
최종적으로 모든 필터의 결과를 연결하여 이진 분류를 수행합니다:

\[
y = \sigma(W_o \cdot p + b_o)
\]

여기서:
- \( W_o \): 출력 레이어의 가중치.
- \( b_o \): 출력 레이어의 편향.
- \( \sigma \): 시그모이드 활성화 함수.

---

### **3. 손실 함수 및 최적화**

#### (1) 바이너리 크로스 엔트로피 손실
이진 분류의 손실 함수는 다음과 같이 정의됩니다:

\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]

여기서:
- \( N \): 샘플 수.
- \( y_i \): 실제 레이블 (0 또는 1).
- \( \hat{y}_i \): 모델의 예측 확률.

#### (2) 옵티마이저
Adam 옵티마이저를 사용하여 가중치를 업데이트합니다:
\[
\theta \gets \theta - \eta \cdot \nabla_\theta \mathcal{L}
\]

여기서:
- \( \eta \): 학습률.
- \( \nabla_\theta \mathcal{L} \): 손실 함수의 가중치에 대한 기울기.

---

### **4. 평가 지표**

#### (1) 정확도 (Accuracy)
정확도는 모델의 올바른 예측 비율로 정의됩니다:

\[
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
\]

#### (2) 혼동 행렬
혼동 행렬은 참/거짓 양성 및 음성의 분포를 나타냅니다:
\[
\begin{bmatrix}
TP & FP \\
FN & TN \\
\end{bmatrix}
\]

여기서:
- \( TP \): 참 긍정 (True Positive).
- \( FP \): 거짓 긍정 (False Positive).
- \( FN \): 거짓 부정 (False Negative).
- \( TN \): 참 부정 (True Negative).

#### (3) F1-Score
F1-Score는 Precision과 Recall의 조화 평균으로 계산됩니다:
\[
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

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

1. 실제 보안성이 뛰어난 코드 추천으로 업그레이드
2. GUI버젼으로 제

---

