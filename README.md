# 컴퓨터 비전 OpenCV 실습 (Edge and Region)

컴퓨터 비전 수업의 에지 검출 및 영역 분할 실습 과제 저장소입니다.  
총 3개의 실습(Sobel Edge, Canny & Hough Transform, GrabCut)으로 구성되어 있으며, 컴퓨터 비전에서 물체를 인식하기 위한 가장 기본적인 단서인 '경계(Boundary)'를 다루는 기법을 배웁니다.

---

## 환경 설정

| 항목 | 버전/도구 |
|---|---|
| Python | 3.10+ |
| 패키지 관리 | Anaconda (conda) |
| 주요 라이브러리 | `opencv-python`, `numpy`, `matplotlib` |

```bash
# conda 가상환경 생성 및 활성화
conda create -n cv python=3.10
conda activate cv

# 필요한 패키지 설치
pip install opencv-python numpy matplotlib
```

## 실행 방법

프로젝트 루트(`computer_vison/3/`)에서 실행:

```bash
python task1_sobel.py
python task2_hough.py
python task3_grabcut.py
```

---

## 실습 01 — Sobel Edge Detection 

### 과제 설명

이미지를 흑백(Grayscale)으로 변환한 뒤, 소벨(Sobel) 필터를 이용해 x축 방향과 y축 방향의 밝기 변화량(미분)을 각각 검출하고 이를 종합하여 최종 에지 강도(Magnitude) 맵을 출력합니다.

- **목적**: 픽셀 밝기가 급격히 변하는 지점을 1차 미분을 통해 수학적으로 검출해냅니다.
- **필수 요구사항 및 제약사항**:
    - **자료형 보존**: 소벨 미분 연산 시 `cv.CV_64F` 형태의 64비트 실수형 연산을 적용해야 음수 미분값의 데이터 소실을 막을 수 있습니다.
    - **강도 계산**: `cv.magnitude()`를 통해 $root(x_{edge}^2 + y_{edge}^2)$ 값을 구해냅니다.
    - **시각화 캐스팅**: 에지 강도를 화면에 정상적으로 표시하기 위해 반드시 `cv.convertScaleAbs()`를 거쳐 8비트 양수 영상(uint8)으로 변환해야 합니다.

### 핵심 코드 설명

```python
# [핵심] 미분값이 음수가 나올 수 있으므로 반드시 64비트 실수형(CV_64F) 지정
grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# [핵심] X축과 Y축 미분값을 바탕으로 최종 에지 강도 산출
magnitude = cv.magnitude(grad_x, grad_y)

# [핵심] 화면상에 시각화하기 위해 절대값을 취해 8비트 양수로 캐스팅
magnitude_8u = cv.convertScaleAbs(magnitude)
```

> **포인트**: OpenCV 연산 시 수학적인 음수 픽셀값이나 오버플로우를 막기 위해 실수형으로 연산한 뒤 화면 표시 직전에 8비트로 축소하는 기법은 실무에서 디버깅 시 자주 마주치는 아주 중요한 사항입니다.

### 전체 코드

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    img_path = 'images/edgeDetectionImage.jpg'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    # 1. 원본 이미지 로드
    img = cv.imread(img_path)
    
    # 2. 흑백(Grayscale) 영상으로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 3. 소벨 필터를 이용한 미분 (CV_64F 자료형 사용 - 핵심 요구사항)
    # x축 방향 에지
    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    # y축 방향 에지
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    
    # 4. x축, y축 미분값을 종합하여 에지 강도(Magnitude) 계산
    magnitude = cv.magnitude(grad_x, grad_y)
    
    # 5. 화면 표시를 위한 8비트 캐스팅 (CV_64F -> uint8 절대값 처리)
    magnitude_8u = cv.convertScaleAbs(magnitude)
    
    # 6. 시각화 (Matplotlib)
    # BGR 이미지를 RGB로 변환하여 Matplotlib에서 정상적인 색상으로 표시
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_8u, cmap='gray')
    plt.title('Sobel Edge Magnitude')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('result_images/result_task1_sobel.png')
    
if __name__ == '__main__':
    main()
```

### 최종 결과물

![task1_result](result_images/result_task1_sobel.png)

---

## 실습 02 — Canny Edge & Hough Transform

### 과제 설명

캐니 에지(Canny Edge) 검출 알고리즘을 사용해 잡음이 제거된 얇은 윤곽선을 추출한 뒤, 허프 변환(Hough Transform) 알고리즘을 적용해 수학적으로 유효한 '직선' 형태만 찾아냅니다.

- **변환 조건**:
    - **캐니 임계값**: 교수님 힌트에 따라 `threshold1=100`, `threshold2=200` 적용.
    - **직선 렌더링**: 검출된 좌표 바탕으로 빨간색 선(`(0, 0, 255)`), 두께 `2` 렌더링 적용.
    - **파라미터 튜닝**: `cv.HoughLinesP()` 함수의 교차점 임계값과 선분 길이 관련 파라미터는 실험적인 튜닝이 쉽게 별도 변수로 분리하여 관리.

### 핵심 코드 설명

```python
# [핵심] Canny 엣지 검출 알고리즘 
edges = cv.Canny(gray, threshold1=100, threshold2=200)

# [핵심] 튜닝에 용이하도록 허프 변환 제어 파라미터 변수화
hough_threshold = 50            # 직선으로 판단할 최소 교차점 수
minLineLength = 50              # 선분의 최소 픽셀 길이
maxLineGap = 10                 # 선분 사이의 최대 허용 여백

# [핵심] 확률적 허프 변환 적용하여 시작값/종료값 좌표 추출
lines = cv.HoughLinesP(edges, rho, theta, hough_threshold, 
                       minLineLength=minLineLength, maxLineGap=maxLineGap)
```

> **포인트**: `cv.HoughLinesP`는 거대한 2D 수학적 공간(매개변수 공간) 투표를 통해 선분을 찾아내며, 원본 해상도와 노이즈 수준에 따라 `hough_threshold`, `minLineLength`, `maxLineGap` 세 가지 파라미터의 민감도가 결정된다. 

### 전체 코드

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    img_path = 'images/dabo.jpg'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    # 1. 원본 이미지 로드
    img = cv.imread(img_path)
    
    # 2. BGR -> Grayscale 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 3. Canny 알고리즘으로 에지 검출 (요구사항: threshold1=100, threshold2=200)
    edges = cv.Canny(gray, threshold1=100, threshold2=200)
    
    # 4. 확률적 허프 변환(HoughLinesP) 파라미터 튜닝을 위한 변수화 선언
    rho = 1                         
    theta = np.pi / 180             
    hough_threshold = 50            
    minLineLength = 50              
    maxLineGap = 10                 
    
    lines = cv.HoughLinesP(edges, rho, theta, hough_threshold, 
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
                           
    # 5. 원본 이미지의 복사본 위에 빨간색(0, 0, 255) 두께 2로 찾은 선 그리기
    result_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    # 6. Matplotlib 시각화를 위해 원본 포맷(BGR)을 RGB로 각각 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)
    
    # 7. 시각화 (원본 이미지 vs 결과 이미지)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_img_rgb)
    plt.title('Hough Lines Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('result_images/result_task2_hough.png')

if __name__ == '__main__':
    main()
```

### 최종 결과물

![task2_result](result_images/result_task2_hough.png)

---

## 실습 03 — GrabCut Interactive Segmentation

### 과제 설명

대화식 영역 분할 툴인 GrabCut 알고리즘을 스크립트로 제어하여, 사용자가 이미지 위 특정 영역의 직사각형을 넘겨주면, 가장 눈에 띄는 물체(전경)를 식별하고 나머지 배경을 지워내는 실습입니다.

- **구현 스텝**:
    - **Step 1: GrabCut 모델 변수 설정**: `cv.grabCut()`를 호출하기 이전에 반드시 `bgdModel`과 `fgdModel`을 $1 \times 65$ 형태의 64비트 실수형 행렬(`np.float64`)로 초기화하여 넘겨줍니다.
    - **Step 2: 동적 Bounding Box 크기 책정**: 해상도 픽셀 값을 직접 하드코딩하는 오류를 막기 위해 원본 사진의 $W$, $H$ 대비 % 수치를 이용해서 대상(컵)을 중심으로 하는 넉넉한 사각형 `(x, y, w, h)` 영역을 산출합니다.
    - **Step 3: 마스크 후처리 연산**: `cv.grabCut()`이 반환한 `mask` 배열 내부의 분류값(0, 1, 2, 3)을 `np.where()` 기법을 통해 대상과 배경 0과 1의 이진 데이터로 최종 합산합니다.
    - **Step 4: 배경 제거**: $H \times W \times 1$ 크기로 늘어난 이진 마스크 행렬과 원본 컬러 이미지 행렬을 스칼라 곱셈(`*`) 하여 배경을 차폐합니다.

### 핵심 코드 설명

```python
# [핵심] GrabCut 구동을 위한 내부 Gaussian Mixture Model 필수 초기화
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# [핵심] 하드코딩 오차 방지를 위해 이미지 비율에 따른 동적 사각형 구역 산출
h, w = img.shape[:2]
rect = (int(w * 0.15), int(h * 0.1), int(w * 0.7), int(h * 0.8))

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# [핵심] Numpy의 where를 활용하여 GrabCut 마스크 값들을 0(배경)과 1(전경)로 분리 
mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# [핵심] 배경 부분 행렬을 0으로 만들어 까맣게 차단
result_img = img * mask_binary[:, :, np.newaxis]
```

> **포인트**: 마스크를 0과 1로 구분하는 `mask_binary` 배열 단일 연산 과정은 반복문보다 훨씬 빠르며, RGB 컬러 3채널과 크기를 맞추기 위해 `np.newaxis`를 이용해 차원을 늘리는 트릭이 컴퓨터 비전 파이썬 엔지니어링의 핵심이다.

### 전체 코드

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    img_path = 'images/coffee cup.JPG'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    # 1. 원본 이미지 로드
    img = cv.imread(img_path)
    
    # 2. GrabCut 알고리즘을 위한 초기 변수 세팅
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # 필수 요구사항: 내부 모델 초기화
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 3. 객체가 있을 법한 관심 영역(Rect) 하드코딩
    h, w = img.shape[:2]
    x, y = int(w * 0.15), int(h * 0.1)
    rect_w, rect_h = int(w * 0.7), int(h * 0.8)
    rect = (x, y, rect_w, rect_h)
    
    # 4. GrabCut 실행 (5번 반복)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    
    # 5. 마스크 후처리 (매우 중요)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # 6. 배경이 제거된 추출 이미지 생성
    result_img = img * mask_binary[:, :, np.newaxis]
    
    # 7. 시각화를 위한 BGR -> RGB 포맷 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)
    
    # 8. Matplotlib을 이용한 결과 3종 나란히 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_binary * 255, cmap='gray')   
    plt.title('GrabCut Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_img_rgb)
    plt.title('Background Removed')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('result_images/result_task3_grabcut.png')

if __name__ == '__main__':
    main()
```

### 최종 결과물

![task3_result](result_images/result_task3_grabcut.png)
