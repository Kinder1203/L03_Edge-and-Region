import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

def main(): 
    # 이미지가 저장된 상대 경로를 문자열로 지정합니다.
    img_path = 'images/edgeDetectionImage.jpg' 
    
    # 지정한 경로에 파일이 실제로 존재하는지 1차적으로 확인합니다.
    if not os.path.exists(img_path): 
        print(f"Error: {img_path} not found.") # 파일이 없다면 에러 메시지를 터미널에 출력합니다.
        return # 더 이상 불필요한 연산을 진행하지 않고 함수를 강제 종료합니다.

    # 1. 파일 경로로부터 이미지를 복호화하여 BGR 포맷의 NumPy 다차원 배열로 읽어옵니다.
    img = cv.imread(img_path) 
    
    # 2. 이미지의 밝기 변화량만을 추적하기 위해 컬러 프레임을 단일 채널인 흑백(Grayscale) 영상으로 변환합니다.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    
    # 3. 소벨(Sobel) 필터를 적용하여 x축(수평) 방향의 밝기 변화량 곡면(미분)을 구합니다.
    # [중요 제약] 미분값이 음수로 떨어지는 데이터 손실을 방지하기 위해 출력 자료형을 64비트 실수형(cv.CV_64F)으로 둡니다.
    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3) 
    
    # 동일한 방식으로 y축(수직) 방향의 밝기 변화량(에지)을 소벨 필터로 검출합니다.
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3) 
    
    # 4. 구해진 x축, y축 두 방향의 미분값을 벡터 합성(Magnitude)하여 최종 에지 강도를 계산합니다.
    magnitude = cv.magnitude(grad_x, grad_y) 
    
    # 5. 넓은 범위의 실수형으로 계산된 강도에 절댓값을 취하고, 시각적 디스플레이가 가능한 8비트 정수형(uint8)으로 자동 축소 캐스팅합니다.
    magnitude_8u = cv.convertScaleAbs(magnitude) 
    
    # 6. Matplotlib으로 원본 비교 시 푸르게 나오는 현상을 막고자, 원본의 BGR 채널들을 RGB 포맷으로 재정렬합니다.
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    
    # 출력물을 띄울 가로 10인치, 세로 5인치 크기의 캔버스(도화지)를 설정합니다.
    plt.figure(figsize=(10, 5)) 
    
    # 1행 2열 레이아웃 구조를 잡은 다음 그 중 첫 번째(왼쪽) 그리기 영역을 선택합니다.
    plt.subplot(1, 2, 1) 
    # 원본(RGB) 배열 이미지를 왼쪽 영역에 렌더링합니다.
    plt.imshow(img_rgb) 
    # 왼쪽 그림 상단에 'Original Image'라는 영문 제목표를 달아줍니다.
    plt.title('Original Image') 
    # 불필요한 주변 XY 그래프 눈금과 테두리 선들을 보이지 않게 꺼버립니다.
    plt.axis('off') 
    
    # 1행 2열 레이아웃 구조 중 두 번째(오른쪽) 그리기 영역으로 포커스를 이동합니다.
    plt.subplot(1, 2, 2) 
    # 검출한 최종 에지 강도 영상을 흑백 컬러맵(gray) 옵션과 함께 오른쪽 영역에 렌더링합니다.
    plt.imshow(magnitude_8u, cmap='gray') 
    # 오른쪽 그림 상단에 'Sobel Edge Magnitude'라는 영문 제목을 지정합니다.
    plt.title('Sobel Edge Magnitude') 
    # 역시 어색함을 주는 좌표 눈금선과 박스를 화면에서 숨깁니다.
    plt.axis('off') 
    
    # 켜진 여러 시각화 컴포넌트 간 여백 및 글씨 겹침을 내부 엔진이 적절히 자동 조절해줍니다.
    plt.tight_layout() 
    
    # 최종적으로 세팅된 비교 컴포넌트들을 미리 지정해둔 폴더 안에 이미지(.png) 파일로 보존합니다.
    plt.savefig('result_images/result_task1_sobel.png') 
    # 파이썬 저장 코드가 정상적으로 수행 완료되었음을 콘솔(터미널)을 통해 운영자에게 알립니다.
    print("성공적으로 result_images/result_task1_sobel.png 파일로 저장되었습니다.") 
    
    # plt.show() # (팝업 이미지를 현장에서 시각화할 때 주석을 제거하고 띄우기 용도로 둡니다)

if __name__ == '__main__': # 현재 구동 환경이 외부 모듈 참조가 아닌 스크립트 단독 직접 실행인지 판단하는 파이썬 표준 관례입니다.
    main() # 직접 실행된 경우에만 핵심 루틴인 main 함수를 작동시킵니다.
