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
    # 검증을 위해 파일로 우선 저장합니다 (GUI 블로킹 방지)
    plt.savefig('result_images/result_task1_sobel.png')
    print("성공的に result_task1_sobel.png 파일로 저장되었습니다.")
    
    # plt.show() # 실제 제출 시에는 이 주석을 풀어서 사용 가능

if __name__ == '__main__':
    main()
