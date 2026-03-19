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
    # 튜닝이 쉽게 별도 변수로 분리합니다.
    rho = 1                         # 거리 해상도 (픽셀)
    theta = np.pi / 180             # 각도 해상도 (라디안)
    hough_threshold = 50            # 직선으로 판단할 최소 교차점 수
    minLineLength = 50              # 선분의 최소 길이 (단위: 픽셀)
    maxLineGap = 10                 # 선분 사이의 최대 허용 간격 (이 간격 이내면 하나의 선으로 취급)
    
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
    # 검증을 위해 파일로 우선 저장합니다 (GUI 블로킹 방지)
    plt.savefig('result_images/result_task2_hough.png')
    print("성공적으로 result_task2_hough.png 파일로 저장되었습니다.")
    
    # plt.show() # 실제 제출 시에는 이 주석을 풀어 사용

if __name__ == '__main__':
    main()
