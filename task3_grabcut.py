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
    # 커피 대상이 가운데에 위치한다고 가정하고 이미지 비율을 이용해 대략적인 박스를 잡음
    h, w = img.shape[:2]
    x, y = int(w * 0.15), int(h * 0.1)
    rect_w, rect_h = int(w * 0.7), int(h * 0.8)
    rect = (x, y, rect_w, rect_h)
    
    # 4. GrabCut 실행 (5번 반복)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    
    # 5. 마스크 후처리 (매우 중요)
    # mask 값: 0(배경), 1(전경), 2(아마도 배경), 3(아마도 전경)
    # 0과 2는 0으로, 1과 3은 1로 만들어 이진 마스크 생성
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # 6. 배경이 제거된 추출 이미지 생성
    # 원본 이미지에 이진 마스크를 곱하여 배경을 까맣게 날려버림
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
    plt.imshow(mask_binary * 255, cmap='gray')   # 0, 1 배열이므로 255를 곱해 완전한 흑/백 이미지로 시각화
    plt.title('GrabCut Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_img_rgb)
    plt.title('Background Removed')
    plt.axis('off')
    
    plt.tight_layout()
    # 검증을 위해 파일로 우선 저장 (GUI 블로킹 방지)
    plt.savefig('result_images/result_task3_grabcut.png')
    print("성공적으로 result_images/result_task3_grabcut.png 파일로 저장되었습니다.")
    
    # plt.show() # 실제 제출 시 주석 해제하여 결과를 팝업으로 시각화

if __name__ == '__main__':
    main()
