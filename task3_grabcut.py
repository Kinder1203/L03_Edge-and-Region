import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
import os 

def main():
    # 대상으로 다뤄질 원본 이미지 형태인 컵 사진의 상대 경로를 스트링 데이터로 저장합니다.
    img_path = 'images/coffee cup.JPG' 
    
    # 파일 탐색기가 현재 작업 중인 공간(CWD) 내에 해당 사진이 실체하는가 안전하게 우선 질의합니다.
    if not os.path.exists(img_path): 
        print(f"Error: {img_path} not found.") # 에이전트 경로 불일치 등에 의한 실행중단을 막고자 단서 메시지를 출력합니다.
        return # 더 이상의 코드 무한 진행과 파괴적 크래시를 막고 즉시 이 함수를 완전히 빠져나옵니다.

    # 1. 뼈대인 원본 이미지를 지정한 위치로부터 BGR 3채널의 색조 해상도를 지닌 숫자 집합(numpy배열) 형태로 생성합니다.
    img = cv.imread(img_path) 
    
    # 2. GrabCut 알고리즘 내내 지속적으로 분류 기준값이 담길 공백 상태의 도화지(마스크) 배열을 사진 가로세로 규격과 동일치로 초기화시킵니다 (0 채움).
    mask = np.zeros(img.shape[:2], np.uint8) 
    
    # ※ 가장 중요한 요구사항 조치구간: GrabCut 백그라운드 구동에 필수적으로 동반될 가우시안 믹스처 모델(GMM)의 뼈대 변수를 빈 실수형 65차원으로 초기화합니다.
    bgdModel = np.zeros((1, 65), np.float64) 
    fgdModel = np.zeros((1, 65), np.float64) 
    
    # 3. 객체(컵)가 정확히 머물고 있을 것으로 짐작되는 초기 바운딩 사각형(Rect) 지역을 추론하기 귀해 해상도를 추출합니다.
    h, w = img.shape[:2] # 높이(Height)와 가로 폭(Width) 치수를 투플 속성에서 끄집어 가져옵니다.
    # 임의 숫자를 하드코딩하지 않고, 이미지 가로폭의 15% 기점(x), 높이의 10% 기점(y) 같이 동적인 화면 변수 기준으로 좌표 이동을 도출합니다.
    x, y = int(w * 0.15), int(h * 0.1) 
    # 가로는 영상 너비의 70%만큼 채우고, 세로는 80%를 아우르는 매우 거대한 사각형 크기를 설정합니다.
    rect_w, rect_h = int(w * 0.7), int(h * 0.8) 
    # 산출해낸 4가지 수치를 단일 튜플 구조(x, y, w, h)로 변환해서 하나로 강하게 뭉쳐줍니다.
    rect = (x, y, rect_w, rect_h) 
    
    # 4. 방금 추론한 렉트 좌표(Rect)를 중심으로 주변은 무조건 배경치고 안쪽은 전경 후보로 상정하여 분류기(GrabCut)를 5사이클 동안 반복 학습하며 실행합니다.
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT) 
    
    # 5. 분류가 끝난 마스크 안에는 네 가지 라벨링 정수(0:배경, 1:전경, 2:배경일듯, 3:전경일듯)가 복잡하게 산재해 있습니다. 후처리 정리가 시급합니다.
    # Numpy의 where 불리언 잣대를 사용하여 값이 2(배경일듯) 와 0(순수배경)일 경우 가차없이 0(검은색)으로 치환하고 나머지를 1로 바꿔 완전한 이진 데이터 파이프라인으로 캐스팅합니다.
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') 
    
    # 6. 원본 이미지 행렬 스칼라에 이진 마스크 행렬을 무작위 곱셈하여, 배경 영역(0)은 무색 암흑화시키고 컵 영역(1)만 밝기 100%를 통과시키도록 오버레이시킵니다.
    # 이때 3차원 컬러 사진과 차원 축을 맞추고자 np.newaxis 트릭으로 억지로 마스크의 채널을 1개 증폭시켜줍니다.
    result_img = img * mask_binary[:, :, np.newaxis] 
    
    # 7. 화면(Matplotlib)에 오버랩 시 색 정보가 멍든 것처럼 청색으로 물드는 참사를 막기 위해 BGR 파장축을 RGB로 순서를 스위칭해줍니다 (원본 프레임 대응).
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    # 배경 따내기가 완료되어 완성된 알맹이 컵 사진도 동일하게 RGB 체제로 바꿔 통일감을 기합니다.
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB) 
    
    # 8. 대규모 도해도(가로 15, 세로 5)를 설계하여 3가지 상태(원본, 마스크, 적용 결과)를 연속으로 보여줄 모니터 환경을 오픈합니다.
    plt.figure(figsize=(15, 5)) 
    
    # 세 폭의 분할 칸막이 중 제일 으뜸(첫 번째) 진영을 점유합니다.
    plt.subplot(1, 3, 1) 
    # 이 구역에 색 교환을 마친 티없이 깨끗한 원본 RGB 이미지를 놓습니다.
    plt.imshow(img_rgb) 
    # 위쪽에 관람자가 구별 가능하게 끔 'Original Image' 명판을 씌웁니다.
    plt.title('Original Image') 
    # 그래프 용도로 생성된 수평 수직 가로 눈금들을 철저하게 비가시화(off)시킵니다.
    plt.axis('off') 
    
    # 세 폭의 분할 칸막이 중 허리 역할을 하는 넘버투(두 번째) 진영을 차지합니다.
    plt.subplot(1, 3, 2) 
    # 0과 1로 포장되어 칠흑같이 어둡게 된 이진 마스크에 고의적으로 255를 전면 곱해, 배경은 검정, 컵은 완벽한 하얀색으로 보이게 한 뒤, 흑백 필터 맵(gray)으로 송출합니다.
    plt.imshow(mask_binary * 255, cmap='gray')   
    # 명확하게 'GrabCut Mask'라는 제목 레이블을 삽입합니다.
    plt.title('GrabCut Mask') 
    # 이곳 역시 좌표 축과 숫자를 허공으로 날려버립니다.
    plt.axis('off') 
    
    # 마지막 분할 칸막이, 대미를 장식할 오른쪽 최고 꼬리 진영(세 번째)을 장악합니다.
    plt.subplot(1, 3, 3) 
    # 최종적으로 주변이 까맣게 날아가고 컵 형상만 아스라이 남은 스크린(결과 행렬)을 매핑합니다.
    plt.imshow(result_img_rgb) 
    # 이 그림 위에도 'Background Removed'라는 유식한 타이틀을 달아둡니다.
    plt.title('Background Removed') 
    # 그래프 외형 장식을 배제하고 그림 자체의 미관만 노출시킵니다.
    plt.axis('off') 
    
    # 3개의 대화면 요소 배열의 중복이나 글씨 침범을 막고자 엔진 스스로 비율과 줄맞춤을 촘촘하게 동기화하도록 구속(tight) 명령을 던집니다.
    plt.tight_layout() 
    # 현재 만들어진 최종 조합 그림판을, result_images 디렉토리 경로에 안전한 범용 파일(.png) 포맷으로 영구 구워냅니다.
    plt.savefig('result_images/result_task3_grabcut.png') 
    # 이 모든 유기적 파이프라인 처리가 탈 없게 돌아가고 저장이 종결됐음을 사람 제어자에게 알림 스피커 통보하듯 콘솔로 띄웁니다.
    print("성공적으로 result_images/result_task3_grabcut.png 파일로 저장되었습니다.") 

if __name__ == '__main__': # 이 문서 자체가 타 스크립트에 라이브러리 식으로 종속된 게 아니라면 내부 동작을 스스로 구동하라는 명령 단서입니다.
    main() # 캡슐처럼 정돈된 main 로직 블록을 최종 기동하며 프로그램을 연주하기 시작합니다.
