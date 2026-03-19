import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os 

def main():
    # 처리 대상 이미지의 상대적인 폴더 경로를 특정 문자열 변수에 바인딩해둡니다.
    img_path = 'images/dabo.jpg' 
    
    # 스크립트 작동 전, 타겟 파일의 실제 위치 도달 가능성을 os.path 모듈로 먼저 점검합니다.
    if not os.path.exists(img_path): 
        print(f"Error: {img_path} not found.") # 파일 고장에 대비해 오류 경고 문구를 터미널상에 표출합니다.
        return # 더 이상의 후속 진행을 전면 백지화하고 함수 작동을 마칩니다.

    # 1. 뼈대가 되는 원본 이미지를 디스크에서 읽어 img 변수에 BGR 3채널의 색상 배열 구조로 가져옵니다.
    img = cv.imread(img_path) 
    
    # 2. 에지 검출 연산 공식을 적용하기 위해 컬러 스페이스를 밝기 성분만 가진 단일 채널(Grayscale)로 강제 병합합니다.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    
    # 3. 내부적으로 부드러움(Gaussian) 보정이 포함된 캐니(Canny) 엣지 알고리즘으로 윤곽선을 극도로 얇고 날카롭게 추출해냅니다.
    # threshold1(100)은 잔가지를 무시할 약한 엣지 하한선이며, threshold2(200)는 확실한 본체라고 판단할 강한 엣지 상한선의 성격을 지닙니다.
    edges = cv.Canny(gray, threshold1=100, threshold2=200) 
    
    # 4. 통계적 투표 방식인 확률적 허프 변환(HoughLinesP)에 투입될 민감한 하이퍼 파라미터들을 코드 상단에 변수화하여 외부에서 편하게 조작토록 뺍니다.
    rho = 1                         # Hough 변환 매개 공간에서 원점 기준 탐색할 수직 선 거리의 분해능을 의미합니다 (1픽셀 단위).
    theta = np.pi / 180             # Hough 공간에서 선을 탐색할 각도의 촘촘한 분해능을 삼각함수 라디안으로 정의합니다 (1도 단위).
    hough_threshold = 50            # 동일 직선 위에 존재한다고 '투표(교차)'한 점의 개수가 50개를 넘어설 때만 직선으로 추려냅니다.
    minLineLength = 50              # 아무리 강한 직선이라도 이 픽셀 길이 이하라면 노이즈(먼지)로 간주해 폐기합니다.
    maxLineGap = 10                 # 도중에 끊어짐이 있는 파선이라도 그 간극이 10픽셀 이내라면 연장된 하나의 직선으로 편입시킵니다.
    
    # 앞서 도출한 에지 영상(edges)을 투입하고, 최적화 튜닝한 5개의 파라미터를 사용해 유의미한 선분들의 시작/끝 좌표 배열집(lines)을 얻어냅니다.
    lines = cv.HoughLinesP(edges, rho, theta, hough_threshold, 
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
                           
    # 5. 기존 원본 컬러 프레임을 보존하면서 새로 선을 그리기 위해, 완전히 분리된 클론(복사) 배열을 하나 더 메모리에 할당해줍니다.
    result_img = img.copy() 
    
    # 만약 허프 변환 조건이 까다로워 검출된 라인이 전혀 없을 경우를 대비해 None 방어 로직을 씌워줍니다.
    if lines is not None: 
        # 발견된 복수 개의 튜플 형태 라인 좌표 그룹들을 배열에서 하나씩 순차적으로 뽑아냅니다.
        for line in lines: 
            # 각 라인은 3차원 내포 구조이므로 [0]번 인덱스에 있는 시작점(x1, y1)과 끝점(x2, y2) 데이터를 풀어헤칩니다(언패킹).
            x1, y1, x2, y2 = line[0] 
            # 할당해둔 복사본 그림 위에 시작점과 끝점을 잇는 빨간색(BGR기준 0,0,255 형상) 굵기 2픽셀의 시각적인 선을 영구적으로 그려 넣습니다.
            cv.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2) 
            
    # 6. 화면 출력을 매끄럽게 전개하도록 OpenCV 특유의 파란색 쏠림 포맷(BGR)을 표준 스크린 픽셀(RGB)로 색 채널만 단순히 역전환해둡니다.
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    # 그려진 빨간 선과 스틸컷 모두 RGB 포맷으로 나란히 맞춰 변색 오류를 봉쇄합니다.
    result_img_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB) 
    
    # 7. 원본 이미지와 추론 이미지를 나란히 도해하기 위한 전체 그림판 규격(가로 10픽셀, 세로 5픽셀) 생성 구문입니다.
    plt.figure(figsize=(10, 5)) 
    
    # 도화지를 가로로 2등분 분할 영역으로 지정한 뒤 그중 첫 번째(왼쪽) 구획의 권한을 부여받습니다.
    plt.subplot(1, 2, 1) 
    # 왼편의 영역 위로 RGB 보정된 아무것도 그리지 않은 순수 원본 이미지를 투과합니다.
    plt.imshow(img_rgb) 
    # 사진 위에다 보기 편하게 'Original Image'라는 영단어 네임택을 기입합니다.
    plt.title('Original Image') 
    # 숫자로 되어 있는 불규칙한 xy 잣대와 네모 여백선을 화면 시야 밖으로 차단(off)시킵니다.
    plt.axis('off') 
    
    # 가로 2등분 영역 중 두 번째(오른쪽) 남음 구획으로 활성화 중심을 갈아탑니다.
    plt.subplot(1, 2, 2) 
    # 빨간 직선들이 강렬하게 삽입된 결과본 이미지를 해당 구획 안에 랜더링합니다.
    plt.imshow(result_img_rgb) 
    # 영문 'Hough Lines Detection'이라는 안내성 제목표식을 추가적으로 명시합니다.
    plt.title('Hough Lines Detection') 
    # 우측 또한 미관을 방해하는 잔가지 그래프 축선들을 모두 숨겨버립니다.
    plt.axis('off') 
    
    # matplotlib 모듈의 타이트 메커니즘을 작동시켜 2개의 서브 이미지들이 보기 좋게 좌우 간극을 조율하게 합니다.
    plt.tight_layout() 
    # 화면으로만 띄우지 않고 디스크 보관 용도로 result_images 디렉토리에 사진을 즉각 다운로드 지시합니다.
    plt.savefig('result_images/result_task2_hough.png') 
    # 다운로드가 온전하게 완료됐음을 백그라운드 환경 작업자에게 메시지로 직관적 체감이 오게 콘솔 통보합니다.
    print("성공적으로 result_images/result_task2_hough.png 파일로 저장되었습니다.") 

if __name__ == '__main__': # 타 파이썬 문서에서 본 모듈을 쓸 목적으로 부를 때는 동작하지 말라는 if 방어막 관례입니다.
    main() # 셸이나 프롬프트에서 직접 명령어를 통해 동작할 땐 정의된 프로세스를 일제히 시작합니다.
