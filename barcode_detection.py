import cv2
import numpy as np
import imutils

def show_wait_destroy(title, img):
    cv2.imshow(title, img)
    cv2.moveWindow(title, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def barcode(filePath):
    # 원본이미지 보기
    img = cv2.imread(filePath,cv2.IMREAD_COLOR)
    show_wait_destroy("img",img)
    
    # grayscale이미지 보기
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #gray scale로 사진 바꿈
    show_wait_destroy("gray",gray)

    #sobel 필터와 averaging Blurring 사용
    gradX = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=-1)  #sobel 필터 사용 x축 3by3   # 1, 0 x축에 관하여 1차미분 하고 y는 그대로
    show_wait_destroy("gray",gradX)

    gradY = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=-1) #sobel 필터 사용 y축 3by3
    show_wait_destroy("gray", gradY)

    gradient = cv2.subtract(gradX, gradY)  # x축이미지에서 y축을 빼면 x축이 강하게 나옴
    show_wait_destroy("gradX",gradient)
    
    gradient = cv2.convertScaleAbs(gradient) #미분한 값에 절대값을 취한다.
    blurred = cv2.blur(gradient, (9, 9)) # 평균 블러링(Averaging Blurring)
    show_wait_destroy("blurred",blurred)

    # 이진화 하기
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY) #이진화를 위해서
    show_wait_destroy("thresh",thresh)

    #closing = deliation 하고 나서 erosion을 해야함
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    show_wait_destroy("closed",closed) 

    # 주변에 있는 노이즈 제거하기 위해서 opening을 사용함
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    show_wait_destroy("closed", closed) 

    # contour 찾기는 검정색 배경에서 흭흰색 물체를 찾는것
    # 따라서 대상은 흰색 배경은 검정색으로 해야함
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    # RETR_EXTERNAL 이미지 가장 바깥쪽의 contour만 추출
    # CHAIN_APPROX_SIMPLE 수평수직 대각선의 방향의 점은 모두 버리고 끝점만
    # 남겨둠 직사격형의 경우 4개의 모서리 점만 남기고 다 버림
    
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    
    # 면적이 가장 큰 순서로 함
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # minAreaRect contour 외접하는 면적이 가장 작은 직사각형을 구함
    # 리턴값으로 x,y 가로 세로 길울어진 각도
    # boxpoint는 좌표값을 float형으로 return함
    # 그값을 int 형으로 바꿔 주면서 

    cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    # 이미지에서
    #-1 img에 실제 그릴 contour 인덱스 파라미터 이 값은 음수이면 contour를 그림
    # 0,255,0 RGB 초록
    # 3 두께
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    File_Path = "./img/barcode.jpg"
    barcode(File_Path)