def region_growing_API(patientID, mouse_position):
    
    import cv2
    import numpy as np
    
    class Point(object):
        def __init__(self,x,y):
            self.x = x
            self.y = y
        def getX(self):
            return self.x
        def getY(self):
            return self.y
    
    #計算像素之間的偏差
    def getGrayDiff(img,currentPoint,tmpPoint):
        return abs(int(img[currentPoint.y,currentPoint.x]) - int(img[tmpPoint.y,tmpPoint.x]))

    #計算與初始種子點的偏差值
    def getPixelDiff(img,pixel,tmpPoint):
        return abs(int(pixel) - int(img[tmpPoint.y,tmpPoint.x]))
    
    #選擇8鄰域   
    #def selectConnects(p):
    def selectConnects():
        #if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                        Point(0, 1), Point(-1, 1), Point(-1, 0)]#八邻域
        #else:
            #connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]#四邻域
        return connects
    
    def regionGrow(img,seeds,thresh, pixel, pixel_thresh, p = 1):
    #讀取圖片寬高，建立和原圖一樣大小的seedMark
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        print('thresh=',thresh)
    #將種子點放入種子列表seedList
        seedList = []
    
        for seed in seeds:
            seedList.append(seed)
        label = 1
    
        #connects = selectConnects(p)
        connects = selectConnects()
    #逐個點開始生長，生長結束條件為seedlist為空，就是沒有生長點
        while(len(seedList)>0):
        #弹出種子點列表的第一个點作為生長點
            currentPoint = seedList.pop(0)#弹出第一個元素
        #將生長點對應的seedMark點赋值label（1），即為白色
            seedMark[currentPoint.y,currentPoint.x] = label
        #以種子點為中心，八鄰域的像素進行比較
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
            #判斷是否為圖像外的點，若是則跳過。  如果種子點是圖像的邊界點，鄰域點就會落在圖像外
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
            #判斷鄰域點和種子點的差值以及不能跟最初始生長點像素值相差過大
                grayDiff  = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
                pixelDiff = getPixelDiff(img,pixel,Point(tmpX,tmpY))
            #如果鄰域點和種子點的差值小于閥值并且是没有被分類的點，則認為是和種子點同類，賦值label，
            #並作為下一个種子點放入seedList
                if grayDiff < thresh and seedMark[tmpY,tmpX] == 0 and pixelDiff <= pixel_thresh:
                    seedMark[tmpY,tmpX] = label
                    seedList.append(Point(tmpX,tmpY))
    
        return seedMark
    
    boxes = []
    boxes.append(mouse_position)
    #讀入圖片的灰階圖 
    imgpath ='C:/Users/leowang/Desktop/dicom_server/00A00101_3_24.png'
    img = cv2.imread(imgpath,0)
    
    gray_three_channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    median = cv2.medianBlur(cl1,5)
    
    for num in boxes:
        pixel = median[num[1],num[0]]
        seeds = [Point(num[0],num[1])]
#================================================#
#================================================#
    #res = np.hstack((img, cl1))#將陣列水平串接起來，方便比較圖片
    #開始從種子點生長
        binaryImg = regionGrow(median,seeds,10, pixel, 10)
        binaryImg1 = np.array(binaryImg,dtype=np.uint8)*255
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(binaryImg1,kernel,iterations = 2)#膨脹 看之前需不需要做開運算?

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        #cv2.drawContours(gray_three_channel, [cnt], 0, (0, 255, 0), 1)
    
    return cnt