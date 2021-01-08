import cv2
import numpy as np
if __name__=="__main__":
    img=cv2.imread("test.jpg")
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)



    # hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # light_orange = (1, 190, 200)
    # dark_orange = (18, 255, 255)
    # red_mask = cv2.inRange(hsv_img, (0,200,0), (10,255,255))
    # green_mask = cv2.inRange(hsv_img, (50,200,0), (100,255,255))

    red_mask = cv2.inRange(img, (0,0,50), (10,10,255))
    green_mask = cv2.inRange(img, (0,150,0), (10,255,10))

    kernel = np.ones((6, 6), np.uint8)
    red_mask=cv2.morphologyEx(red_mask,cv2.MORPH_OPEN,kernel)

    kernel = np.ones((2, 2), np.uint8)
    green_mask=cv2.morphologyEx(green_mask,cv2.MORPH_OPEN,kernel)


    # green_mask = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(green_mask, 127, 255, 0)

    ret, thresh = cv2.threshold(green_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour=cv2.drawContours(np.zeros_like(red_mask), contours, -1, (0, 255, 0), 3)

    # thresh = cv2.threshold(red_mask, 60, 255, cv2.THRESH_BINARY)[1]
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)

    cv2.imshow("test",img)
    cv2.imshow("green_mask",green_mask)
    cv2.imshow("red_mask",red_mask)
    cv2.imshow("contour",red_mask)


    cv2.waitKey(0)
    cv2.destroyAllWindows()