import cv2
import numpy as np
if __name__=="__main__":
    img=cv2.imread("test.jpg")



    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    red_mask = cv2.inRange(img, (0,0,50), (10,10,255))
    green_mask = cv2.inRange(img, (0,150,0), (10,255,10))

    kernel = np.ones((7, 7), np.uint8)
    red_mask=cv2.morphologyEx(red_mask,cv2.MORPH_CLOSE,kernel)
    kernel = np.ones((5, 5), np.uint8)
    green_mask=cv2.morphologyEx(green_mask,cv2.MORPH_CLOSE,kernel)

    def get_contours(img):
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_mass_centroid(contours):
        contours=contours.copy()
        centroids=[]
        total_points=0
        for cnt in contours:
            # cnt = contours[0]
            M = cv2.moments(cnt)
            pts=len(cnt)
            total_points+=pts
            cx = int(M['m10'] / M['m00'])*pts
            cy = int(M['m01'] / M['m00'])*pts
            centroids.append([cx,cy])
        return (np.array(centroids).sum(0)/total_points).tolist()

    def get_minmax_centroid(contours):
        contours=contours.copy()
        centroids=[]
        total_points=0
        maxx=maxy=0
        minx=miny=1e5
        for cnt in contours:
            # cnt = contours[0]
            M = cv2.moments(cnt)
            pts=len(cnt)
            total_points+=pts
            np_cnt=np.array(cnt)
            maxxy= np_cnt.max(0).squeeze().tolist()
            minxy= np_cnt.min(0).squeeze().tolist()
            maxx =  maxxy[0] if maxxy[0]>maxx else maxx
            maxy =  maxxy[1] if maxxy[1]>maxy else maxy
            minx =  minxy[0] if minxy[0]<minx else minx
            miny =  minxy[1] if minxy[1]<miny else miny

            # cx = int(M['m10'] / M['m00'])*pts
            # cy = int(M['m01'] / M['m00'])*pts
            # centroids.append([cx,cy])

        res=[int((maxx+minx)/2),int((maxy+miny)/2)]
        # res=int((maxx+minx)/2)
        return res #(np.array(centroids).sum(0)/total_points).tolist()

    contours_green=get_contours(green_mask)
    contours_red=get_contours(red_mask)

    center_green=get_minmax_centroid(contours_green)
    center_red=get_minmax_centroid(contours_red)

    print("len  green contours:", len(contours_green))
    print("len   red  contours:", len(contours_red))

    red_mask=cv2.cvtColor(red_mask,cv2.COLOR_GRAY2BGR)
    green_mask=cv2.cvtColor(green_mask,cv2.COLOR_GRAY2BGR)
    red_mask=cv2.circle(red_mask, tuple(map(lambda x:int(x),center_red)), 3, (0, 0, 255), -1)
    green_mask=cv2.circle(green_mask, tuple(map(lambda x:int(x),center_green)), 3, (0, 0, 255), -1)



    print(center_green,center_red)
    cv2.imshow("test",img)
    cv2.imshow("green_mask",green_mask)
    cv2.imshow("red_mask",red_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()