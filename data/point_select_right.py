import cv2
import glob
import math
import copy
import numpy as np
width = 600
height = 800

# name = "/home/ayoung/Pictures/right_parking_data/right_frame0.jpg"
st = "/home/snuzero/PAE/data/"
pre = "left_"
pt1 = (-1, -1)
pt2 = (-1, -1)
pt3 = (-1, -1)
pt4 = (-1, -1)
# pts = np.array([list(pt1), list(pt2), list(pt3), list(pt4), list(pt1)], dtype = np.float32)
p_num = 1
img = 0

def birdeye(img):
    global width, height
    #x = 320 symmetry
    srcPoint = np.array([[126, 239], [534, 238], [0, 338], [640, 378]], dtype=np.float32)
    #x = 300 symmetry
    dstPoint=np.array([[0,0], [600, 0], [163, 600], [379,600]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    out_img = cv2.warpPerspective(img, matrix, (width, height))
    return out_img

def rotate(pt, cx, cy, cs, sn):
    return (sn*(pt[0]-cx)+cs*(pt[1]-cy)+cx, -cs*(pt[0]-cx)+sn*(pt[1]-cy)+cy)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(img, (x,y), 6, (0,0,0), -1)
        global p_num
        print(p_num, x, y, )
        #cv2.imshow(title, img)
        if p_num == 1:
            global pt1
            pt1 = (x, y)
        if p_num == 2:
            global pt2
            pt2 = (x, y)
        if p_num == 3:
            global pt3
            pt3 = (x, y)
        if p_num == 4:
            global pt4
            pt4 = (x, y)

        global img
        # img = cv2.imread(nm)
        img = birdeye(cv2.imread(nm))
        img = cv2.copyMakeBorder(img, 0, 0, 100, 100, cv2.BORDER_CONSTANT)
        if not pt1 == (-1,-1):
            cv2.circle(img, pt1, 6, (0,0,255), -1)
        if not pt2 == (-1,-1):
            cv2.circle(img, pt2, 6, (0,255,0), -1)
        if not pt3 == (-1,-1):
            cv2.circle(img, pt3, 6, (255,0,0), -1) 
        if not pt4 == (-1,-1):
            cv2.circle(img, pt4, 6, (150,0,150), -1)
        cv2.imshow(title,img)


name = list(glob.glob(st+'backup/*.jpg'))
name.sort()

# for nm in name:
i = 0
while i<len(name):

    nm = name[i]
    print(nm[-9:-4])
    dat = []
    img_ori = cv2.imread(nm)
    title = "N : Append Current Data, ESC : Save, Next Image, P : Previous Image, D : Delete Current Data"
    img_bird = birdeye(img_ori)
    img = cv2.copyMakeBorder(img_bird, 0, 0, 100, 100, cv2.BORDER_CONSTANT)
    img_copy = copy.deepcopy(img)
    img_copy = cv2.resize(img_copy, (640, 640))

    # cv2.polylines(img, [pts], False, (0, 0, 255))
    cv2.imshow(title, img)
    cv2.setMouseCallback(title, onMouse)
    i = i+1

    while True:
        key = cv2.waitKey(0)

        if key == ord('1'):
            p_num = 1
        if key == ord('2'):
            p_num = 2
        if key == ord('3'):
            p_num = 3
        if key == ord('4'):
            p_num = 4

        #maybe should be modified to be used for padding
        if key == ord('N') or key == ord('n'):
            global cen_x, cen_y
            cen_x = (pt1[0]+pt2[0]+pt3[0]+pt4[0])/4
            cen_y = (pt1[1]+pt2[1]+pt3[1]+pt4[1])/4
            mid_x = (pt1[0]+pt4[0])/2
            mid_y = (pt1[1]+pt4[1])/2
            dx = mid_x-cen_x
            dy = mid_y-cen_y
            rt = math.sqrt(dx*dx+dy*dy)
            dx = dx/rt
            dy = dy/rt
            hei = (math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)+math.sqrt((pt3[0]-pt4[0])**2+(pt3[1]-pt4[1])**2))/2
            wid = (math.sqrt((pt1[0]-pt4[0])**2+(pt1[1]-pt4[1])**2)+math.sqrt((pt3[0]-pt2[0])**2+(pt3[1]-pt2[1])**2))/2
            dat.append((cen_x/800, cen_y/800, dx, dy, wid/800, hei/800))
            print(dat)
            cs = dx
            sn = dy
            pt1_pred = (cen_x-wid/2, cen_y+hei/2)
            pt2_pred = (cen_x-wid/2, cen_y-hei/2)
            pt3_pred = (cen_x+wid/2, cen_y-hei/2)
            pt4_pred = (cen_x+wid/2, cen_y+hei/2)
            pt1_pred = rotate(pt1_pred, cen_x, cen_y, cs, sn)
            pt2_pred = rotate(pt2_pred, cen_x, cen_y, cs, sn)
            pt3_pred = rotate(pt3_pred, cen_x, cen_y, cs, sn)
            pt4_pred = rotate(pt4_pred, cen_x, cen_y, cs, sn)
            pts = np.array([list(pt1_pred), list(pt2_pred), list(pt3_pred), list(pt4_pred), list(pt1_pred)], dtype = np.float32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, np.int32([pts]), False, (0, 0, 255))
            cv2.arrowedLine(img, (int(cen_x), int(cen_y)), (int(mid_x), int(mid_y)), (0, 0, 255), 1)
            cv2.imshow(title,img)
            # img = cv2.copyMakeBorder(birdeye(cv2.imread(nm)), 0, 0, 100, 100, cv2.BORDER_CONSTANT)

        if key == ord('d') or key == ord('D'):
            print("Deleted", end = " ")
            dat.pop()
            print(dat)

        if key == ord('p') or key == ord('P'):
            print("Previous image")
            i = i-1
            break

        if cv2.waitKey(0) & 0xFF == 27:
            print(dat)
            fname = open(st+"labels/"+pre+nm[-9:-4]+".txt", "w")
            for indat in dat:
                for ele in indat:
                    fname.write(str(ele)+" ")
                fname.write("\n")
            fname.close()
            cv2.imwrite(st+"images/"+pre+nm[-9:-4]+".jpg", img_copy)
            cv2.destroyAllWindows()
            break
        
    pt1 = (-1, -1)
    pt2 = (-1, -1)
    pt3 = (-1, -1)
    pt4 = (-1, -1)
    p_num = 1