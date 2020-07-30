import cv2
import glob

st = "/home/snuzero/PAE/data/"
pre = "right_"

def xflip(img):
    subpre = "x_"
    img_xflip = cv2.flip(img, 1)
    cv2.imwrite(st+"images/"+pre+subpre+nm[-9:-4]+".jpg", img_xflip)
    fname = open(st+"labels/"+pre+nm[-9:-4]+".txt", "r")
    fname_x = open(st+"labels/"+pre+subpre+nm[-9:-4]+".txt", "w")
    while True:
        line = fname.readline()
        if not line:
            break
        lst = line.split()
        fname_x.write(str((1-float(lst[0])))+" "+lst[1]+" "+str(float(lst[2])*(-1))+" "+lst[3]+" "+lst[4]+" "+lst[5]+"\n")
    fname_x.close()
    
def yflip(img):
    subpre = "y_"
    img_yflip = cv2.flip(img, 0)
    cv2.imwrite(st+"images/"+pre+subpre+nm[-9:-4]+".jpg", img_yflip)
    fname = open(st+"labels/"+pre+nm[-9:-4]+".txt", "r")
    fname_y = open(st+"labels/"+pre+subpre+nm[-9:-4]+".txt", "w")
    while True:
        line = fname.readline()
        if not line:
            break
        lst = line.split()
        fname_y.write(lst[0]+" "+str((1-float(lst[1])))+" "+lst[2]+" "+str(float(lst[3])*(-1))+" "+lst[4]+" "+lst[5]+"\n")
    fname_y.close()

def xyflip(img):
    subpre = "xy_"
    img_xyflip = cv2.flip(cv2.flip(img, 0), 1)
    cv2.imwrite(st+"images/"+pre+subpre+nm[-9:-4]+".jpg", img_xyflip)
    fname = open(st+"labels/"+pre+nm[-9:-4]+".txt", "r")
    fname_xy = open(st+"labels/"+pre+subpre+nm[-9:-4]+".txt", "w")
    while True:
        line = fname.readline()
        if not line:
            break
        lst = line.split()
        fname_xy.write(str((1-float(lst[0])))+" "+str((1-float(lst[1])))+" "+str(float(lst[2])*(-1))+" "+str(float(lst[3])*(-1))+" "+lst[4]+" "+lst[5]+"\n")
    fname_xy.close()



name = list(glob.glob(st+'images/*.jpg'))
name.sort()

for nm in name:
    img = cv2.imread(nm)
    xflip(img)
    yflip(img)
    xyflip(img)