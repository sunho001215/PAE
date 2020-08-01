import cv2
import glob

st = "/home/snuzero/PAE/data/"

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

name = list(glob.glob(st+'images/*.jpg'))
name.sort()

for nm in name:
    img = cv2.imread(nm)
    img_bri = increase_brightness(img)
    subpre = "_br"
    cv2.imwrite(nm[:-4]+subpre+".jpg", img_bri)
    fname = open(nm[:len(st)]+"labels/"+nm[len(st)+len("images/"):-4]+".txt", "r")
    fname_bri = open(nm[:len(st)]+"labels/"+nm[len(st)+len("images/"):-4]+subpre+".txt", "w")
    while True:
        line = fname.readline()
        if not line:
            break
        lst = line.split()
        fname_bri.write(lst[0]+" "+lst[1]+" "+lst[2]+" "+lst[3]+" "+lst[4]+" "+lst[5]+"\n")
    fname_bri.close()