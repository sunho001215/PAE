import cv2
import glob

st = "/home/snuzero/PAE/data/"
name = list(glob.glob(st+'labels/*.txt'))
name.sort()
lstname = []

for nm in name:
    fname = open(nm, "r")
    line = fname.readlines()
    fname.close()
    fname_new = open(nm, "w")
    for i in range(len(line)):
        lst = line[i].split()
        if float(lst[2])**2+float(lst[3])**2 < 0.5:
            lstname.append(fname)
            fname_new.write(lst[0]+" "+lst[1]+" "+str(float(lst[2])*800)+" "+str(float(lst[3])*800)+" "+lst[4]+" "+lst[5]+"\n")
        else:
            fname_new.write(lst[0]+" "+lst[1]+" "+lst[2]+" "+lst[3]+" "+lst[4]+" "+lst[5]+"\n")
    fname_new.close()
            

for strr in lstname:
    print(strr)

