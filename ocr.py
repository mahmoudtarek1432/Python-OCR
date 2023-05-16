#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tkinter import *
import cv2
from tkinter import filedialog
from rembg import remove
from PIL import Image
from easyocr import easyocr
import numpy as np
import pytesseract

wi=Tk()
wi.title("choose image")
file=filedialog.askopenfilename()
path=open(file, 'r')
name=path.name
print(name)

input_path = name
output_path = 'moraaa.png'
#sharp_path = 'sharp.png'
input = Image.open(input_path)
output = remove(input)
print("sda")
output.save(output_path)


img = cv2.imread(output_path)
#resized=cv2.resize(img,(350,450))
blurred = cv2.blur(img, (5,5))
kernel = np.array([[-1,-1,-1] ,[-1,9,-1],[-1,-1,-1]])
sharpened = cv2.filter2D(blurred, -1, kernel)
canny = cv2.Canny(sharpened, 50, 200)
pts = np.argwhere(canny>0)
y1,x1 = pts.min(axis=0)
y2,x2 = pts.max(axis=0)
cropped = img[y1:y2, x1:x2]
w,h,c=cropped.shape
o=int(w/2)
i=int(h/2.5)
n=int(h/6)
cr=cropped[n-10:i+7,o:]
cropped_img=cropped[i+10:,o+7:]
cv2.imwrite("newimg.png",cropped_img)
cv2.imshow("unsharp",cr)
cv2.waitKey(0)
cv2.destroyAllWindows()
text=pytesseract.image_to_string(cr,lang='ara',config='--psm 11 --oem 3')
splited=text.split('\n')

state=0
data=[]
if len(text.split('\n'))==8:
    state=1
    print(state)
    firstname=splited[0]
    secondname=splited[2]
    adress=splited[4]+" "+splited[6]
    
    data=[firstname,secondname,adress]
    for i in data:
        if i==None:
            print(data)
            break
        else:
            #print("good img")
            imgs = cv2.imread('newimg.png',0)
            gauss = cv2.GaussianBlur(imgs, (7,7), 0)
            unsharp_image = cv2.addWeighted(imgs, 2, gauss, -1, 0)
            cv2.imshow("unsharp",unsharp_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()    
            #####
            s=easyocr.Reader(['ar','ar'])
            o=s.readtext(unsharp_image, detail = 0,text_threshold = 0.2
            ,width_ths = 0.8,low_text= .17)
            m=0
            #size=len(o[0])
            #print(size)
            #m=0
            #for i in range(len(o[-1])):
            #    m=m+1
            #if m!=20:
            #    print("error!please reenter img")
            #else:
            data.append(o)
            #print(o)
            print(data)
            #print(o)
            break
else:
    state=2
    imgs = cv2.imread('newimg.png',0)
    gauss = cv2.GaussianBlur(imgs, (7,7), 0)
    unsharp_image = cv2.addWeighted(imgs, 2, gauss, -1, 0)
    s=easyocr.Reader(['ar','ar'])
    d=s.readtext(cr, detail = 0,text_threshold = 0.18
    ,width_ths = 0.9,low_text= 0.17)
    o=s.readtext(unsharp_image, detail = 0,text_threshold = 0.18
    ,width_ths = 0.9,low_text= 0.17)
    print(state)
    if o==None or d==None:
        print(data)
    else:
        data.append(o[0])
        data.append(d)
        print(data)
#elif state==4:
#    print("reenter img")



