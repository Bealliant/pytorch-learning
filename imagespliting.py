# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:30:42 2021

@author: Bealliant
"""

import os,sys
import cv2 as cv
import numpy as np

images = []
dir = r"D:/Bealliant"   #open file directory
sdir = r"D:/Bealliant\samples" #save directory
def find(dir):
    
    files = os.listdir(dir)
    for file in files:
        a,b = os.path.splitext(file)
        if b == ".jpg" or b == ".png":
            images.append(dir+'\\'+file)
        elif b == "":
            find(dir +"\\"+ a)
        #这个地方通过递归实现了在文件夹嵌套的情况下，搜索一个母文件下的所有图像文件,返回一个由图像地址组成的list
    return images


def imagesplitting(dir,sdir):
    file_name = os.path.basename(dir)
    file_name = os.path.splitext(file_name)[0]
    
    os.makedirs(sdir+'\\'+file_name+"\\"+"512pix")
    os.makedirs(sdir+"\\"+file_name+"\\"+'256pix')
    img = RGB_Splitting(dir)
    try:
        size = img.shape[0:2]
        m = size[0]//512
        n = size[1]//512

        a = 0
        for i in range(m):
            for j in range(n):
                temp_img = img[i*512:(i+1)*512,j*512:(j+1)*512] 
                name = "{image}_{part}".format(image = file_name, part = a)
                name += ".jpg"
                cv.imwrite(sdir+"/"+file_name+"/512pix/"+name, temp_img)
                print("{imgname} saves successfully".format(imgname = name))
                b = 0
                for k in range(2):
                    for l in range(2):
                        sub_img = temp_img[k*256:(k+1)*256,l*256:(l+1)*256]
                        sub_img_name = sdir+"/"+file_name+"/256pix/"+"{img}_{part}_{sub}.jpg".format(img = file_name, part = a, sub = b)
                        cv.imwrite(sub_img_name,sub_img)
                        b += 1
                a+=1
    except AttributeError:
    #在调试的时候，如果文件夹中的某个图片文件名含中文，貌似cv无法打开。会报错AttributeError: 'NoneType' object has no attribute 'shape'
        print("cv2 cannot open the image obj{}".format(file_name))

def RGB_Splitting(dir):
    img = cv.imread(dir)
    b,g,r = cv.split(img)
    retval, r = cv.threshold(r, 119, 255, cv.THRESH_BINARY)
    return r 
    
imagesplitting(r"D:\Bealliant\DJI_0406.png", sdir)
