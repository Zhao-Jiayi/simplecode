from dis import dis
import os
import json
from bisect import bisect_left,bisect_right
import cv2
from tqdm import tqdm
import numpy as np
import copy
def hProject(binary,h_s,h_e,w_s,w_e):
 


    # 创建h长度都为0的数组
       # 创建 w 长度都为0的数组

    h_h = np.sum(binary,axis=1) 
    return h_h

# 垂直反向投影
def wProject(binary,h_s,h_e,w_s,w_e):
    

    # 创建 w 长度都为0的数组
 


    
    w_w = np.sum(binary,axis=0) 

    
    # for i in range(0,w_e-w_s):
    #     for j in range(0,h_e-h_s):
    #         if binary[j+h_s, i+w_s ] == 0:
    #             w_w[i] += 1
    return w_w

def rgb2Binary(img,threshold = 230,value=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, threshold,value, cv2.THRESH_BINARY)
    return th
def adjustBound(img_file,bboxes,max_x_shift=5,max_y_shift = 10,max_dis = 0):
    binary_img = rgb2Binary(img_file)
    h,w = binary_img.shape
    new_bboxes = []
    for b in bboxes:
        x = []
        y = []
        for bb in bboxes:
            if((abs(bb[0]-b[0])<=max_y_shift or abs(bb[2]-b[2])<=max_y_shift)
):
                x.append(bb[1])
                x.append(bb[3])
            if(abs(bb[1]-b[1])<=max_x_shift or abs(bb[3]-b[3]) <= max_x_shift):
                y.append(bb[0])
                y.append(bb[2])
        x.sort()
        y.sort()
        new_b = copy.deepcopy(b)
        h_s = max(0,int(b[1])-max_x_shift-1)
        h_e = min(int(b[3])+max_x_shift-1,h-1)
        w_s = max(0,int(b[0])-max_y_shift-1)
        w_e = min(int(b[2])+max_y_shift-1,w-1)
        is_black = copy.deepcopy(binary_img[h_s:h_e+1,w_s:w_e+1])
     
        is_black = np.where(is_black == 0,1,0)
        h_h = hProject(is_black,h_s,h_e,w_s,w_e)#>0黑色
        w_w = wProject(is_black,h_s,h_e,w_s,w_e)#>0黑色

        h_h = np.where(h_h>0,1,0)
        w_w = np.where(w_w>0,1,0)
        print(b)
        print(h_h)
        print(w_w)
        # for i in range(0,h_e-h_s):
        #     if(h_h[i] == 0):
        #         h_white.append(i+h_s)
        #     else:
        #         h_black.append(i+h_s)
        # for i in range(0,w_e-w_s):
        #     if(w_w[i] == 0):
        #         w_white.append(i+w_e)
        #     else:
        #         w_black.append(i+w_e)
        # #求上边界位置

        
        
        p = findp(h_h,int(b[1]-h_s),-1,1)
        new_b[1] = p+h_s
        if(x[0] != b[1]):

            l = bisect_left(x,b[1])
     
            if(new_b[1] < int(x[l-1])):
                new_b[1] = int(x[l-1])
        

                new_b[1] += max_dis
            else:
                new_b[1] -= max_dis
        else:
            new_b[1] -= max_dis
            # if(p > 0):
            #     new_b[1] = h_white[p-1]-max_dis
        # else:
        #     p = findp(h_black,b[1])
        #     if(p < len(h_black) and p >= 0):
        #         print(p)
        #         new_b[1] = h_black[p]-max_dis
        #求左边界位置
        
        p = findp(w_w,int(b[0]-w_s),-1,0)
        new_b[0] = p+w_s
        if(b[0] != y[0]):
            l = bisect_left(y,b[0])
            if(new_b[0] < int(y[l-1])):
                new_b[0] = int(y[l-1])


                # new_b[0] += max_dis
        #     else:
        #         new_b[0] -= max_dis
        # else:
        #     new_b[0] -= max_dis
            # if(p > 0):
            #     print(p)
            #     new_b[0] = w_white[p-1]-max_dis
        # else:
        #     p = findp(w_black,b[0])
        #     if(p < len(w_black) and p >= 0):
        #         print(p)
        #         new_b[0] = w_black[p]-max_dis         
        #求下边界位置
        p = findp(h_h,int(b[3]-h_s),1,1)
        new_b[3] = p+h_s
        if(x[-1] != b[3]):
      
            r = bisect_right(x,b[3])

            if(new_b[3] > int(x[r])):
                new_b[3] = int(x[r])



        #     else:
        #         new_b[3] += max_dis
        # else:

        #     new_b[3] += max_dis
            # if(p < len(h_white) and p >=0):
            #     print(p)
            #     new_b[3] = h_white[p]+max_dis
        # else:
        #     p = findp(h_black,b[3])
        #     if(p > 0):
        #         print(p)
        #         new_b[3] = h_black[p-1]+max_dis
        #求右边界位置
        p = findp(w_w,int(b[2]-w_s),1,0)
        new_b[2] = p+w_s
        if(b[2] != y[-1]):
  
            r = bisect_right(y,b[2])
      
            if(new_b[2] > int(y[r])):
                new_b[2] = int(y[r])
   

        #         new_b[2] -= max_dis
        #     else:
        #         new_b[2] += max_dis
        # else:
        #     new_b[2] += max_dis
            # if(p >= 0):
            #     print(p)
            #     new_b[2] = w_white[p]+max_dis
        # else:
        #     p = findp(w_black,b[2])
        #     if(p > 0):
        #         print(p)
        #         new_b[2] = w_black[p-1]+max_dis

        new_bboxes.append(new_b)
    return new_bboxes
def findp(nums,start,add,is_x,shift_x=2,shift_y = 3):
    init_shift = 0
    if(is_x):
        init_shift = shift_x
    else:
        init_shift = shift_y
    shift_thresh = init_shift
    while(start > 0 and start < len(nums)):
        if(nums[start] == 0):
            shift_thresh-=1
            if(shift_thresh == 0):
                break
        else:
            shift_thresh= init_shift
        start += add
    shift_thresh = init_shift
    while(start> 0 and start < len(nums)):
        if(nums[start] == 1):
            shift_thresh-=1
            if(shift_thresh == 0):
                return start + add * init_shift
        else:
            shift_thresh= init_shift
        start -= add
    return start
file = "/ssd8/other/zhaojiayi/c++/test/ff950683ed0529e554755895ba50b386_020.txt"
img = cv2.imread(file[:-3]+'jpg')
bboxs = []
with open(file) as f:
    n = f.readlines(1)
    n = int(n[0])
    print(n)
    for i in range(2,n+2):
        s = f.readlines(i)[0][:-2]
        print(s)
        bboxs.append([int(num) for num in s.split(' ')])
adjustBound(img,bboxs)