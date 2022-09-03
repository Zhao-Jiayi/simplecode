from dis import dis
import os
import json
from bisect import bisect_left,bisect_right
import cv2
from tqdm import tqdm
import numpy as np
import copy

LABEL_DICT = {
    'Text':1, 'Title':2, 'Figure':3, 'Figure Caption':4, 'Table':5,
    'Table Caption':6, 'Header':7, 'Footer':8, 'Reference':9,
    'Equation':10, 'Abstract':11, 'Author':12, 'List':13,'Section':14, 
    'Bib Info':15,'Content':16,'Code':17,'Affiliation':18,'PageNum':19}

# LABEL_DICT = {
#     'Text':1, 'Title':2, 'Figure':3, 'Equation':4, 'Table':5,
#     'Caption':6, 'Other':7}
def findALLFiles(path,last_name='.json'):
    files = os.listdir(path)

    res = []
    for f in files:
        if(f[-4:] == "json"):
            res.append(f)
    return res
def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def make_data_json(data_dir, list_file):
    error_file = []
    images = []
    annos = []
    image_id = 1
    idx = 0
    for file_id in list_file:

        # image_id += 100000
        print(os.path.join(data_dir, file_id )),
        anno_dict = json.load(
            open(os.path.join(data_dir, file_id ),'r'))

        images.append({
            "license"       : 0,
            "file_name"     : file_id[:-5] + '.jpg',
            "coco_url"      : "",
            "height"        : anno_dict['imageHeight'],
            "width"         : anno_dict['imageWidth'],
            "date_captured" : "",
            "flickr_url"    : "",
            "id"            : image_id})
        bboxes = []
        anno_id = 0
        c = 0
        for shape_id, shape in enumerate(anno_dict['shapes']):
            
            anno_id = image_id * 100 + shape_id
            
            if shape['label'] not in LABEL_DICT.keys(): 
                continue
            else:
                cate_id = LABEL_DICT[shape['label']]
          
            if shape['shape_type'] == 'rectangle':
                if(len(shape['points'] )< 2):
                    error_file.append(file_id)
                    continue
                    
                polys = [[
                    round(shape['points'][0][0]), round(shape['points'][0][1]),
                    round(shape['points'][1][0]), round(shape['points'][0][1]),
                    round(shape['points'][1][0]), round(shape['points'][1][1]),
                    round(shape['points'][0][0]), round(shape['points'][1][1])]]
            elif shape['shape_type'] == 'polygon':
                polys = [np.round(shape['points']).reshape(-1).tolist()]
                if len(polys[0]) < 6: continue
            else:
                print('fuck', shape['shape_type'])
                continue
            polys = [[int(s) for s in polys[0]]]
            area = PolyArea(np.array(polys[0])[0::2], np.array(polys[0])[1::2])
            
            left  = np.min(polys[0][0::2])
            right = np.max(polys[0][0::2])
            upper = np.min(polys[0][1::2])
            lower = np.max(polys[0][1::2])
            bbox = [int(left), int(upper), int(right - left), int(lower - upper)]
            bboxes.append([int(left), int(upper), int(right), int(lower)])
            annos.append({
                "id"           : anno_id,
                "image_id"     : image_id,
                "category_id"  : cate_id,
                "segmentation" : polys,
                "area"         : area,
                # "bbox"         : bbox,
                "iscrowd"      : 0})
            idx+=1
            c+=1
        image_id += 1
        img = cv2.imread('/ssd8/other/zhaojiayi/mydata712/val2017/'+file_id[:-5] + '.jpg')
        
    
        bboxes = adjustBound(img,bboxes)
    
        for b in bboxes:
            b[3] = b[3] - b[1]
            b[2] = b[2] - b[0]
            annos[idx-c]["bbox"] = b
            c-=1
    info_dict = {}
    info_dict['info'] = {
        "description"   : "Question Segmentation 7-class - train6k",
        "url"           : "",
        "version"       : "train6k",
        "year"          : "2022",
        "contributor"   : "zhao jia yi",
        "date_created"  : "2022/6/9"}
    info_dict['licenses'] = [{
            "url": "",
            "id" : 0,
            "name": "Do not distribute under any fucking status !!!"}]
    info_dict['categories'] = [{"supercategory": "none", "id": 1, "name": "Text"}, {"supercategory": "none", "id": 2, "name": "Title"}, {"supercategory": "none", "id": 3, "name": "Figure"}, {"supercategory": "none", "id": 4, "name": "Figure Caption"}, {"supercategory": "none", "id": 5, "name": "Table"}, {"supercategory": "none", "id": 6, "name": "Table Caption"}, {"supercategory": "none", "id": 7, "name": "Header"}, {"supercategory": "none", "id": 8, "name": "Footer"}, {"supercategory": "none", "id": 9, "name": "Reference"}, {"supercategory": "none", "id": 10, "name": "Equation"}, {"supercategory": "none", "id": 11, "name": "Abstract"}, {"supercategory": "none", "id": 12, "name": "Author"}, {"supercategory": "none", "id": 13, "name": "List"}, {"supercategory": "none", "id": 14, "name": "Section"}, {"supercategory": "none", "id": 15, "name": "Bib Info"}, {"supercategory": "none", "id": 16, "name": "Content"}, {"supercategory": "none", "id": 17, "name": "Code"}, {"supercategory": "none", "id": 18, "name": "Affiliation"}, {"supercategory": "none", "id": 19, "name": "PageNum"}]

    # [
    #     {"supercategory": "none", "id": 1, "name": "Text"},
    #     {"supercategory": "none", "id": 2, "name": 'Title'},
    #     {"supercategory": "none", "id": 3, "name": 'Figure'},
    #     {"supercategory": "none", "id": 4, "name": 'Equation'},
    #     {"supercategory": "none", "id": 5, "name": 'Table'},
    #     {"supercategory": "none", "id": 6, "name": 'Caption'},
    #     {"supercategory": "none", "id": 7, "name": 'Other'},


    #     ]


    info_dict['images'] = images
    info_dict['annotations'] = annos
    json.dump(info_dict, open(output_path, 'w'))


# 水平方向投影
def hProject(binary):
 


    # 创建h长度都为0的数组
       # 创建 w 长度都为0的数组

    h_h = np.sum(binary,axis=1) 
    return h_h

# 垂直反向投影
def wProject(binary):
    

    # 创建 w 长度都为0的数组
 


    
    w_w = np.sum(binary,axis=0) 

    
    # for i in range(0,w_e-w_s):
    #     for j in range(0,h_e-h_s):
    #         if binary[j+h_s, i+w_s ] == 0:
    #             w_w[i] += 1
    return w_w
def findBound(nums,start_idx,cur_idx,is_x,direction,max_x_shift,max_y_shift):
    while(cur_idx>=0 and cur_idx <= len(nums)-1):
        # if(is_x):
        #     if(abs(nums[start_idx][1]-nums[cur_idx][1])<=max_x_shift and \
        #     abs(nums[start_idx][2]-nums[cur_idx][2])<=max_x_shift
        #     ):
        #         return cur_idx
        # else:
        #     if(abs(nums[start_idx][1]-nums[cur_idx][1])<=max_y_shift and \
        #     abs(nums[start_idx][2]-nums[cur_idx][2])<=max_y_shift
        #     ):
        #         return cur_idx
        # cur_idx+=direction
        s = max(nums[start_idx][1],nums[cur_idx][1])
        e = min(nums[start_idx][2],nums[cur_idx][2])
        if(s < e and (e-s) > (nums[start_idx][2]-nums[start_idx][1])*0.05):
            # print(nums[start_idx][0],nums[start_idx][1],nums[start_idx][2])
            # print(nums[cur_idx][0],nums[cur_idx][1],nums[cur_idx][2])
            return cur_idx 
        cur_idx+=direction
    return -1
def rgb2Binary(img,threshold = 232,value=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, threshold,value, cv2.THRESH_BINARY)
    return th
def binSearch(nums,tar):
    l = 0
    r = len(nums)-1
    while(l <= r):
      
        mid = (l+r)>>1
        if(nums[mid][0] > tar):
            r = mid - 1
        elif(nums[mid][0]<tar):
            l = mid + 1
        else:
            return mid
    return mid

def adjustBound(img_file,bboxes,max_x_shift=7,max_y_shift = 10,max_dis = 0):
    binary_img = rgb2Binary(img_file)
    h,w = binary_img.shape
    binary_img = np.where(binary_img == 0,1,0)
    new_bboxes = []
    x_sort = []
    y_sort = []
    for b in bboxes:
        x_sort.append([b[0],b[1],b[3]])
        x_sort.append([b[2],b[1],b[3]])
        y_sort.append([b[1],b[0],b[2]])
        y_sort.append([b[3],b[0],b[2]])
    x_sort.sort(key=(lambda x:x[0]))
    y_sort.sort(key=(lambda x:x[0]))

    # print(x_sort,y_sort)
    # print("start",x1_sort)
    # print(x2_sort)
    # print(y1_sort)
    # print(y2_sort,"end")
    for b in bboxes:
        
        new_b = copy.deepcopy(b)
        
        w_s = b[0]
        w_e = b[2]
        hs = max(int(b[1]-max_x_shift),0)
        he = min(int(b[3]+max_x_shift),h-1)
        ws = max(0,int(b[0]-max_y_shift))
        we = min(w-1,int(b[2]+max_y_shift))
        is_black = binary_img[hs:he,int(b[0]):int(b[2])]

        h_h = hProject(is_black)#>0黑色

      
        h_h = np.where(h_h>0,1,0)
        #上边界范围
 
        p = findp(h_h,int(b[1])-hs,-1,1)
        new_b[1] = hs+p
   
        p = findp(h_h,int(b[3])-hs,1,1)
        new_b[3] = hs+p

       
        is_black = binary_img[int(new_b[1]):int(new_b[3]),ws:we]
        w_w = wProject(is_black)#>0黑色
        w_w = np.where(w_w>0,1,0)
  
        p = findp(w_w,int(b[0]-ws),-1,0)
        new_b[0] = p+ws
     
        p = findp(w_w,int(b[2]-ws),1,0)
        new_b[2] = p+ws
        l = binSearch(y_sort,b[1])
        l = findBound(y_sort,l,l-1,1,-1,max_y_shift,max_x_shift)
        if(l != -1):
            if(new_b[1] < int(y_sort[l][0])):
           
                new_b[1] = int(y_sort[l][0])
                new_b[1] += max_dis
            else:
                new_b[1] -= max_dis
        else:
            new_b[1] -= max_dis
        #求左边界位置
        
        l = binSearch(x_sort,b[0])
     
        l = findBound(x_sort,l,l-1,0,-1,max_x_shift,max_y_shift)
  
        if(l != -1):
            if(new_b[0] < int(x_sort[l][0])):
                new_b[0] = int(x_sort[l][0])
            else:
                new_b[0] -= max_dis
        else:
            new_b[0] -= max_dis
       #求下边界位置

        
        
        r = binSearch(y_sort,b[3])
     
        r = findBound(y_sort,r,r+1,1,1,max_x_shift,max_y_shift)
      
        if(r != -1):
            if(new_b[3] > int(y_sort[r][0])):
                
                new_b[3] = int(y_sort[r][0])
            else:
                new_b[3] += max_dis
        else:
            new_b[3] += max_dis
        #求右边界位置
       
        r = binSearch(x_sort,b[2])
        r = findBound(x_sort,r,r+1,0,1,max_x_shift,max_y_shift)
        if(r != -1):
            if(new_b[2] > int(x_sort[r][0])):
                new_b[2] = int(x_sort[r][0])
            else:
                new_b[2] += max_dis
        else:
            new_b[2] += max_dis
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
if __name__ == '__main__':

    data_dir = '/ssd8/other/zhaojiayi/mydata712/val2017'
    output_path = '/ssd8/other/zhaojiayi/mydata712/annotations/instances_val_2017.json'
    path = '/ssd8/other/zhaojiayi/mydata712//val2017'
    json_file = findALLFiles(path)
    print(len(json_file))
    make_data_json(data_dir, json_file)