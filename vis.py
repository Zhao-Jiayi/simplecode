import imghdr
from turtle import shape


import cv2
import json
import os
import random
import numpy as np
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
def findALLUseful(path,all_useful_files,suf="jpg"):
    all_files = os.listdir(path)
    all_files = set(all_files)
    for each_path in all_files:
        cur_path = path + '/' + each_path
        if(os.path.isdir(cur_path)):
            findALLUseful(cur_path,all_useful_files)
        else:
            last_name = each_path[-len(suf):]
            if(last_name == suf):
                all_useful_files.append(cur_path)
def ramdomvis(path,outpath,count):
    files = []
    findALLUseful(path,files)
    

    for f in files:
        if(os.path.isdir(path+"/"+f) ):
            files.remove(f)
    if(count != 0):
        file_count = len(files)
        files = list(files)
        samples = random.sample(range(file_count), count)
        print(len(samples))
        for idx in samples:
            files.append(files[idx])
    vis(files,outpath)
def vis(files,outpath,oldpath = '/ssd8/other/zhaojiayi/code/vis823'):
    old_filse = set(os.listdir(oldpath))
    if(os.path.exists(outpath) == 0):
        os.mkdir(outpath)
    for img_file in files:

        
        if(img_file.split('/')[-1] in old_filse):
            continue
        json_file = img_file[0:-4]+".json"
        out_image = outpath + "/" + json_file.split('/')[-1][:-4] + "jpg"
        
        with open(json_file) as f:
            data = json.load(f)
            shapes = data["shapes"]

            img = cv2.imread(img_file)
            cls_id = 0
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':
                    if(len(shape['points'] )< 2):
       
                        continue
                    
                    polys = [[
                        round(shape['points'][0][0]), round(shape['points'][0][1]),
                        round(shape['points'][1][0]), round(shape['points'][0][1]),
                        round(shape['points'][1][0]), round(shape['points'][1][1]),
                        round(shape['points'][0][0]), round(shape['points'][1][1])]]
                elif shape['shape_type'] == 'polygon':
                    polys = [np.round(shape['points']).reshape(-1).tolist()]
                if len(polys[0]) < 6: continue
                polys = [[int(s) for s in polys[0]]]
    
                
                left  = np.min(polys[0][0::2])
                right = np.max(polys[0][0::2])
                upper = np.min(polys[0][1::2])
                lower = np.max(polys[0][1::2])
                try:
                    start = (left,upper)
                    end = (right,lower)
                except:
                    continue
                color = (_COLORS[cls_id%len(_COLORS)] * 255).astype(np.uint8).tolist()
                text = shape["label"]
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(img, start, end, color, 2)

         

                cv2.putText(img, text, (start[0]-5, start[1]), font, 0.75, color, thickness=2)
                cls_id+=1
            cv2.imwrite(out_image,img)
if __name__ == "__main__":
    # data = json.load(open("/ssd8/other/zhaojiayi/code/res.json"))
    # img_f = []
    # for f in data['res']:
    #     if(len(f)):
    #         for ff in f:
    #             img_f.append(ff[:-4]+"jpg")
    fs = []
    findALLUseful("../nopaper811/val2017",fs)
    vis(fs,"./vis830")