from dis import dis
import os
import json
from bisect import bisect_left,bisect_right
import cv2
from tqdm import tqdm
import numpy as np
import copy
import coco
def run(json_file,imgage_path,out_path,last = '_true'):
    c = coco.COCO(json_file)
    data = json.load(open(json_file))
    bbox_d = {}
    for a in data['annotations']:
        if bbox_d.get(c.id2filename[a['image_id']]) == None:
            bbox_d[c.id2filename[a['image_id']]] = [a['bbox']]
        else:
            bbox_d[c.id2filename[a['image_id']]].append(a['bbox']) 
  
    for key,val in bbox_d.items():
        with open(imgage_path+'/'+key[:-4]+last+'.txt','w') as f:
            f.write("%s"%(len(val)))
            f.write("\n")
            for b in val:
                f.write("%s %s %s %s \n" % (b[0], b[1], b[0] + b[2], b[1] + b[3]))
        os.system('cp {} {}'.format(imgage_path+'/'+key,out_path))
        os.system('mv {} {}'.format(imgage_path+'/'+key[:-4]+last+'.txt',out_path))
run('/ssd8/other/zhaojiayi/mydata712/annotations/instances_val_2017.json','/ssd8/other/zhaojiayi/mydata712/val2017','/ssd8/other/zhaojiayi/c++/test')