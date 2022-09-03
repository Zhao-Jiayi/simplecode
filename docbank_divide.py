from asyncore import poll
import json
import os
import re
from multiprocessing import Pool
def divide(path,img_path,outpath,prop_count,comand):
    with open(path) as f:
        data = json.load(f)
        pool = Pool(prop_count)
        images = data["images"]
        dis = len(images) / prop_count
        idx = 0
        my_prop = []
        for i in range(prop_count):
    
            if idx + dis + 10 > len(images):
                pool.apply_async(prop,(idx,len(images),comand,img_path,outpath,images,))
            else:
                pool.apply_async(prop,(idx,idx+dis,comand,img_path,outpath,images,))
            idx += dis
        pool.close()
        pool.join()
       
        
def prop(start_idx,end_idx,comand,img_path,outpath,images):
    
    for idx in range(int(start_idx),int(end_idx)):
        
        file_name = images[idx]["file_name"]
        print(file_name)
        os.system("{} {} {}".format(comand,img_path+"/"+file_name,outpath+"/"+file_name))
divide("/ssd8/other/zhaojiayi/docbank/COCO/annotations/instances_train2017.json","/ssd8/other/zhaojiayi/mydataset/DocBank_500K_ori_img/DocBank_500K_ori_img",'/ssd8/other/zhaojiayi/docbank/COCO/train2017',10,'cp')