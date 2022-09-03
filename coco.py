import json
import re
import random
from tkinter import image_names
import numpy as np
import os

import cv2
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
class COCO:
    def __init__(self,json_file,error_file = [10196]) -> None:
        self.data = None
        self.images = None
        self.filename2id = dict()
        self.id2filename = dict()
        self.bboxes = dict()
        self.annotions = None
        self.image_count = 0
        self.json_file = json_file
        self.error_file = error_file
      
        self.data_class = re.findall("/(.*)?2017",json_file)[0]
        self.catgoryid_to_name = dict()
      
        with open(json_file) as f:
            self.data = json.load(f)
            self.images = self.data["images"]
            self.image_count = len(self.images)
            self.annotions = self.data["annotations"] 
            self.id2idx = {}
            idx = 0
            for each_iamge in self.images:
                self.id2filename[each_iamge["id"]] = each_iamge["file_name"]
                self.id2idx[each_iamge["id"]] = idx
                idx+=1
                self.filename2id[each_iamge["file_name"]] = each_iamge["id"]
                self.bboxes[each_iamge["id"]] = []
            cur_box = []
            for each_cat in self.data["categories"]:
                self.catgoryid_to_name[each_cat["id"]] = each_cat["name"]
            for each_anno in self.data["annotations"]:
                each_anno["bbox"].append(each_anno["category_id"])
                
                if(each_anno["image_id"] in error_file):
                    continue
                self.bboxes[each_anno["image_id"]].append(each_anno["bbox"])
                
             
                
    def romdamChoice(self,out_json_file,count):
        out_data = {}
        out_data["info"] = self.data["info"]
        out_data[ "categories"] = self.data[ "categories"]
        out_data["licenses"] = self.data["licenses"]
        out_data["images"] = []
        out_data["annotations"] = []

        samples = random.sample(range(1,self.image_count+1),count)
        samples = set(samples)
        for each_image in self.images:
            cur_id = each_image["id"]
            if cur_id in samples:
                out_data["images"].append(each_image)
        for each_annotions in self.annotions:
            if each_annotions["image_id"] in samples:
                out_data["annotations"].append(each_annotions)

        with open(out_json_file,'w') as f:
            json.dump(out_data,f,indent = 6)
    def findClass(self,id):
        idxs = set()
        for each_annotation in self.annotions:
            if(each_annotation["category_id"]== id):
                if(each_annotation["image_id"] in self.error_file):
                    continue
                idxs.add(int(self.id2idx[each_annotation["image_id"]]))
        return idxs
    def randGenerate(self,path,count):
        samples = random.sample(range(self.image_count), count)
        res = []
        
        return samples
    def vis(self,idxs, path,outpath):
        if(os.path.exists(outpath) == False):
            os.mkdir(outpath)
        
        for idx in idxs:

            image_file = self.images[idx]["file_name"]
            img = cv2.imread(path + "/" + image_file)
            cls_id = 0
            # print(self.bboxes[self.images[idx]["id"]])
            # print(self.images[idx])
            for box in self.bboxes[self.images[idx]["id"]]:
              
                start = (int(box[0]),int(box[1]))
                end =  (int(box[0]+box[2]),int(box[1]+box[3]))
                color = (_COLORS[cls_id%79] * 255).astype(np.uint8).tolist()
                text = self.catgoryid_to_name[box[4]]
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(img, start, end, color, 2)

        

                cv2.putText(img, text, (start[0]-5, start[1]), font, 0.75, color, thickness=2)
                cls_id+=1
            
            cv2.imwrite(outpath+'/'+image_file,img)
    def visEval(self,eval_path,img_path,vispath):
        data = {}
        mydata = {}
        with open(eval_path) as f:
            data = json.load(f)
        for each_d in data:
            if(mydata.get(each_d["image_id"]) == None):
                mydata[each_d["image_id"]] = [each_d]
            else:
                mydata[each_d["image_id"]].append(each_d)
        if(os.path.exists(vispath)==0):
            os.mkdir(vispath)
        for img_file,annotations in mydata.items():
            img = "{}{}.jpg".format(img_path,img_file)
 
            img = cv2.imread(img)
            cls_id = 0
            for each_b in annotations:
                
                start = (int(each_b["bbox"][0]),int(each_b["bbox"][1]))
                end =  (int(each_b["bbox"][0]+each_b["bbox"][2]),int(each_b["bbox"][1]+each_b["bbox"][3]))
                color = (_COLORS[cls_id%79] * 255).astype(np.uint8).tolist()
                text = self.catgoryid_to_name[each_b["category_id"]]
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(img, start, end, color, 2)

        

                cv2.putText(img, text, (start[0]-5, start[1]), font, 0.75, color, thickness=2)
                cls_id+=1
            print(vispath+'/'+str(img_file)+".jpg")
            cv2.imwrite(vispath+'/'+str(img_file)+".jpg",img)
if __name__ == "__main__":
    c = COCO("/ssd8/other/zhaojiayi/mydata712/annotations/instances_val_2017.json")
    c.vis(c.findClass(15),"/ssd8/other/zhaojiayi/mydata712/val2017","./visbibinfo")
