import json
import re
import random
from tkinter import image_names
import numpy as np
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
    def __init__(self,json_file) -> None:
        self.data = None
        self.images = None
        self.filename2id = dict()
        self.id2filename = dict()
        self.bboxes = dict()
        self.annotions = None
        self.image_count = 0
        print(re.findall("instances_(.*)?.json",json_file)[0])
        self.data_class = re.findall("/(.*)?2017",json_file)[0]
        self.catgoryid_to_name = dict()
      
        with open(json_file) as f:
            self.data = json.load(f)
            self.images = self.data["images"]
            self.image_count = len(self.images)
            self.annotions = self.data["annotations"] 
            for each_iamge in self.images:
                self.id2filename[each_iamge["id"]] = each_iamge["file_name"]
                self.filename2id[each_iamge["file_name"]] = each_iamge["id"]
                self.bboxes[each_iamge["id"]] = []
            cur_box = []
            for each_cat in self.data["categories"]:
                self.catgoryid_to_name[each_cat["id"]] = each_cat["name"]
            for each_anno in self.data["annotations"]:
                each_anno["bbox"].append(each_anno["category_id"])
                
                
                self.bboxes[each_anno["image_id"]].append(each_anno["bbox"])
                
             
                
    def romdamChoice(self,out_json_file,count):
        out_data = {}
        out_data["info"] = self.data["info"]
        out_data[ "categories"] = self.data[ "categories"]
        out_data["licenses"] = self.data["licenses"]
        out_data["images"] = []
        out_data["annotations"] = []
        print(self.image_count)
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
    def vis(self, count,path,outpath):

        samples = random.sample(range(self.image_count), count)
        for idx in samples:
   
            image_file = self.images[idx]["file_name"]
            img = cv2.imread(path + "/" + image_file)
            cls_id = 0
            print(self.bboxes[self.images[idx]["id"]])
            print(self.images[idx])
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
c = COCO("/ssd8/other/zhaojiayi/mydata630/instances_train2017.json")
c.vis(50,"/ssd8/other/zhaojiayi/mydata630/train2017","./vis620")