from ..yolov7.utils.metrics import ConfusionMatrix
import os
import json
import torch
names = ['Text', 'Title', 'Figure', 'Figure Caption', 'Table',
    'Table Caption', 'Header', 'Footer', 'Reference',
    'Equation', 'Abstract', 'Author', 'List','Section', 
    'Bib Info','Content','Code','Affiliation','PageNum']
def draw(data_dir,res_path,class_nums):
    confusion_matrix = ConfusionMatrix(nc=class_nums)
    image_id_dict = {i['id']:i['file_name'] for i in json.load(open(os.path.join(data_dir, 'annotations/instances_val_2017.json')))['images']}
    image_filename_dict= {i['file_name'] :i['id'] for i in  json.load(open(os.path.join(data_dir, 'annotations/instances_val_2017.json')))['images']}
    data_pred = {}
    data_true = {}
    with open(res_path) as f:
        data = json.load(f)
        for each_d in data:
            cur_pred = each_d["bbox"]
            cur_pred.append(each_d["score"])
            cur_pred.append(each_d["category_id"])
            if(data_pred.get(each_d["image_id"]) == None):             
                data_pred[each_d["image_id"]] = [cur_pred]
            else:
                data_pred[each_d["image_id"]].append(cur_pred)
    with open(os.path.join(data_dir, 'annotations/instances_val_2017.json')) as f:
        data = json.load(f)["annotations"]
        for each_d in data:
            cur_pred = each_d["category_id"]
            for xyxy in each_d["bbox"]:
                cur_pred.append(xyxy)
            if(data_true.get(each_d["image_id"]) == None):             
                data_true[each_d["image_id"]] = [cur_pred]
            else:
                data_true[each_d["image_id"]].append(cur_pred)
    for key,val in data_true: 
        confusion_matrix.process_batch(val, data_true[key])
    confusion_matrix.plot(save_dir="matric.jpg", names=names)
draw("/ssd8/other/zhaojiayi/mydata712","/ssd8/other/zhaojiayi/post_27k.json",19)