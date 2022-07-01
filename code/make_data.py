import os
import json

import cv2
from tqdm import tqdm
import numpy as np


LABEL_DICT = {
    'Text':1, 'Title':2, 'Figure':3, 'Figure Caption':4, 'Table':5,
    'Table Caption':6, 'Header':7, 'Footer':8, 'Reference':9,
    'Equation':10, 'Abstract':11, 'Author':12, 'List':13,'Section':14, 
    'Bib Info':15,'Content':16,'Code':17,'Affiliation':18,'PageNum':19}

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
    for file_id in list_file:

        # image_id += 100000
        print(os.path.join(data_dir, file_id ))
        anno_dict = json.load(
            open(os.path.join(data_dir, file_id )), strict=False)

        images.append({
            "license"       : 0,
            "file_name"     : file_id[:-5] + '.jpg',
            "coco_url"      : "",
            "height"        : anno_dict['imageHeight'],
            "width"         : anno_dict['imageWidth'],
            "date_captured" : "",
            "flickr_url"    : "",
            "id"            : image_id})
        
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
            
            annos.append({
                "id"           : anno_id,
                "image_id"     : image_id,
                "category_id"  : cate_id,
                "segmentation" : polys,
                "area"         : area,
                "bbox"         : bbox,
                "iscrowd"      : 0})
        image_id += 1
        
        
    info_dict = {}
    info_dict['info'] = {
        "description"   : "Question Segmentation 19-class - train6k",
        "url"           : "",
        "version"       : "train6k",
        "year"          : "2022",
        "contributor"   : "zhao jia yi",
        "date_created"  : "2022/6/9"}
    info_dict['licenses'] = [{
            "url": "",
            "id" : 0,
            "name": "Do not distribute under any fucking status !!!"}]
    info_dict['categories'] = [
        {"supercategory": "none", "id": 1, "name": "Text"},
        {"supercategory": "none", "id": 2, "name": 'Title'},
        {"supercategory": "none", "id": 3, "name": 'Figure'},
        {"supercategory": "none", "id": 4, "name": 'Figure Caption'},
        {"supercategory": "none", "id": 5, "name": 'Table'},
        {"supercategory": "none", "id": 6, "name": 'Table Caption'},
        {"supercategory": "none", "id": 7, "name": 'Header'},
        {"supercategory": "none", "id": 8, "name": 'Footer'},
        {"supercategory": "none", "id": 9, "name": 'Reference'},
        {"supercategory": "none", "id": 10, "name": 'Equation'},
        {"supercategory": "none", "id": 11, "name": 'Abstract'},
        {"supercategory": "none", "id": 12, "name": 'Author'},
        {"supercategory": "none", "id": 13, "name": 'List'},
        {"supercategory": "none", "id": 14, "name": 'Section'},
        {"supercategory": "none", "id": 15, "name": 'Bib Info'},
        {"supercategory": "none", "id": 16, "name": 'Content'},
        {"supercategory": "none", "id": 17, "name": 'Code'},
        {"supercategory": "none", "id": 18, "name": 'Affiliation'},
        {"supercategory": "none", "id": 19, "name": 'PageNum'}

        ]

    info_dict['images'] = images
    info_dict['annotations'] = annos
    json.dump(info_dict, open(output_path, 'w'))
    print(error_file)


if __name__ == '__main__':

    data_dir = '/ssd8/other/zhaojiayi/mydata630/val'
    output_path = '/ssd8/other/zhaojiayi/mydata630/instances_val2017.json'
    path = '/ssd8/other/zhaojiayi/mydata630/val'
    json_file = findALLFiles(path)
    print(len(json_file))
    make_data_json(data_dir, json_file)