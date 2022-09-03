from cProfile import label

import torch
import pickle
import cv2
import coco
import numpy as np
import json
import os
import vis
import re
import random
from tqdm import tqdm
def findBadHead(jsonfile,img_path,th = 0.7):
    c = coco.COCO(jsonfile)
    idxs = c.findClass(7)
    bad_files = []
    if(os.path.exists("badhead.pkl",)):
        bad_files = pickle.load(open("badhead.pkl",'rb'))
    for idx in idxs:
        image_file = img_path+ "/"+ c.images[idx]["file_name"]
        json_file =  image_file[0:-3] + "json"
        json_dict = {}
        with open(json_file) as f:
            json_dict = json.load(f)
            img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
            _,img = cv2.threshold(img, 240,255, cv2.THRESH_BINARY)
            for shape in json_dict["shapes"]:
                if(shape['label'] == "Header"):
                    ws = min(shape["points"][0][0],shape["points"][1][0])
                    hs = min(shape["points"][0][1],shape["points"][1][1])
                    we = max(shape["points"][0][0],shape["points"][1][0])
                    he = max(shape["points"][0][1],shape["points"][1][1])
                    
                    black = img[hs:he,ws:we].copy()

                    black = np.where(black == 0,1,0)
                    ww = np.sum(black,axis=0)
                    ww = np.where(ww>0,1,0)

                    sum_1 = ww.sum()
                    ration = sum_1/len(ww)
             
                    if(ration< th):
                        bad_files.append(image_file)                      
                        break
    print(len(bad_files))
    pickle.dump(bad_files,open("badhead.pkl",'wb'))
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)
def findBadSection(jsonfile,img_path,th = 0.3):
    c = coco.COCO(jsonfile)
    idxs = c.findClass(14)
    bad_files = []
    if(os.path.exists("badsection.pkl",)):
        bad_files = pickle.load(open("badsection.pkl",'rb'))
    for idx in idxs:
        image_file = img_path+ "/"+ c.images[idx]["file_name"]
        json_file =  image_file[0:-3] + "json"
        json_dict = {}
        with open(json_file) as f:
            json_dict = json.load(f)
            is_bad = 0
            
            for shape in json_dict["shapes"]:
                if(shape['label'] == "Section"):
                    ws = min(shape["points"][0][0],shape["points"][1][0])
                    hs = min(shape["points"][0][1],shape["points"][1][1])
                    we = max(shape["points"][0][0],shape["points"][1][0])
                    he = max(shape["points"][0][1],shape["points"][1][1])
             
                    width_e = 2 * (he-hs) + we
                    
                    width_s =  ws
                    print(width_e,width_s)
                    for shp in json_dict["shapes"]:
                        
                        if(shp["points"][0][0] < width_e\
                            and shp["points"][0][0] > width_s
                            and shp['label'] != "Section"
                            ):
                            min_h1 = 10000
                            min_h2 = 0
                            for points in shp["points"]:
                                min_h1 = min(min_h1,points[1])
                                min_h2 = max(min_h2,points[1])
                            
                            min_h1 = max(hs,min_h1)
                            min_h2 = min(min_h2,he)
                            
                            if(min_h2 < min_h1):
                                continue
                            
                            if((min_h2-min_h1)/(he-hs)>th and len(shp["points"])>2):

                                is_bad = 1
                                break
                    if is_bad:
                        bad_files.append(image_file)
                        break
    print(len(bad_files))
    pickle.dump(bad_files,open("badsection.pkl",'wb'))  
def findBadFooter(jsonfile,img_path):
    
    bad_files = []
    if(os.path.exists("badfoot2.pkl",)):
        bad_files = pickle.load(open("badfoot2.pkl",'rb'))
    c = coco.COCO(jsonfile)
    idxs = c.findClass(8)
    for idx in idxs:
        image_file = img_path+ "/"+ c.images[idx]["file_name"]
        json_file =  image_file[0:-3] + "json"
        data = json.load(open(json_file))
        max_height = 10000
        max_pos = 0
        for shape in data["shapes"]: 
            max_height = min(max_height,abs(shape["points"][1][1] - shape["points"][0][1]))
            if shape['label'] == 'Footer':
                max_pos = max(max(shape["points"][1][1],shape["points"][0][1]),max_pos)
        is_bad = 0
        for shape in data["shapes"]: 
            # if(min(shape["points"][0][1],shape["points"][0][0]) > max_pos and shape["label"] !="Footer" and shape["label"] !="Page Num"):

            #     print(shape["label"])
            #     print(json_file)
            #     is_bad = 1
            #     break
            if(shape["label"] == "Footer"):
                if(abs(shape["points"][1][1] - shape["points"][0][1])>max_height*4):
                    print(abs(shape["points"][1][1] - shape["points"][0][1]),max_height)
                    print(json_file)
                    is_bad = 1
                    break
        if(is_bad):
            bad_files.append(image_file)
    print(len(bad_files))
    pickle.dump(bad_files,open("badfoot2.pkl",'wb'))  
def findBadfigure(jsonfile,img_path):
    bad_files = []
    if(os.path.exists("../badfigure.pkl",)):
        bad_files = pickle.load(open("../badfigure.pkl",'rb'))
    c = coco.COCO(jsonfile)
    idxs = c.findClass(3)
    for idx in idxs:
        json_file = img_path+ "/"+ c.images[idx]["file_name"][0:-3]+ "json"
        json_dict = json.load(open(json_file))
        is_bad = False
        for shape in json_dict["shapes"]:
            if(is_bad):
                break
            if(shape['label'] == "Figure"):                
                hs = min(shape["points"][0][1],shape["points"][1][1])
                he = max(shape["points"][0][1],shape["points"][1][1])
                for shp in json_dict["shapes"]:
                    if(shp['label'] != 'Figure'):
                       
                        if(shp['points'][0][1] > shape['points'][0][1] and
                            shp['points'][0][0] > shape['points'][0][0] and
                            shp['points'][1][1] < shape['points'][1][1]  and
                            shp['points'][1][0] < shape['points'][1][0] 
                         ):
                            is_bad = True
                            break
                        
        if(is_bad):
            bad_files.append(img_path+ "/"+ c.images[idx]["file_name"])
                       

    print(bad_files)
    print(len(bad_files))
    pickle.dump(bad_files,open("../badfigure.pkl",'wb'))  
def findBadBigInfo(jsonfile,img_path):
    bad_files = []
    if(os.path.exists("../badbiginfo.pkl",)):
        bad_files = pickle.load(open("../badbiginfo.pkl",'rb'))
    c = coco.COCO(jsonfile)
    idxs = c.findClass(15)
    for idx in idxs:
        image_file = img_path+ "/"+ c.images[idx]["file_name"]
        
        bad_files.append(image_file)
    print(len(bad_files))
    pickle.dump(bad_files,open("../badbiginfo.pkl",'wb'))  
def findWindth(jsonfile,img_path):
    c = coco.COCO(jsonfile)
    bad_files = []
    if(os.path.exists("badwidth.pkl",)):
        bad_files = pickle.load(open("badwidth.pkl",'rb'))
    for img in c.images:
        if(img["height"]<img["width"]):
            bad_files.append(img_path+ "/"+ img["file_name"])
    print(len(bad_files))
    pickle.dump(bad_files,open("badwidth.pkl",'wb'))  
def getAllbad(badfiles,outpath):
    realation = {"head":1,"foot1":2.1,"foot2":2.2,"biginfo":3,"affiliation":4,"section":5,"figure":6}
    files = os.listdir(badfiles)
    final_data = []
    appers = dict()
    idx = 0
    with open(outpath,"w") as outf:
        for f in files:
            problem_class = re.findall('bad(.*)?.pkl',f)[0]
            bads = pickle.load(open(os.path.join(badfiles,f),'rb'))
            bads = set(bads)
            print(f,len(bads))
            for bad in bads:
                file_name = bad.split('/')[-1][:-4]
                if(appers.get(file_name) != None):
                    final_data[appers[file_name]]["num_list"].append(realation[problem_class])
                else:
                    cur_dict = {"file_name":file_name+".json","num_list":[realation[problem_class]]}
                    final_data.append(cur_dict)
                    appers[file_name] = idx
                    idx+=1
        json.dump(final_data,outf,indent=6)
def findBadAffiliation(jsonfile,img_path):
    bad_files = []
    if(os.path.exists("badaffiliation.pkl",)):
        bad_files = pickle.load(open("badaffiliation.pkl",'rb'))
    c = coco.COCO(jsonfile)
    idxs = c.findClass(18)
    for idx in idxs:
        image_file = img_path+ "/"+ c.images[idx]["file_name"]
        bad = 1
        json_file =  image_file[0:-3] + "json"
        data = json.load(open(json_file))
        for shape in data["shapes"]:
            if(shape["label"] == "Title" or shape["label"] == "Author" or  shape["label"] == "Abstract"):
                bad = 0
                break
        if bad:
            bad_files.append(image_file)
    pickle.dump(bad_files,open("badaffiliation.pkl",'wb'))  
def findBadFigure(jsonfile,imgpath,th=0.3):
    c = coco.COCO(jsonfile)
    idxs = c.findClass(3)
    bad_files = []
    if(os.path.exists("badfigure.pkl",)):
        bad_files = pickle.load(open("badfigure.pkl",'rb'))
    for idx in idxs:
        image_file = imgpath+ "/"+ c.images[idx]["file_name"]
        json_file =  image_file[0:-3] + "json"
        json_dict = {}
        with open(json_file) as f:
            json_dict = json.load(f)
            img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
            _,img = cv2.threshold(img, 240,255, cv2.THRESH_BINARY)
            is_bad = 0
            for shape in json_dict["shapes"]:
                if(is_bad):
                    break
                if(shape['label'] == "Figure"):                
                    hs = min(shape["points"][0][1],shape["points"][1][1])
                    he = max(shape["points"][0][1],shape["points"][1][1])
                    for shp in json_dict["shapes"]:
                        if(shp['label'] == 'Header' or shp['label'] == 'Footer'):
                            s =  min(shp["points"][0][1],shp["points"][1][1])
                            e = max(shp["points"][0][1],shp["points"][1][1])
                            min_h1 = max(hs,s)
                            min_h2 = min(he,e)
                            
                            if(min_h2 < min_h1):
                                continue
                            if(min_h2-min_h1/(e-s)>th):
                                is_bad = 1
                                break
            if(is_bad):
                bad_files.append(image_file)
    print(len(bad_files))
 
    pickle.dump(bad_files,open("badfigure.pkl",'wb'))
def visBad(path,outpath):
    all_files = pickle.load(open(path,'rb'))
    samples = set(random.sample(range(len(all_files)), len(all_files)-10))
    files = []
    for i in range(len(all_files)):
        if(i in samples):
            files.append(all_files[i])
    vis.vis(files,outpath)
def processFooter(path,th=0.3):
    imgpath = pickle.load(open("./bads/badfoot2.pkl",'rb'))
    bad_files = []
    if(os.path.exists("badfoot.pkl")):
        bad_files = pickle.load("badfoot.pkl")
    for img in tqdm(imgpath):
        data = json.load(open(img.replace("jpg","json")))
        foot_bboxs = []
        other_bboxs = []
        txt_file = path + "/" + img.split('/')[-1].replace("jpg",'txt')
        with open(txt_file) as f:
             for line in f:
                b = line.split(' ')[:4]
                for num in range(len(b)):
                    b[num] = int(b[num])
                other_bboxs.append(b)
        for shape in data["shapes"]:
            if(shape['label'] == 'Footer'):
                ws = min(shape["points"][0][0],shape["points"][1][0])
                hs = min(shape["points"][0][1],shape["points"][1][1])
                we = max(shape["points"][0][0],shape["points"][1][0])
                he = max(shape["points"][0][1],shape["points"][1][1])
                foot_bboxs.append([ws,hs,we,he])
        foot_bboxs = torch.tensor(foot_bboxs)
        other_bboxs = torch.tensor(other_bboxs)
        ious = bboxes_iou(foot_bboxs,other_bboxs)

        is_bad = 0
      
        for iou in ious:
            print(torch.where(iou>th))
            num = torch.where(iou>th)[0].shape[0]
            if(num > 1):
                is_bad = 1
                break
        if(is_bad):
            bad_files.append(img)
    print(len(bad_files))
    pickle.dump(bad_files,open("badfoot2.pkl",'wb'))
def visGoodFoot(path1,path2,outpath):
    list1 = set(pickle.load(open(path1,'rb')))
    list2 = set(pickle.load(open(path2,'rb')))
    final_list = []
    for f in list2:
        if f not in list1:
            if random.randint(1,30) == 1:
                final_list.append(f)
    vis.vis(final_list,outpath)
def copyBad(src,dst):
    # all_has = set()
    # bad_files = os.listdir(src)
    # for f in bad_files:
    #     cur_l = pickle.load(open(os.path.join(src,f),'rb'))

    #     all_has.update(cur_l)
    for f in tqdm(open(src,'r')):
        os.system("cp {} {}".format(f.split('\n')[0],dst))
        os.system("cp {} {}".format(f.split('\n')[0][:-4]+"jpg ",dst))
def findAllVal(valpath,badfilepath):
    all_has = set()
    bad_files = os.listdir(badfilepath)
    for f in bad_files:
        cur_l = pickle.load(open(os.path.join(badfilepath,f),'rb'))

        all_has.update(cur_l)
    val_files = os.listdir(valpath)
    add_files = []

    for v in val_files:
        if v[-1] == 'g':
            if(valpath+'/'+v not in all_has):
                add_files.append(valpath+'/'+v[:-3]+"json")
    print(len(add_files))
    with open("val_add.txt",'w') as f:
        for each in add_files:
            f.write("{}\n".format(each))
# 
with open("badfiguretrain0.txt",'w') as f:
    l = pickle.load(open("../badfigure.pkl",'rb'))
    for ll in l:
        if("train" in ll):
            f.writelines(ll.split("/")[-1])
            f.write('\n')
    # findWindth("/ssd8/other/zhaojiayi/mydata712/annotations/instances_train_2017.json","/ssd8/other/zhaojiayi/mydata712/train2017")
#visBad("/ssd8/other/zhaojiayi/code/bads/badaffiliation.pkl","./visaff")
# getAllbad("/ssd8/other/zhaojiayi/code/bads","bad.json")
# processFooter("/ssd8/other/liyx/txts")
# visBad("/ssd8/other/zhaojiayi/code/badfoot2.pkl","./viswidth")
# visGoodFoot("./badfoot2.pkl","./bads/badfoot2.pkl","visgoodfoot")
# findAllVal("/ssd8/other/zhaojiayi/mydata712/val2017","./bads")
# copyBad("/ssd8/other/zhaojiayi/code/val_add.txt","/ssd8/other/zhaojiayi/biaozhuproblem/val")