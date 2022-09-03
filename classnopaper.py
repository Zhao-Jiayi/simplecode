from logging.config import valid_ident
import os
import numpy as np
import json
import random
from functools import cmp_to_key
def findALLUseful(path,all_useful_files,flags,suf="jpg"):
    all_files = os.listdir(path)
    all_files = set(all_files)
    for each_path in all_files:
        cur_path = path + '/' + each_path
        if(os.path.isdir(cur_path)):
            findALLUseful(cur_path,all_useful_files,flags,suf)
        else:
            last_name = each_path[-len(suf):]
            if(last_name == suf):
                has = 0
                for flag in flags:
                    if(flag in cur_path):
                        has = 1
                        break
                if(has):
                    continue
                all_useful_files.append(cur_path)
def cmp(s1,s2):
    if(len(s1) == len(s2)):
        if s1 < s2:
            return -1
        else:
            return 1
    else:
        if(len(s1) < len(s2)):
            return -1
        else:
            return 1

def choiceFile(path,outpath,flags=["论文","文章"]):
    fs = []
    findALLUseful(path,fs,flags)
    prename_dict = {}
    
    idx_to_files = []
    for f in fs:
        pre = f.split('_')[0]
        if(prename_dict.get(pre) == None):
            prename_dict[pre] = [f]
            idx_to_files.append(pre)
        else:
            prename_dict[pre].append(f)
    for key,_ in prename_dict.items():
        if(len(prename_dict[key]) <= 30):
            continue
        prename_dict[key].sort(key=cmp_to_key(cmp))
        
        new_val = prename_dict[key][0:10] 
        
        for f in prename_dict[key][-10:]:
            new_val.append(f)
        samples = random.sample(range(0,len(prename_dict[key])-20),10)
        for s in samples:
            new_val.append(prename_dict[key][10+s])
        
        prename_dict[key] = new_val
    for _,val in prename_dict.items():
        for f in val:
            os.system("ln -s '{}' '{}'".format(f,outpath))
def findNopaper(path):
    fs = []
    findALLUseful(path,fs,flags=[],suf = "jpg")
    prename_dict = {}
    
    papers = []
    for f in fs:
        pre = f.split('_')[0]
        if(prename_dict.get(pre) == None):
            prename_dict[pre] = [f]
 
        else:
            prename_dict[pre].append(f)
    for _,fs in prename_dict.items():
        cur_file_count = len(fs)
        is_paper = 0
        no_paper = 0
        w_gt_h = 0
        for f in fs :
            
            data = json.load(open(f))
            
            if(data['imageHeight'] < data['imageWidth']):
                w_gt_h+=1
                if(w_gt_h > cur_file_count * 0.3):
                    no_paper = 1
            if(len(data["shapes"])==0):
                no_paper = 1
            for shp in data[ "shapes"]:
                if(shp['label'] == 'Bib Info' or shp['label'] == 'Reference'):
                    is_paper = 1
        if(no_paper or len(fs) < 4):
            continue
        if(is_paper):
            for ff in fs:
                papers.append(ff[:-4]+"jpg")
    print(len(prename_dict))
    # data = {}
    # for p in papers:
    #     data[p] = os.path.realpath(p)
    # json.dump(data,open("nopaper.json",'w'),indent=6)
def divide(path,outpath):
    
    all_files = os.listdir(path)
    train_list = {}
    val_list = {}
    all_file_idxs = []
    all_file_pairs=dict()
    # all_file_idxs,all_file_pairs = getRelations(json_file="/ssd8/other/zhaojiayi/mydata712/anno_data_27k.json")
   
    for each_file in all_files:
        pre = each_file.split("_")[0]


        if pre not in all_file_idxs:
            all_file_idxs.append(pre)
        if(each_file[-3:] == "jpg"):
            if(all_file_pairs.get(pre) == None):
                all_file_pairs[pre] = [[each_file[:-4]+'json',each_file]]
            else:
                all_file_pairs[pre].append([each_file[:-4]+'json',each_file])

    files_count = len(all_file_pairs)
    samples = random.sample(range(0,files_count), 193)
    samples = set(samples)
    img_file = 1
    for idx in range(files_count):
        if(idx in samples):
            for each_page in all_file_pairs[all_file_idxs[idx]]:
                # if(os.path.exists(path+"/"+each_page[0]) and os.path.exists(path+"/"+each_page[1]) ):
                # os.system("cp '{}' '{}'".format(path+"/"+each_page[0],outpath+'/'+each_page[0]))
                os.system("cp '{}' '{}'".format(path+"/"+each_page[1],outpath+'/'+each_page[1]))
                # else:
                    
            val_list[all_file_idxs[idx]] = img_file
      
    
        img_file+=1
def divide2(path,outpath):
    
    all_files = os.listdir(path)
    train_list = {}
    val_list = {}
    all_file_idxs = []
    all_file_pairs=dict()
    # all_file_idxs,all_file_pairs = getRelations(json_file="/ssd8/other/zhaojiayi/mydata712/anno_data_27k.json")
    ss = os.listdir(outpath)
    for each_file in ss:
        
        if(os.path.isdir(each_file)):
            continue
        pre = each_file.split("_")[0]
  
        
        if pre not in all_file_idxs:
            all_file_idxs.append(pre)
        if(each_file[-3:] == "jpg"):
            if(each_file not in all_files):
                print(each_file)
            if(all_file_pairs.get(pre) == None):
                all_file_pairs[pre] = [[each_file[:-4]+'jpg',each_file]]
            else:
                all_file_pairs[pre].append([each_file[:-4]+'jpg',each_file])

    files_count = len(all_file_pairs)
    p = ["1455","1456","2927","4750","9420","11630","14488"]
    print(files_count)
    # for pre,fs in all_file_pairs.items():
    #     if(pre in p):
    #         for each_page in fs:
    #             # if(os.path.exists(path+"/"+each_page[0]) and os.path.exists(path+"/"+each_page[1]) ):
    #             os.system("mv '{}' '{}'".format(path+"/"+each_page[0],outpath+'/1/'+each_page[0]))
    #             os.system("mv '{}' '{}'".format(path+"/"+each_page[1],outpath+'/1/'+each_page[1]))
    #             # else:
                    
    #     else:
    #          for each_page in fs:
    #             # if(os.path.exists(path+"/"+each_page[0]) and os.path.exists(path+"/"+each_page[1]) ):
    #             os.system("mv '{}' '{}'".format(path+"/"+each_page[0],outpath+'/2/'+each_page[0]))
    #             os.system("mv '{}' '{}'".format(path+"/"+each_page[1],outpath+'/2/'+each_page[1]))
                # else:
# findNopaper("/ssd8/other/zhaojiayi/class811")
divide2("/ssd8/other/zhaojiayi/nopaper817val/val819/2","/ssd8/other/zhaojiayi/val819/nopaper817val")