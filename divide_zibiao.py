from email.mime import image
import json
import os
import random
import re 
import numpy as np
from ssl import SSLSocket
def divide(path,out_path,ratio = 0.05,old_path = ["/ssd8/other/zhaojiayi/mydata712/train2017","/ssd8/other/zhaojiayi/mydata712/val2017"]):
    print(os.curdir)
    print(path)
    old_pre = {}
    if old_path != []:
        for each_path  in old_path:
            fs = os.listdir(each_path)
            for f in fs:
                if(old_pre.get(f.split('_')[0])) == None:
                    old_pre[f.split('_')[0]] = []
                old_pre[f.split('_')[0]].append(each_path+"/"+f)
    
    all_files = os.listdir(path)
    print(all_files)
    train_list = {}
    val_list = {}
    all_file_idxs = []
    all_file_pairs=dict()
    # all_file_idxs,all_file_pairs = getRelations(json_file="/ssd8/other/zhaojiayi/mydata712/anno_data_27k.json")
   
    for each_file in all_files:
        pre = each_file.split("_")[0]
        if(old_pre.get(pre) != None):
            for f in old_pre[pre]:
                if('val' in f):
                    os.system("ln -s '{}' '{}'".format(path+"/"+each_file,out_path+'/'+'val2017'))
                    os.system("ln -s '{}' '{}'".format(path+"/"+each_file,out_path+'/'+'val2017'))
                else:
                    os.system("ln -s '{}' '{}'".format(path+"/"+each_file,out_path+'/'+'train2017'))
                    os.system("ln -s '{}' '{}'".format(path+"/"+each_file,out_path+'/'+'train2017'))
            continue

        if pre not in all_file_idxs:
            all_file_idxs.append(pre)
        if(each_file[-4:] == "json"):
            if(all_file_pairs.get(pre) == None):
                all_file_pairs[pre] = [[each_file[:-4]+'jpg',each_file]]
            else:
                all_file_pairs[pre].append([each_file[:-4]+'jpg',each_file])

    files_count = len(all_file_pairs)
    samples = random.sample(range(0,files_count), int(files_count*ratio))
    samples = set(samples)
    img_file = 1
    for idx in range(files_count):
        if(idx in samples):
            for each_page in all_file_pairs[all_file_idxs[idx]]:
                # if(os.path.exists(path+"/"+each_page[0]) and os.path.exists(path+"/"+each_page[1]) ):
                os.system("ln -s '{}' '{}'".format(path+"/"+each_page[0],out_path+'/'+'val2017'+'/'+each_page[0]))
                os.system("ln -s '{}' '{}'".format(path+"/"+each_page[1],out_path+'/'+'val2017'+'/'+each_page[1]))
                # else:
                    
            val_list[all_file_idxs[idx]] = img_file
        else:
            for each_page in all_file_pairs[all_file_idxs[idx]]:
                # if(os.path.exists(path+"/"+each_page[0]) and os.path.exists(path+"/"+each_page[1]) ):
                os.system("ln -s '{}' '{}'".format(path+"/"+each_page[0],out_path+'/'+'train2017'+'/'+each_page[0]))
                os.system("ln -s '{}' '{}'".format(path+"/"+each_page[1],out_path+'/'+'train2017'+'/'+each_page[1]))
                # else:
                #     print(os.path.exists(path+"/"+each_page[0]))
            train_list[all_file_idxs[idx]] = img_file
    
        img_file+=1
        json.dump(train_list, open(out_path+"/train.json", 'w'),indent=6)
        json.dump(val_list, open(out_path+"/val.json", 'w'),indent=6)
def findOld(oldfilelist,new_file_path=""):
    old_file = set()
    for i in oldfilelist:
        with open(i) as f:
            data = json.load(f)
            for key in data.keys():
                print(key)
                file_name = re.findall(".*?_(.*)-",key)
                if len(file_name) > 0:
                    file_name = file_name[0]
                else:
                    file_name = re.findall("(.*?)[（|\(]",key)
                print(file_name)
                old_file.add(file_name[0])
def getRelations(json_file):
    data = {}
    all_files_idxs =[]
    all_files = {}
    with open(json_file) as f:
        data = json.load(f)
        for key,value in data.items():
            
            paper = key.split('_')[0]
            print(paper)
            if(all_files.get(paper) == None):
                all_files[paper] = [[key+".json",key+".jpg"]]
                all_files_idxs.append(paper)

            else:
                all_files[paper].append([key+".json",key+".jpg"])
    return all_files_idxs,all_files
def replaceold(path,out_path,ratio = 0.05,old_path = ["/ssd8/other/zhaojiayi/mydata712/train2017","/ssd8/other/zhaojiayi/mydata712/val2017"]):
    old_files = []
    for old_p in old_path:
        fs = os.listdir(old_p)
        for f in fs :
            old_files.append(old_p + '/' + f)
    for f in old_files:
        if('val' in f):
            os.system("ln -s {} {}".format(path + "/" + f.split('/')[-1],out_path+'/'+"val"))
        else:
            os.system("ln -s {} {}".format(path + "/" + f.split('/')[-1],out_path+'/'+"train"))
# divide("/ssd8/other/zhaojiayi/data0702/","/ssd8/other/zhaojiayi/mydata72/")
replaceold("/ssd8/other/liyx/pdf_labelme_data/paper/anno_data_27k_fix2","/ssd8/other/zhaojiayi/mydata712")