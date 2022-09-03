# import pickle
# import os
# l = pickle.load(open("/ssd8/other/zhaojiayi/code/badbiginfo.pkl",'rb'))
# print(len(l))
# for ll in l :
#     os.system("cp {} {} ".format(ll,"/ssd8/other/zhaojiayi/code/allbibinfo"))

import os 
from tqdm import tqdm
import random
import numpy as np
def findALLUseful(path,all_useful_files,flags,suf="jpg"):
    all_files = os.listdir(path)
    all_files = set(all_files)
    for each_path in all_files:
        cur_path = path + '/' + each_path
        if(os.path.isdir(cur_path)):
            findALLUseful(cur_path,all_useful_files,flags)
        else:
            last_name = each_path[-3:]
            if(last_name == suf):
                has = 0
                for flag in flags:
                    if(flag in cur_path):
                        has = 1
                        break
                if(has):
                    continue
                all_useful_files.append(cur_path)
def run(path,out):
    files = os.listdir(path)
    c = 0
    d = {}
    for f in tqdm(files):
        c+=1
        d[f.split('_')[0]] = 0
        if(c > 28500 and d.get(f) == None):
            break
        os.system("ln -s {} {} ".format(path+"/"+f,out))
def randomChoice(path,outpath,flags =["论文","文章"]):
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
    idxs = set(np.random.randint(0,len(idx_to_files),100))
    for idxf in range(1,len(idx_to_files)):
        if(idxf in idxs):
            print(idx_to_files[idxf])
            prename_dict[idx_to_files[idxf]] = set(prename_dict[idx_to_files[idxf]] )
            for f in prename_dict[idx_to_files[idxf]]:
                outpaths = f.split('/')
                temp = outpath + '/'+ outpaths[-2] 
      
                if(os.path.isdir(temp) == 0):
                    os.mkdir(temp)
                print("ln -s '{}' '{}'".format(f,temp))
                os.system("ln -s '{}' '{}'".format(f,temp))
# run("/ssd8/other/zhaojiayi/class811/应聘","/ssd8/other/zhaojiayi/Classification_MobileNetV3/image/original/2")
randomChoice("/ssd8/other/zhaojiayi/class811","/ssd8/other/zhaojiayi/nopaper100")