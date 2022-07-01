import json
import os
classes = {"computer":0,"other":0,"finance":0,"biotechnology":0}
def getCount(dir):
    all_files = os.listdir(dir)
    for f in all_files:
        if f[0] == '1':
            classes["computer"]+=1
        elif f[0] == '2':
            classes["finance"]+=1
        elif f[0] == '3':
            classes["biotechnology"]+=1
        else:
            classes["other"]+=1
getCount("/ssd8/other/zhaojiayi/mydata/COCO/jsonfile/")
for key in classes:
    print(key,classes[key])