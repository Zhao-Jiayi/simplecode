import json
import os
import cv2
import pickle

data = json.load(open("/ssd8/other/zhaojiayi/code/bibinfo.json"))
fs = []
for f in data["resval"]:
    for ff in f:
        if(len(ff)):
            fs.append(ff)
print(len(fs))
for f in fs:
    img_path = ""
    json_path = ""
    if("g" == f[-1]):

        img_path = f
        json_path = f[:-3] + "json"
    else:
        img_path = f[:-4] + "jpg"
        json_path = f
    os.system("cp {} {}".format(img_path,"/ssd8/other/zhaojiayi/code/bibinfo"))
    os.system("cp {} {}".format(json_path,"/ssd8/other/zhaojiayi/code/bibinfo"))