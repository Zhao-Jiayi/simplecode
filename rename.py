import os
import json
file_path = "/ssd8/other/zhaojiayi/mydata72/train2017"
all_files = os.listdir(file_path)
idx = 1
out_f = "/ssd8/other/zhaojiayi/train.json"
out_d = {}
pre_d = {}
for f in all_files:
    
    ff = file_path + "/" + f
    pre = ""
    if('.jpg' in f):
        pre = f[:len(f)-4]
    else:
        pre = f[:len(f)-5]
    print(pre)
    if(pre_d.get(pre) == None):
        pre_d[pre] = idx
        out_d[pre] = idx
        ft = file_path + "/" + str(idx)
        if('.json' in f):
            ft += ".json"
        else:
            ft += ".jpg"
        
        os.system("mv '{}' '{}'".format(ff,ft))
        idx+=1
    else:
        ft = file_path + "/" + str(pre_d[pre])
        if('.json' in f):
            ft += ".json"
        else:
            ft += ".jpg"
        os.system("mv '{}' '{}'".format(ff,ft))
with open(out_f,'w') as f:
    json.dump(out_d, f, indent=2, sort_keys=True, ensure_ascii=False)