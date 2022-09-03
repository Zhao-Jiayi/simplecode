import os 
def findALLUseful(path,all_useful_files,suf="jpg"):
    all_files = os.listdir(path)
    all_files = set(all_files)
    for each_path in all_files:
        cur_path = path + '/' + each_path
        if(os.path.isdir(cur_path)):
            findALLUseful(cur_path,all_useful_files)
        else:
            last_name = each_path[-3:]
            if(last_name == suf):
                all_useful_files.append(cur_path)
def lnimg(path,out):
    files = []
    findALLUseful(path,files)
    for f in files:
        os.system("ln -s '{}' '{}'".format(f,out))

lnimg("/ssd8/other/zhaojiayi/image77","/ssd8/other/zhaojiayi/Classification_MobileNetV3/image/original/2")
