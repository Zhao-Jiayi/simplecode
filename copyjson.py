import os
def copyJson(source_dir,tar_dir):
    all_fs = os.listdir(tar_dir)
    for f in all_fs:
        f = f[:-3] + "json"
        os.system("cp {} {}".format(source_dir+'/'+f,tar_dir+'/'+f))
copyJson("/ssd8/other/liyx/anno_data_27k_fix",'/ssd8/other/zhaojiayi/mydata712/val2017')