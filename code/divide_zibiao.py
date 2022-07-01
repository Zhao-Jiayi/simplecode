from email.mime import image
import json
import os
import random
def divide(path,out_path,ratio = 0.05):
    print(os.curdir)
    print(path)
    all_files = os.listdir(path)
    print(all_files)
    all_file_pairs = []
    val_file = []
    train_file = []
    for each_file in all_files:
        if(each_file[-4:] == "json"):
            all_file_pairs.append([path+'/'+each_file,path+'/'+each_file[:-4]+'jpg'])
        
    print(len(all_file_pairs))
    files_count = len(all_file_pairs)
    samples = random.sample(range(0,files_count), int(files_count*ratio))
    samples = set(samples)

    for idx in range(files_count):
        if(idx in samples):
            os.system("cp '{}' '{}'".format(all_file_pairs[idx][0],out_path+'/'+'val'+'/'+all_file_pairs[idx][0].split('/')[-1]))
            os.system("cp '{}' '{}'".format(all_file_pairs[idx][1],out_path+'/'+'val'+'/'+all_file_pairs[idx][1].split('/')[-1]))
        else:
            os.system("cp '{}' '{}'".format(all_file_pairs[idx][0],out_path+'/'+'train'+'/'+all_file_pairs[idx][0].split('/')[-1]))
            os.system("cp '{}' '{}'".format(all_file_pairs[idx][1],out_path+'/'+'train'+'/'+all_file_pairs[idx][1].split('/')[-1]))
divide("/ssd8/other/zhaojiayi/data630/PDF_parsing_data/anno_data","/ssd8/other/zhaojiayi/mydata630/")
