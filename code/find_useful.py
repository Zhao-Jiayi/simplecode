import json
import os
def findALLUseful(path,all_useful_files):
    all_files = os.listdir(path)
    all_files = set(all_files)
    for each_path in all_files:
        cur_path = path + '/' + each_path
        if(os.path.isdir(cur_path)):
            findALLUseful(cur_path,all_useful_files)
        else:
            last_name = each_path.split('.')
            last_name = last_name[len(last_name)-1]
   
            if(last_name == "jpg"):
                json_file_name = each_path[:len(each_path)-4]+'.json'
          
                if(json_file_name in all_files):
                    all_useful_files.append(cur_path)
                
def copyFiles(filelist):
    for file in filelist:
        target_file_jpg = file.split('/')
        target_file_jpg = target_file_jpg[len(target_file_jpg)-1]
        target_file_json = target_file_jpg[:len(target_file_jpg)-4]+".json"  
        print(target_file_json)
        os.system("cp {} {}".format(file,target_file_jpg))
        os.system("cp {} {}".format(file[:len(file)-4]+".json",target_file_json))
if __name__ == "__main__":
    filelist = []
    findALLUseful("/ssd8/other/wugx/data/pdf/log_datasets/dataset1",filelist)
   
    copyFiles(filelist)