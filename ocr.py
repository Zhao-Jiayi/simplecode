from asyncore import poll
import errno
import cv2
import base64
import urllib
import json
import urllib.request
import pickle
import os
from tqdm import tqdm
import time
import torch
import numpy as np
from multiprocessing import Pool
ocr_url = 'http://api2.ocr.youdao.com/accurate_ocr'
badfilepath = "/ssd8/other/zhaojiayi/code/badbiginfo.pkl"
file_list = pickle.load(open(badfilepath,'rb'))
print(len(file_list))
pos_dict = {}
freq = {}
text_bbox = {}

def getPos(name_list,d = 20):
    data = json.load(open("./text_box.json"))
    name_list = set(name_list)
    for poses in data:
        if(poses[0] in name_list):
            cur_d = {}
            for p in poses[1]:
                p_list = [int(int(i)/d) for i in p.split(',')]
                p_s = "{},{},{},{}".format(p_list[0],p_list[1],p_list[4],p_list[5])
                if(cur_d.get(p_s) == None):
                    cur_d[p_s] = 0
                cur_d[p_s]+=1
            cur_d =  sorted(cur_d.items(), key=lambda d: d[1],reverse=True)
            len_d = min(3,len(cur_d))
            new_d = cur_d[0:len_d]
            pos_dict[poses[0]] = []
            for i in new_d:
                pos_dict[poses[0]].append([int(ii)*d for ii in i[0].split(',')])
    print(pos_dict)
def request_ocr_service(image):
    try:
        imgb64 = base64.b64encode(cv2.imencode('.jpg', image)[1].tostring())
    except:
        return {}
    input_dict = {'img':imgb64}
    
    data = urllib.parse.urlencode(input_dict).encode("utf-8")
    f = urllib.request.urlopen(url=ocr_url, data=data)
    res_dict = json.loads(json.loads(f.read())['Result'])
    
    return res_dict
def findF(files,file_path):

    error_file = []
    for f in tqdm(files):
        print("当前进程：", os.getpid(), " 父进程：", os.getppid())
        json_file = ""
        jpg_file = ""
        if(f[-1] == 'g'):
            json_file = f[:-3] + "json"
            jpg_file = f
        else:
            continue
        json_file = os.path.join(file_path,json_file)
        jpg_file = os.path.join(file_path,jpg_file)
        data = json.load(open(json_file))
        b1 = []
        b2 = []
        has_bib = 0
        for shape in data["shapes"]:
            if(shape['label'] == "Bib Info"):
                has_bib = 1
                break
        if(has_bib == 0):

            # try:
            #     polys = []
            #     if shape['shape_type'] == 'rectangle':
                    
                        
            #         polys = [[
            #             round(shape['points'][0][0]), round(shape['points'][0][1]),
            #             round(shape['points'][1][0]), round(shape['points'][0][1]),
            #             round(shape['points'][1][0]), round(shape['points'][1][1]),
            #             round(shape['points'][0][0]), round(shape['points'][1][1])]]
            #     if shape['shape_type'] == 'polygon':
            #         polys = [np.round(shape['points']).reshape(-1).tolist()]
            #         if len(polys[0]) < 6: continue
           
            #     polys = [[int(s) for s in polys[0]]]
              
                
            #     left  = np.min(polys[0][0::2])
            #     right = np.max(polys[0][0::2])
            #     upper = np.min(polys[0][1::2])
            #     lower = np.max(polys[0][1::2])
            #     bbox = [int(left), int(upper), right, lower]

            #     b1.append(bbox)
            # except:
            #     continue
            res_dict = {}
        
            try:
                res_dict = request_ocr_service(cv2.imread(jpg_file))
            except:
                continue
            if(res_dict == {}):
                continue
            
            for region in res_dict["regions"]:
                for line in region["lines"]:
                    if(pos_dict.get(line['text'])):
                        b = [int(i) for i in line['boundingBox'].split(',')]
                        b2 = [[b[0],b[1],b[4],b[5]]]
            
            
            
                        if(caluculatIou(pos_dict[line['text']],b2) == 0):
                            print(json_file)
                            error_file.append(json_file)
            
    error_file = list(set(error_file))
    # print(error_file)
    # json.dump(errno,open('res{}.json'.format(os.getpid()),'w'))
    return error_file
def caluculatIou(b1,b2,th = 1e-1):   
    t_b1 = []
    t_b2 = []
    for b in b1:
        t_b1.append(torch.tensor(b))
    for b in b2:
        t_b2.append(torch.tensor(b))

    t_b1 = torch.stack(t_b1)
    t_b2 = torch.stack(t_b2)

    tl = torch.max(t_b1[:, None, :2], t_b2[:, :2])
    br = torch.min(t_b1[:, None, 2:], t_b2[:, 2:])

    area_b = torch.prod(t_b2[:, 2:] - t_b2[:, :2], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    overlap = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    ious = overlap / area_b
    res = torch.max(ious,dim =0)[0] > th
    
    return res.all()
def run():
    for each_f in tqdm(file_list):


        image = cv2.imread(each_f)

        
        try:
            res_dict = request_ocr_service(image)
        except:
            continue

        for region in res_dict["regions"]:
            for line in region["lines"]:
                if(len(line['text']) < 100):
                    if(freq.get(line['text'])==None):
                        text_bbox[line['text']] = set()
                        freq[line['text']] = 0
                    text_bbox[line['text']].add(line['boundingBox'])
                    freq[line['text']]+=1
        
    del_fs = []
    for key,val in freq.items():
        if(val < 5):
            del_fs.append(key)
    for d in del_fs:
        freq.pop(d)
        text_bbox.pop(d)
    new_freq =  sorted(freq.items(), key=lambda d: d[1],reverse=True)
    new_bboxs = []
    for key,_ in new_freq:
        new_bboxs.append([key,list(text_bbox[key])])
    print(new_bboxs)
    json.dump(new_freq,open("freq.json",'w'),indent=6)
    json.dump(new_bboxs,open("text_box.json",'w'),indent=6)
    for key,val in freq.items():
        print(key,val)
def processJson(json_path):
    data = json.load(open(json_path))
    count = 0
    for _,val in data.items():
        count += val
    new_data =  sorted(data.items(), key=lambda d: d[1],reverse=True)
    print(new_data)
    json.dump(new_data,open("sort.json",'w'),indent=6)
# img = cv2.imread("/ssd8/other/zhaojiayi/mydata712/train2017/facbb193b0b23fddd729ca842e491af9_013.jpg")
# print(request_ocr_service(img))
# run()
getPos(["ELSEVIER","CrossMark","MDPI","Taylor & Francis","Taylor & Francis Group","ORIGINAL RESEARCH","BMC"\
    "frontiers", "Routledge","NIH-PA Author Manuscript","HHS Public Access"])
def mutiprocess(file_path):
    files = set(os.listdir(file_path))
    all_files = []
    for i in files:
        if(i[-1] == 'g'):
            all_files.append(i)
   
    c = 20
 
    mypool = Pool(c)
    start = 0
    count = len(all_files)
    results = []
    
    for i in range(0,c):
        
        if(i == c-1):
           result = mypool.apply_async(findF,(all_files[int((i)/c*count):],file_path,))
        else:
           result =  mypool.apply_async(findF,(all_files[int((i)/c*count):int((i+1)/c*count)],file_path,))
        print(int((i)/c*count),int((i+1)/c*count))
        results.append(result)

    mypool.close()
    mypool.join()
    final_res = []
    for res in results:
        final_res.append(res.get())
    data = {}
    data = json.load(open("bibinfo.json",'r'))
    data['resval'] = final_res

    json.dump(data,open("bibinfo.json",'w'),indent=6)
    pickle.dump(final_res,open("bibinfo.pkl",'wb'))
mutiprocess("/ssd8/other/zhaojiayi/mydata712/val2017")
# data = json.load(open("/ssd8/other/zhaojiayi/code/res.json"))
# img_f = []
# json_f = []
# for f in data['res']:
#     if(len(f)):
#         for ff in f:
#             json_f.append(ff)
#             img_f.append(ff[:-4]+"jpg")

# data = json.load(open("/ssd8/other/zhaojiayi/code/resval.json"))
# for f in data['res']:
#     if(len(f)):
#         for ff in f:
#             json_f.append(ff)
#             img_f.append(ff[:-4]+"jpg")
# for f in img_f:
#     os.system("cp {} /ssd8/other/zhaojiayi/problem/{} ".format(f,f.split('/')[-1]))
# for f in json_f:
#     os.system("cp {} /ssd8/other/zhaojiayi/problem/{} ".format(f,f.split('/')[-1]))