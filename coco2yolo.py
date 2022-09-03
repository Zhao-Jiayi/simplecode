
import os 
import json
from tqdm import tqdm
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='/ssd8/other/zhaojiayi/mydata712/annotations/instances_train_2017.json',type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='/ssd8/other/zhaojiayi/mydata712/labels/train2017', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
 
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def generateLabels():
    json_file =   arg.json_path # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径
 
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    
    id_map = {} # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']): 
            f.write(f"{category['name']}\n") 
            id_map[category['id']] = i
    # print(id_map)
 
    anns = {}
    for ann in data['annotations']:
        imgid = ann['image_id']
        anns.setdefault(imgid, []).append(ann)
 
    print('got anns')
 
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
 
        ann_img = anns.get(img_id, [])
        for ann in ann_img:
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
def generateImages(imagepath,outfilename):
    files = os.listdir(imagepath)
    with open(outfilename,'w') as f:
        for file in tqdm(files):
            if(file[-1] == 'g'):
                f.write("{}/{}\n".format(imagepath,file))

if __name__ == '__main__':
    generateLabels()
