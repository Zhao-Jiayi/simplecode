import cv2
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
def findBadImage(filepath):
    file = os.listdir(filepath)
    for f in tqdm(file):
        if(f[-1] == 'n'):
            continue
        try:
            img = Image.open(os.path.join(filepath,f))
        
            if(np.max(img) ==  None):
                print(f)
        except:
            print(f)
            os.system("rm {}".format(filepath+"/"+f))
            os.system("rm {}".format(filepath+"/"+f[0:-3] + "json"))
findBadImage("/ssd8/other/zhaojiayi/mydata712/train2017")