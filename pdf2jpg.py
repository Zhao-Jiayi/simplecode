
import json
from multiprocessing import Pool
from selectors import PollSelector
import sys, fitz, os, datetime
from tqdm import tqdm
data = ["报告",'电子书','方案','教辅资料&课程资料','论文',"财务报表","法律&规定",\
"合同","介绍&讲解","应聘","邮件","表格、单据","不完整论文","不完整文章",\
"测试、开发与实践记录","产品、服务和流程规范","产品参数标准","产品广告与介绍","会议记录",\
"建议信","申请书","声明书","说明书","医嘱","娱乐","证明","指南","专利单","offer","教学","作业"]
def pyMuPDF_fitz(files, imagePath,s,e):

    for idx in tqdm(range(s,e)):
        pdfPath = files[idx]

        pdfDoc = fitz.open(pdfPath)
        cur_path = ""
        for each in data:
            if(pdfPath.find(each) > 0):
                cur_path = imagePath + '/' + each
        if cur_path == "":
            cur_path = imagePath + "/" + "其他"
        
        if(os.path.isdir(cur_path) == 0):
          
            os.mkdir(cur_path)
        for pg in range(pdfDoc.page_count):

            page = pdfDoc[pg]
            rotate = int(0)
            # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
            # 此处若是不做设置，默认图片大小为：792X612, dpi=96
            zoom_x = 2.3 #(1.33333333-->1056x816)   (2-->1584x1224)
            zoom_y = 2.3
            
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            
            pix = page.get_pixmap(matrix=mat, alpha=False)

            pix._writeIMG(cur_path+'/'+'%s_%s.jpg' % (idx,pg),format = 1)#将图片写入指定的文件夹内

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
   
            if(last_name == "pdf"):

                all_useful_files.append(cur_path)


def run(out,p = 10):
    mypool = Pool(10)
    relation = {}
    files = []
    findALLUseful("/ssd8/other/liyx/2.已完成分类-全部数据",files)
    i = 0
    for f in files:
        i += 1
        relation[f] = i
    json.dump(relation,open('./relation.json','w'))
    l = len(files)

    for i in range(p):
        print(int((i)/p*l),int((i+1)/p*l))
        if i == p-1:
            mypool.apply_async(pyMuPDF_fitz,(files,out,int((i)/p*l),l,))
        else:
            mypool.apply_async(pyMuPDF_fitz,(files,out,int((i)/p*l),int((i+1)/p*l),))
    mypool.close()
    mypool.join()
run("/ssd8/other/zhaojiayi/class811")