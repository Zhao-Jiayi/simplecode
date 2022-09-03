#include <functional>
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <string.h>
#include <dirent.h>
#include <unordered_map>

/**
 * @Process 用于模型预测的bboxes进行后处理
 * 主要通过对图片进行二值化，然后再x,y方向进行投影，根据投影值确定bbox的真实边界
 */
template<typename T>
class Process{
	public:
		typedef  std::map<T, std::pair<T, T> > mdp;
		int findPos(const std::vector<int>& proj, int start, int dir, bool is_x) const;
		typename mdp::const_iterator findBound(const  mdp& bbox_sort,typename mdp::const_iterator start_iter,typename mdp::const_iterator cur_iter, bool is_x, int direction,int len) const;
		std::vector<std::vector<int>> adjustBound(const std::string& filepath, const std::vector<std::vector<T>>& bboxes) ;
		inline void setShift(int x1 = 5,int y1 = 10,int x2 = 2,int y2 = 3,int dis = 0,int threshold = 232,int x_overlap = 0.3,int y_overlap = 0.3);
		inline const std::vector<std::vector<int>>& getRes() const;
		inline const std::unordered_map<std::string,std::vector<std::vector<int>>>& getReses() const;
		Process() = default;
		Process(const std::string&  filepath, std::vector<std::vector<T>>& bboxes);
		Process(const std::unordered_map<std::string,std::vector<std::vector<T>>>&m);
		inline void clearRes();
		inline void setRes(const std::unordered_map<std::string,std::vector<std::vector<int>>>&m);
		inline void setRes(const std::vector<std::vector<int>>&v);
		inline const std::string& getLastFile();
	private:
		int proj_shift_x;
		int proj_shift_y;
		int max_dis;
		int continue_same_pixel_x;
		int continue_same_pixel_y;
		int binary_threshold;
		int x_overlap;
		int y_overlap;
		std::string cur_file_name;
		std::vector<std::vector<int>> res;
		std::unordered_map<std::string,std::vector<std::vector<int>>> reses;
};
/*
* @Process::clearRes 清空当前保存的结果
*/
template<typename T>
inline void Process<T>::clearRes(){
	res.clear();
	reses.clear();
}
/*
* @Process::setRes 设置一组图片及对应的调整后的bboxes
* @pram m: 需要保存的一组图片以及bboxes
*/
template<typename T>
inline void Process<T>::setRes(const std::unordered_map<std::string,std::vector<std::vector<int>>>&m){
	reses.clear();
	reses = m;
}
/*
* @Process::setRes 设置一张图片调整后的bboxes
* @pram v: 需要保存的一张图片的bboxes
*/
template<typename T>
inline void Process<T>::setRes(const std::vector<std::vector<int>>& v){
	res.clear();
	res = v;
}
/*
* @Process::getLastFile 获取最后一张图片的路径
* @return : 最后一张图片的路径
*/
template<typename T>
inline const std::string& Process<T>::getLastFile(){return cur_file_name;}
/*
* @Process::setShift 设置投影时扩张范围以及遇到多少连续相同像素停止
* @parm x1 : 投影时x方向扩张范围
* @parm y1 : 投影时y方向扩张范围
* @parm x2 : x方向遇到多少相同像素停止
* @parm y2 : y方向遇到多少相同像素停止
* @parm dis :对调整后边界的四个方向分别做max_dis扩张，防止由于二值化时导致将前景判断为背景导致的误差
* @parm x_over : x方向上重叠大于该阈值作为限制
* @parm y_over : y方向上重叠大于该阈值作为限制
* @parm threshold : 二值化阈值
*/
template<typename T>
inline void Process<T>::setShift(int x1 ,int y1 ,int x2 ,int y2 ,int dis ,int threshold,int x_over,int y_over){
	proj_shift_x = x1;
	proj_shift_y = y1;
	continue_same_pixel_x = x2;
	continue_same_pixel_y = y2;
	max_dis = dis;
	binary_threshold = threshold;
	x_overlap = x_over;
	y_overlap = y_over;
}
/*
* @Process::Process 构造函数，自动对一张图片的bbox进行调整
* @parm filepath: 图片路径
* @parm bboxes: 调整前边界
*/
template<typename T>
Process<T>::Process(const std::string&  filepath, std::vector<std::vector<T>>& bboxes){
	setShift();
	res = adjustBound(filepath,bboxes);
}
/*
* @Process::Process 构造函数，自动对一组图片的bbox进行调整
* @parm m: 图片路径以及bbox对
*/
template<typename T>
Process<T>::Process(const std::unordered_map<std::string,std::vector<std::vector<T>>>& m){
	setShift();
	for(const std::pair<std::string,std::vector<std::vector<T>>>& p : m){
		reses[p.first] = std::move(adjustBound(p.first,p.second));
	}
}
/*
* @Process::getReses 返回储存的一组图片-bbox对
* @return : 图片路径以及bbox对
*/
template<typename T>
inline const std::unordered_map<std::string,std::vector<std::vector<int>>>& Process<T>::getReses() const {return reses;}
/*
* @Process::getRes 返回最后一次调整的的bbox
* @return : 最后一次调整的的bbox
*/
template<typename T>
inline const std::vector<std::vector<int>>& Process<T>::getRes() const {return res;}
/*
* @Process::findPos 从start位置找到出现连续相同像素的位置
* @parm proj : x或y方向投影
* @parm start : 开始位置在proj中的索引
* @parm add : 表示扩张的方向
* @parm is_x : 要调整的时x坐标还是y坐标
* @return : 出现连续相同像素的位置
*/
template<typename T>
int Process<T>::findPos(const std::vector<int>& proj, int start, int add, bool is_x) const {
	//确定当前方向上需要的连续的相同像素个数
	int init_shift = is_x ? continue_same_pixel_x : continue_same_pixel_y;
	int shift_thresh = init_shift;
	//首先进行边界扩张，如果出现连续init_shift个0（白色）像素，则start-init_shift则为扩张后的边界
	while (start > 0 and start < proj.size()) {
		if (proj[start] == 0) {
			shift_thresh--;
			if (!shift_thresh) {
				break;
			}
		}
		else {
			//出现了黑色重新计算
			shift_thresh = init_shift;	
			
		}
		start += add;
	}
	//边界收缩，如果出现连续init_shift个1（黑色）像素，则start-init_shift则为缩小后的边界
	shift_thresh = init_shift;
	while (start > 0 and start < proj.size()) {
		if (proj[start] == 1) {
			shift_thresh--;
			if (!shift_thresh) {
				return start + add * init_shift;
			}
		}
		else {
			//出现白色重新计算
			shift_thresh = init_shift;
		}
		start -= add;
	}
	return start;
}
/*
* @Process::findBound 找到第一个可以限制调整前坐标的坐标
* @parm bbox_sort : 以x或y坐标为key,另外一个方向的两个位置为value的map
* @parm start_iter : 调整前坐标对应的迭代器
* @parm cur_iter : 当前迭代器
* @parm is_x : 要调整的时x坐标还是y坐标
* @parm direction : 迭代器前进方向
* @return : 第一个可作为限制的迭代器
*/
template<typename T> 
typename Process<T>::mdp::const_iterator Process<T>::findBound(const  mdp& bbox_sort, typename mdp::const_iterator start_iter, typename mdp::const_iterator cur_iter, bool is_x, int direction,int len)const {
	//根据方向确定
	double limit_shift = is_x ? x_overlap : y_overlap;
	while (cur_iter != bbox_sort.end()) {
		//当当前方向为x时，若该bbox的y坐标与start_iter的y坐标有一定重叠
		int overlap_point1 = std::min(cur_iter->second.first,start_iter->second.first);
		int overlap_point2 = std::max(cur_iter->second.second,start_iter->second.second);
		if (overlap_point2 > overlap_point1 && (double)(overlap_point2-overlap_point1)/len > limit_shift) {
			return cur_iter;
		}
		//到了begin还没找到，说明该坐标不受限制，防止越界直接退出
		if (cur_iter == bbox_sort.begin()) {
			break;
		}
		std::advance(cur_iter, direction);
	}
	return bbox_sort.end();
}
/*
* @Process::adjustBount 根据投影信息调整预测框的边界
* @parm filepath : 图片路径
* @parm bboxes : 调整前bboxes
* @return : 调整后bboxes
*/
template<typename T>
std::vector<std::vector<int>> Process<T>::adjustBound(const std::string& filepath, const std::vector<std::vector<T>>& bboxes ){
	//RGB转灰度图
	cv::Mat img = cv::imread(filepath,1);
	cur_file_name = filepath;
	int img_h = img.rows;
	int img_w = img.cols;
	int n = bboxes.size();
	cv::Mat gray_img(img_h, img_w, CV_8UC1);
	cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
	cv::Mat binary_img(img_h, img_w, CV_8UC1);
	cv::threshold(gray_img, binary_img, 230, 255, cv::THRESH_BINARY);
	//二值化，黑色为1，白色为0
	for (int row = 0; row < img_h; row++) {
		unsigned char* data = binary_img.ptr<unsigned char>(row);
		for (int col = 0; col < img_w; col++) {
			data[col] = data[col] == 0 ? 1 : 0;
		}
	}
	//x，y方向上所有的坐标以及另外方向的两个坐标
	mdp x_sort;
	mdp y_sort;
	for (const std::vector<T>& b : bboxes) {
		x_sort[b[0]] = std::make_pair(b[1], b[3]);
		x_sort[b[2]] = std::make_pair(b[1], b[3]);
		y_sort[b[1]] = std::make_pair(b[0], b[2]);
		y_sort[b[3]] = std::make_pair(b[0], b[2]);
	}
	//结果
	std::vector<std::vector<int>> new_bboxes;
	for (const std::vector<T>& b : bboxes) {
		int hs, he, ws, we;
		//其他类型转int
		std::vector<int> d2i(b.begin(),b.end());
		new_bboxes.emplace_back(std::move(d2i));
		std::vector<int>& b_int = new_bboxes.back();
		//投影区域范围
		hs = std::max(0,  b_int[1]- proj_shift_x - 1);
		he = std::min(b_int[3]+ proj_shift_x - 1,  img_h- 1);
		ws = std::max(0, b_int[0] -proj_shift_y - 1);
		we = std::min(b_int[2] + proj_shift_y - 1, img_w - 1);
	
		//进行投影，若为1则该方向上有前景，直接返回
		std::vector<int> ww(we - ws + 1,0);
		std::vector<int> hh(he - hs + 1,0);
		
	
		
		//lambad 表达式，用于对边界进行限制，逻辑较简单，但重复度高，下面以对左上角x坐标（dir=-1,is_x=1）做限制为例子
		std::function<void(const  mdp& ,int ,int ,int ,int)> limitBound = [&](const mdp& m ,int val,int is_x ,int dir,int idx)->void{
			typename mdp::const_iterator cur_iter = m.begin();
			int len = is_x ? we - ws : he - hs;
			if(dir == -1){
				//利用map有序性找到val所在迭代器
				cur_iter = m.lower_bound(val);
				//其上方无其他检测框，直接返回
				if(cur_iter == m.begin()){
					return;
				}
				//查找第一个可作为其上边界限制的迭代器
				cur_iter = findBound(m,cur_iter,--cur_iter,is_x,dir,len);
				//没有找到限制
				if(cur_iter == m.end()){
					//进行适当上移
					b_int[idx] += dir * max_dis;
					return;
				}
				else{
					//如果当前x小于找到的限制，则应将其限制在其之下
					if(b_int[idx] < static_cast<int>(cur_iter->first)){
						b_int[idx] = cur_iter->first;
						b_int[idx] -= dir * max_dis;
					}
					else{
						//适当上移
						b_int[idx] += dir * max_dis;
					}

				}
			}
			else{
				cur_iter = m.upper_bound(val);
				if(cur_iter == m.end()){
					return;
				}
				cur_iter = findBound(m,cur_iter,cur_iter,is_x,dir,len);
				if(cur_iter == m.end()){
					b_int[idx] += dir * max_dis;
					return;
				}
				else{
					if(b_int[idx] > static_cast<int>(cur_iter->first)){
						b_int[idx] = cur_iter->first;
						b_int[idx] -= dir * max_dis;
					}
					else{
						b_int[idx] += dir * max_dis;
					}

				}
			}

		};
		//调整每个坐标
		std::function<void(bool)> adjustEachPoint = [&](bool is_x)->void{
			int xl = !is_x ? b_int[0] : ws;
			int w = !is_x ? b_int[2] - b_int[0]  : we - ws ;
			int yl = is_x ? b_int[1] : hs;
			int h = is_x ? b_int[3] - b_int[1] : he - hs ;
			cv::Mat select_roi = binary_img(cv::Rect(xl, yl,w, h));
			std::vector<int> &proj = is_x ? ww : hh;
			//投影
			if(!is_x){
				for (int row = 0; row < select_roi.rows ; row++) {
					for (int col = 0; col < select_roi.cols; col++) {
						if (select_roi.at<unsigned char>(row, col) == 1) {
							proj[row] = 1;
							break;
						}
					}
				}
			}
			else{
				for (int col = 0; col < select_roi.cols ; col++) {
					for (int row = 0; row < select_roi.rows; row++) {
						if (select_roi.at<unsigned char>(row, col) == 1) {
							proj[col] = 1;
							break;
						}
					}
				}
			}
			int& xy1 = is_x ? b_int[0] : b_int[1];
			int& xy2 = is_x ? b_int[2] : b_int[3];
			int s = is_x ? ws : hs;
			int p = findPos(proj, xy1 - s, -1, is_x);
			xy1 = p + s;
			p = findPos(proj,xy2 - s, 1, is_x);
			xy2 = p + s;
			mdp& m = is_x ? x_sort : y_sort;
			int xy3 = is_x ? b[0] : b[1];
			int xy4 = is_x ? b[2] : b[3];
			int idx1 = is_x ? 0 : 1;
			int idx2 = is_x ? 2 : 3;
			//边界限制
			limitBound(m,xy3,is_x,-1,idx1);
			limitBound(m,xy4,is_x,1,idx2);
		};
		adjustEachPoint(0);
		adjustEachPoint(1);
		// for(int i = 0;i < 4;i++){
		// 	std::vector<int>& proj = (i == 0 || i == 2) ? ww : hh;
		// 	int w_or_h = (i == 0 || i == 2) ? ws : hs;
		// 	int dir = (i == 0 || i == 1) ? -1 : 1;
		// 	bool is_x = (i == 0 || i == 2) ? 1 : 0;
		// 	int p = findPos(proj, b_int[i] - w_or_h, dir, is_x);
		// 	b_int[i] = p + w_or_h;
		// 	mdp& m = (i == 0 || i == 2) ? x_sort : y_sort;
		// 	limitBound(m,b[i],is_x,dir,i);
		// }
		}
	
	//储存结果并返回
	setRes(new_bboxes);
	return new_bboxes;
}

/*
* @Process::getFiles 获取当前文件下所有.jpg文件储存到files中
* @parm path : 目录
* @parm files : 储存jpg路径
*/
int getFiles(const std::string& path, std::vector<std::string>& files){
	
	int iFileCnt = 0;
	DIR *dirptr = NULL;
	struct dirent *dirp;
 
	if((dirptr = opendir(path.c_str())) == NULL)//打开一个目录
	{
		return 0;
	}
	while ((dirp = readdir(dirptr)) != NULL)
	{

		if ((dirp->d_type == DT_REG) && 0 ==(strcmp(strchr(dirp->d_name, '.'), ".jpg")))//判断是否为文件以及文件后缀名
		{
			files.push_back(path+"/"+dirp->d_name);
		}
		iFileCnt++;
	}
	closedir(dirptr);
	
	return iFileCnt;
}
/*
* @Process::test 测试后处理效果,目录下需要有jpg以及调整前bbox的txt,txt名字与jpg相同,之后会在同一目录下生成调整后txt
*txt 内容为：
n bbox数量
x y x y
......
x y x y
* @parm filename : 目录或文件名
*/
void test(const std::string filename){
	DIR *dirptr = opendir(filename.c_str());
	if(dirptr == nullptr){ //filename 是文件名
		std::ofstream out_f;
		std::ifstream in_f;
		in_f.open(filename);	
		std::string s;
		int n;
		in_f >> n;
		std::vector<std::vector<double>> bboxes(n,std::vector<double>(4,0));
		int i = 0;
		//读取bbox并储存到bboxes中
		while(i != n){
			in_f >> bboxes[i][0] >> bboxes[i][1] >> bboxes[i][2] >> bboxes[i][3]; 
			i++;
		}
		const std::string img_path = filename.substr(0,filename.size()-3)+"jpg";
		//后处理
		Process<double> p(img_path,bboxes);
		const std::string out_path = filename.substr(0,filename.size()-4)+"adj.txt";
		out_f.open(out_path);
		out_f << n <<'\n';
		//获取后处理结果并写入out_f
		for(const std::vector<int>&v : p.getRes()){
			out_f << v[0] << " " << v[1] << " " << v[2] << " "<< v[3] << " "<<'\n';
		}
	}
	else{//是目录
		std::vector<std::string> all_files;
		getFiles(filename,all_files);//获取目录下jpg文件
		std::unordered_map<std::string,std::vector<std::vector<double>>> inputs;
		//读取每个图片的bbox
		for(std::string& f : all_files){	
			std::ifstream in_f;
			in_f.open(f.substr(0,f.size()-3)+"txt");	
			std::string s;
			int n;
			in_f >> n;
			std::vector<std::vector<double>> bbox(n,std::vector<double>(4,0));
			int i = 0;
			while(i != n){
				
				in_f >> bbox[i][0] >> bbox[i][1] >> bbox[i][2] >> bbox[i][3]; 
				i++;
			}
			inputs[f] = std::move(bbox);
		}
		//后处理
		Process<double> p(inputs);
		//获取后处理结果
		std::unordered_map<std::string,std::vector<std::vector<int>>> reses= p.getReses();
		//后处理结果写入txt
		for(auto&p : reses){
			std::ofstream out_f;
			std::string f = p.first;
			const std::string out_path = f.substr(0,f.size()-4)+"adj.txt";
			out_f.open(out_path);
			int n = p.second.size();
			out_f << n <<'\n';
			for(const std::vector<int>&v : p.second){
				out_f << v[0] << " " << v[1] << " " << v[2] << " "<< v[3] << " "<<'\n';
			}
		}
	
	}
}

int main(int argc,char* argv[]){
	test(argv[1]);
	return 0;
}