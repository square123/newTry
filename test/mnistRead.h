#ifndef MNISTRESD_H
#define MNISTRESD_H
#include<vector>
#include<cmath>
#include <string>
#include <cstring>
using namespace std;
//读取文件的头文件
struct MnistImageFileHeader
{
	int MagicNumber;
	int NumberOfImages;
	int NumberOfRows;
	int NumberOfColums;
};
struct MnistLabelFileHeader
{
	int MagicNumber;
	int NumberOfLabels;
};

class Mnist 
{
public:
	// 将数据读取和训练分开									
	vector<vector<int> >	PixelOfImages;			// 图片的像素值
	vector<int>	LabelOfImages;			// 图片的标签值
	bool readData(const string &dataString,const string &labelString);


private:
	//读取文件用
	MnistImageFileHeader	PixelsFile_Header;
	MnistLabelFileHeader	LabelsFile_Header;
	//分析图像觉得应该是需要二值化的，这样计算更方便，也算是一种去噪处理
	int ImageBinary(int PixelVal);
	int ReverseInt(int i);
	unsigned char					pixelVal;	//相当于一个中间变量					
	unsigned char					labelVal;   //
};

#endif