#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include<io.h>   
#include <vector>
#include "bp.h"

using namespace cv;
using namespace std;

int main()
{
	bp testBp;
	testBp.loadModel("100-9735-50.txt");
	string pathName="..//testt//*.*";  //输入要检查的路径
	vector<string> temp;
	_finddata_t file;  
	long lf;  
	//输入文件夹路径   
	if((lf = _findfirst(pathName.c_str(), &file))==-1)    //输入的文件必须是图片，不然会出错，这里为了简便就不判断了
		cout<<"Not Found!"<<endl;  
	else{  
		while(_findnext( lf, &file)==0){  
			temp.push_back(file.name);   
		}  
	}  
	_findclose(lf);  
	for (auto &i:temp)
	{
		if (i.size()>3) //必须大于bmp
		{
			if(i.substr(i.size()-3,3)=="bmp"||i.substr(i.size()-3,3)=="BMP"||i.substr(i.size()-3,3)=="jpg"||i.substr(i.size()-3,3)=="JPG"||i.substr(i.size()-3,3)=="PNG"||i.substr(i.size()-3,3)=="png")
			{
				string tempPath=pathName.substr(0,pathName.size()-3);
				tempPath=tempPath+i;
				Mat pic=imread(tempPath,0);//先转换成灰度图
				Mat pic28(28,28,CV_8UC1);
				resize(pic,pic28,Size(28,28));//变换成固定大小
				int tempArray[28*28];
				for (int i=0;i<28;++i)
				{
					for (int j=0;j<28;++j)
					{
						tempArray[i*28+j]=pic28.at<uchar>(i,j)>128?1:0;//二值化	
					}
				}
				//开始测试：
				memcpy(testBp.input,tempArray,sizeof(testBp.input));
				cout<<i<<" is aimed to the recognizeed result "<<testBp.testing()<<endl;	
			}
		}
	}
	waitKey(0);
	system("pause");
	return 0;  
}
