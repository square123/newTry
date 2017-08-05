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
	string pathName="..//testt//*.*";  //����Ҫ����·��
	vector<string> temp;
	_finddata_t file;  
	long lf;  
	//�����ļ���·��   
	if((lf = _findfirst(pathName.c_str(), &file))==-1)    //������ļ�������ͼƬ����Ȼ���������Ϊ�˼��Ͳ��ж���
		cout<<"Not Found!"<<endl;  
	else{  
		while(_findnext( lf, &file)==0){  
			temp.push_back(file.name);   
		}  
	}  
	_findclose(lf);  
	for (auto &i:temp)
	{
		if (i.size()>3) //�������bmp
		{
			if(i.substr(i.size()-3,3)=="bmp"||i.substr(i.size()-3,3)=="BMP"||i.substr(i.size()-3,3)=="jpg"||i.substr(i.size()-3,3)=="JPG"||i.substr(i.size()-3,3)=="PNG"||i.substr(i.size()-3,3)=="png")
			{
				string tempPath=pathName.substr(0,pathName.size()-3);
				tempPath=tempPath+i;
				Mat pic=imread(tempPath,0);//��ת���ɻҶ�ͼ
				Mat pic28(28,28,CV_8UC1);
				resize(pic,pic28,Size(28,28));//�任�ɹ̶���С
				int tempArray[28*28];
				for (int i=0;i<28;++i)
				{
					for (int j=0;j<28;++j)
					{
						tempArray[i*28+j]=pic28.at<uchar>(i,j)>128?1:0;//��ֵ��	
					}
				}
				//��ʼ���ԣ�
				memcpy(testBp.input,tempArray,sizeof(testBp.input));
				cout<<i<<" is aimed to the recognizeed result "<<testBp.testing()<<endl;	
			}
		}
	}
	waitKey(0);
	system("pause");
	return 0;  
}
