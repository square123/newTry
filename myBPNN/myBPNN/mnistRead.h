#ifndef MNISTRESD_H
#define MNISTRESD_H
#include<vector>
#include<cmath>
#include <string>
using namespace std;
//��ȡ�ļ���ͷ�ļ�
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

class Mnist //�ο������ϵģ�������һЩ
{
public:
	// �����ݶ�ȡ��ѵ���ֿ�									
	vector<vector<int>>	PixelOfImages;			// 60000��ѵ��ͼƬ��784�����ص������ֵ
	vector<int>						LabelOfImages;			// 60000��ѵ��ͼƬ��784�����ص�Ķ�Ӧ�ı�ǩֵ
	bool readData(const string &dataString,const string &labelString);


private:
	//��ȡ�ļ���
	MnistImageFileHeader	PixelsFile_Header;
	MnistLabelFileHeader	LabelsFile_Header;
	//����ͼ�����Ӧ������Ҫ��ֵ���ģ�������������㣬Ҳ����һ��ȥ�봦��
	int ImageBinary(int PixelVal);
	int ReverseInt(int i);
	unsigned char					pixelVal;	//�൱��һ���м����					
	unsigned char					labelVal;   //
};

#endif