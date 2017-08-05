#ifndef MNISTRESD_H
#define MNISTRESD_H
#include<vector>
#include<cmath>
#include <string>
#include <cstring>
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

class Mnist 
{
public:
	// �����ݶ�ȡ��ѵ���ֿ�									
	vector<vector<int> >	PixelOfImages;			// ͼƬ������ֵ
	vector<int>	LabelOfImages;			// ͼƬ�ı�ǩֵ
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