#include "mnistRead.h"
#include <fstream>
#include <iostream>



bool Mnist::readData(const string &dataString,const string &labelString)
{
	ifstream ImagePixelsFile(dataString, ios::binary);
	ifstream ImageLabelsFile(labelString, ios::binary);

	if (ImagePixelsFile.is_open() && ImageLabelsFile.is_open())
	{
		// 读取头部无效数据
		const int HEAD_BYTE_NUM = 4;			// 头部无效数据都是以4字节为单位
		//data
		ImagePixelsFile.read((char*)&PixelsFile_Header.MagicNumber, HEAD_BYTE_NUM);
		ImagePixelsFile.read((char*)&PixelsFile_Header.NumberOfImages, HEAD_BYTE_NUM);//
		ImagePixelsFile.read((char*)&PixelsFile_Header.NumberOfRows, HEAD_BYTE_NUM);//28
		ImagePixelsFile.read((char*)&PixelsFile_Header.NumberOfColums, HEAD_BYTE_NUM);//28
		//label
		ImageLabelsFile.read((char*)&LabelsFile_Header.MagicNumber, HEAD_BYTE_NUM);
		ImageLabelsFile.read((char*)&LabelsFile_Header.NumberOfLabels, HEAD_BYTE_NUM);

		// 大小端转换
		//data
		PixelsFile_Header.MagicNumber = ReverseInt(PixelsFile_Header.MagicNumber);
		PixelsFile_Header.NumberOfImages = ReverseInt(PixelsFile_Header.NumberOfImages);
		PixelsFile_Header.NumberOfRows = ReverseInt(PixelsFile_Header.NumberOfRows);
		PixelsFile_Header.NumberOfColums = ReverseInt(PixelsFile_Header.NumberOfColums);
		//label
		LabelsFile_Header.MagicNumber = ReverseInt(LabelsFile_Header.MagicNumber);
		LabelsFile_Header.NumberOfLabels = ReverseInt(LabelsFile_Header.NumberOfLabels);

		// 判断MagicNumber是否正确
		if (PixelsFile_Header.MagicNumber != 2051 || LabelsFile_Header.MagicNumber != 2049)
			return false;

		// 读取有效数据
		// 中部有效数据都是以1字节为单位: const int DATA_BYTE_NUM = sizeof(pixelVal) = sizeof(labelVal);
		// 注意：只有文件头的个别数字需要大小端转换，其余的60000个有效数据则不需要。

		// 读像素点
		for (int imageIdx = 0; imageIdx < PixelsFile_Header.NumberOfImages; ++imageIdx)			// 60000次，即TrainPixelsFile_Header.NumberOfImages = 60000
		{
			vector<int> pixelValsOfOneImage;
			for (int pixelIdx = 0; pixelIdx < PixelsFile_Header.NumberOfRows * PixelsFile_Header.NumberOfColums; ++pixelIdx)	// 28*28次
			{
				ImagePixelsFile.read((char*)&pixelVal, sizeof(pixelVal));
				pixelValsOfOneImage.push_back(ImageBinary(pixelVal));// 二值化
			}
			PixelOfImages.push_back(pixelValsOfOneImage);
			//显示进度
			if (imageIdx%5000==0)
			{
				cout<<"已读取了Data "<<imageIdx<<"个数据"<<endl;
			}
		}

		// 读标签值
		for (int imageIdx = 0; imageIdx < LabelsFile_Header.NumberOfLabels; ++imageIdx)			// 60000次，即TrainPixelsFile_Header.NumberOfImages = 60000
		{
			ImageLabelsFile.read((char*)&labelVal, sizeof(labelVal));
			LabelOfImages.push_back(int(labelVal));
			//显示进度
			if (imageIdx%5000==0)
			{
				cout<<"已读取了Label "<<imageIdx<<"个数据"<<endl;
			}
		}
		ImagePixelsFile.close();
		ImageLabelsFile.close();
	}

	if (PixelOfImages.size() != PixelsFile_Header.NumberOfImages && LabelOfImages.size() !=  LabelsFile_Header.NumberOfLabels)
		return false;

	return true;
}

int Mnist::ImageBinary(int PixelVal)
{
	 return PixelVal<128? 0:1;
}

int Mnist::ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}