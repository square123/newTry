#include <iostream>
#include "mnistRead.h"
#include "myBP.h"
/**************************************************
 * \file main.cpp
 * \date 2017/08/03 15:53
 * \author 正那个方
 * Contact: kfhao123@163.com
 * \note 该程序针对NN的训练过程：
 *1.读取文件的过程
 *2.初始神经网络的过程（默认设置是两层网络）
 *3.储存模型的过程
 *4.输出测试数据的过程
**************************************************/

//输入文件的路径
#define trainDataPath "F:\\nn\\train-images.idx3-ubyte"
#define trainLabelPath "F:\\nn\\train-labels.idx1-ubyte"
#define testDataPath "F:\\nn\\t10k-images.idx3-ubyte"
#define testLabelPath "F:\\nn\\t10k-labels.idx1-ubyte"

int main()
{

	//train
	//Mnist trainData1;
	//trainData1.readData( trainDataPath, trainLabelPath);	//读取文件	
	//myBP bp1;
	//bp1.useBpNN(trainData1.PixelOfImages,trainData1.LabelOfImages,TRAIN);
	//bp1.SaveTrainedModel("new.txt");
	//test
	Mnist trainData2;
	trainData2.readData( testDataPath, testLabelPath);	//读取文件	
	myBP bp2;
	bp2.LoadTrainedModel("model_h64_t5_l35.txt");
	bp2.useBpNN(trainData2.PixelOfImages,trainData2.LabelOfImages,TEST);

	system("pause");

}