#include <iostream>
#include "mnistRead.h"
#include "myBP.h"
/**************************************************
 * \file main.cpp
 * \date 2017/08/03 15:53
 * \author ���Ǹ���
 * Contact: kfhao123@163.com
 * \note �ó������NN��ѵ�����̣�
 *1.��ȡ�ļ��Ĺ���
 *2.��ʼ������Ĺ��̣�Ĭ���������������磩
 *3.����ģ�͵Ĺ���
 *4.����������ݵĹ���
**************************************************/

//�����ļ���·��
#define trainDataPath "F:\\nn\\train-images.idx3-ubyte"
#define trainLabelPath "F:\\nn\\train-labels.idx1-ubyte"
#define testDataPath "F:\\nn\\t10k-images.idx3-ubyte"
#define testLabelPath "F:\\nn\\t10k-labels.idx1-ubyte"

int main()
{

	//train
	//Mnist trainData1;
	//trainData1.readData( trainDataPath, trainLabelPath);	//��ȡ�ļ�	
	//myBP bp1;
	//bp1.useBpNN(trainData1.PixelOfImages,trainData1.LabelOfImages,TRAIN);
	//bp1.SaveTrainedModel("new.txt");
	//test
	Mnist trainData2;
	trainData2.readData( testDataPath, testLabelPath);	//��ȡ�ļ�	
	myBP bp2;
	bp2.LoadTrainedModel("model_h64_t5_l35.txt");
	bp2.useBpNN(trainData2.PixelOfImages,trainData2.LabelOfImages,TEST);

	system("pause");

}