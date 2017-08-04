#ifndef MYBP_H
#define MYBP_H

#include<vector>
#include<cmath>
#include<cstdlib> //产生随机数
#include<fstream>
#include<ctime>
#include <iostream>
#include <algorithm> //打乱顺序

#define INPUT_LAYER_NODE_CNTS		784				// 输入层节点个数
#define HIDDEN_LAYER_NODE_CNTS		64				//定义中间层的结点数为64 
#define OUTPUT_LAYER_NODE_CNTS		10				//输出层
#define LEARNING_RATE				0.35			// 学习速率		 不要过大，应该比0.1要小

#define TRAIN 0							//训练的标签
#define TEST 1							//测试的标签

#define TRAIN_NUM 5					//暂时要训练的次数


//由于没有采用正则化的方法，因此训练很可能会过拟合，所以采用一半一半训练神经网络来避免过拟合（从其他地方看到的），1/2训练上半层，1/2训练下半层，最后取平均
using namespace std;
class myBP 
{
public:

	//读取文件夹的路径，然后按照路径将图片全部处理 应该有大小的判断 这个读图片就很尴尬
	void useBpNN(vector<vector<int>> &,vector<int> &,int);//训练过程和测试过程应该是一个东西 只是输入不一样 应该有个条件判断
	void SaveTrainedModel(const string &dstFilePath);			// 保存已训练好的模型 这个必须是自己去选的
	void LoadTrainedModel(const string &srcFilePath);			// 加载已训练好的模型 这个必须是自己去选的
	myBP();

private:
	//将权值保存为数组
	double weight1To2Layer[INPUT_LAYER_NODE_CNTS][HIDDEN_LAYER_NODE_CNTS]; //第一二层的权值
	double weight2To3Layer[HIDDEN_LAYER_NODE_CNTS][OUTPUT_LAYER_NODE_CNTS]; //第二三层的权值
	double bias1To2Layer[HIDDEN_LAYER_NODE_CNTS]; //第一层的偏置所含有的权值 记住偏置的单元输入是1
	double bias2To3Layer[OUTPUT_LAYER_NODE_CNTS]; //第二层的偏置所含有的权值
	//输出
	double realOutput2Layer[HIDDEN_LAYER_NODE_CNTS];
	double realOutput3Layer[OUTPUT_LAYER_NODE_CNTS];
	//误差
	double delta2Layer[HIDDEN_LAYER_NODE_CNTS];
	double delta3Layer[OUTPUT_LAYER_NODE_CNTS];
	//小函数
	double sigmoid(double val);
	double randInt();
	//训练
	void Initialize();					//将定义的数组初始化
	void forward(vector<vector<int>> &dataVec,int index);
	void backward(vector<vector<int>> &dataVec,vector<int> & labelVec,int index);
	//测试
	int testForOne(vector<vector<int>> &dataVec,int index); //输出的是索引
};


#endif