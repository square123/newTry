#ifndef MYBP_H
#define MYBP_H

#include<vector>
#include<cmath>
#include<cstdlib> //���������
#include<fstream>
#include<ctime>
#include <iostream>
#include <algorithm> //����˳��

#define INPUT_LAYER_NODE_CNTS		784				// �����ڵ����
#define HIDDEN_LAYER_NODE_CNTS		64				//�����м��Ľ����Ϊ64 
#define OUTPUT_LAYER_NODE_CNTS		10				//�����
#define LEARNING_RATE				0.35			// ѧϰ����		 ��Ҫ����Ӧ�ñ�0.1ҪС

#define TRAIN 0							//ѵ���ı�ǩ
#define TEST 1							//���Եı�ǩ

#define TRAIN_NUM 5					//��ʱҪѵ���Ĵ���


//����û�в������򻯵ķ��������ѵ���ܿ��ܻ����ϣ����Բ���һ��һ��ѵ�����������������ϣ��������ط������ģ���1/2ѵ���ϰ�㣬1/2ѵ���°�㣬���ȡƽ��
using namespace std;
class myBP 
{
public:

	//��ȡ�ļ��е�·����Ȼ����·����ͼƬȫ������ Ӧ���д�С���ж� �����ͼƬ�ͺ�����
	void useBpNN(vector<vector<int>> &,vector<int> &,int);//ѵ�����̺Ͳ��Թ���Ӧ����һ������ ֻ�����벻һ�� Ӧ���и������ж�
	void SaveTrainedModel(const string &dstFilePath);			// ������ѵ���õ�ģ�� ����������Լ�ȥѡ��
	void LoadTrainedModel(const string &srcFilePath);			// ������ѵ���õ�ģ�� ����������Լ�ȥѡ��
	myBP();

private:
	//��Ȩֵ����Ϊ����
	double weight1To2Layer[INPUT_LAYER_NODE_CNTS][HIDDEN_LAYER_NODE_CNTS]; //��һ�����Ȩֵ
	double weight2To3Layer[HIDDEN_LAYER_NODE_CNTS][OUTPUT_LAYER_NODE_CNTS]; //�ڶ������Ȩֵ
	double bias1To2Layer[HIDDEN_LAYER_NODE_CNTS]; //��һ���ƫ�������е�Ȩֵ ��סƫ�õĵ�Ԫ������1
	double bias2To3Layer[OUTPUT_LAYER_NODE_CNTS]; //�ڶ����ƫ�������е�Ȩֵ
	//���
	double realOutput2Layer[HIDDEN_LAYER_NODE_CNTS];
	double realOutput3Layer[OUTPUT_LAYER_NODE_CNTS];
	//���
	double delta2Layer[HIDDEN_LAYER_NODE_CNTS];
	double delta3Layer[OUTPUT_LAYER_NODE_CNTS];
	//С����
	double sigmoid(double val);
	double randInt();
	//ѵ��
	void Initialize();					//������������ʼ��
	void forward(vector<vector<int>> &dataVec,int index);
	void backward(vector<vector<int>> &dataVec,vector<int> & labelVec,int index);
	//����
	int testForOne(vector<vector<int>> &dataVec,int index); //�����������
};


#endif