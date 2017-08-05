#ifndef BP_H
#define BP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <time.h>


//Ϊ�˷��㽫��д��һ���ļ���

#define inputNum 784
#define hidderNum  100
#define outputNum  10
#define learnRate  0.3
using namespace std;
class bp
{
public:
	//���������еĲ�����ֻ���һ�����ݵ�
	void training(); // ѵ����ʱ����Ҫread���ԾͲ����ˡ�
	int testing();	//����Ҫ���غ��������ǰ������Ĳ�����Ӧ�ý����Ĳ�������
	void saveModel(const string &dstFilePath);
	void loadModel(const string &dstFilePath);
	void initialize();
	int input[inputNum];		//���ֵ��ÿ�ζ��任��				
	int target[outputNum];
private:
	//Ȩֵ �����ݲ��֣�
	double weight1To2[inputNum][hidderNum];			
	double weight2To3[hidderNum][outputNum];				
	double bias1[hidderNum];						
	double bias2[outputNum];	
	//�������
	double outputLayer2[hidderNum];				
	double outputLayer3[outputNum];					
	double deltaLayer2[hidderNum];		
	double deltaLayer3[outputNum];	
	//С����
	double sigmoid(double x);
	double randInt();
	//������
	void forward1To2();
	void forward2To3();
	void backward3To2();
	void backward2To1();
	void renew1To2();
	void renew2To3();

};

#endif