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


//为了方便将类写在一个文件下

#define inputNum 784
#define hidderNum  100
#define outputNum  10
#define learnRate  0.3
using namespace std;
class bp
{
public:
	//这里面所有的操作都只针对一条数据的
	void training(); // 训练的时候需要read测试就不用了。
	int testing();	//必须要加载函数，但是包含读的操作，应该将读的操作分离
	void saveModel(const string &dstFilePath);
	void loadModel(const string &dstFilePath);
	void initialize();
	int input[inputNum];		//这个值是每次都变换的				
	int target[outputNum];
private:
	//权值 （数据部分）
	double weight1To2[inputNum][hidderNum];			
	double weight2To3[hidderNum][outputNum];				
	double bias1[hidderNum];						
	double bias2[outputNum];	
	//输入输出
	double outputLayer2[hidderNum];				
	double outputLayer3[outputNum];					
	double deltaLayer2[hidderNum];		
	double deltaLayer3[outputNum];	
	//小函数
	double sigmoid(double x);
	double randInt();
	//处理函数
	void forward1To2();
	void forward2To3();
	void backward3To2();
	void backward2To1();
	void renew1To2();
	void renew2To3();

};

#endif