#include "myBP.h"

myBP::myBP()
{
	//先将存储输出和差别置零
	memset(realOutput2Layer,0,sizeof(realOutput2Layer));
	memset (realOutput3Layer,0,sizeof(realOutput3Layer));
	memset(delta2Layer,0,sizeof(delta2Layer));
	memset(delta3Layer,0,sizeof(delta3Layer));
}
inline double myBP::sigmoid(double val)
{
	return 1.0 / (1.0 + exp(-val));
}

inline double myBP::randInt()
{
	return ((rand()/(double)(RAND_MAX)*2*0.076)-0.076); //经验 sqrt(6)/sqrt(N_in+N_out) 
}

void myBP::Initialize()				//将定义的数组初始化   如果不是训练就不需要初始化了
{
	//权值随机初始化
	for (int i=0;i<INPUT_LAYER_NODE_CNTS;++i)
	{
		for (int j=0;j<HIDDEN_LAYER_NODE_CNTS;++j)
		{
			weight1To2Layer[i][j]=randInt();
		}
	}
	for (int i=0;i<HIDDEN_LAYER_NODE_CNTS;++i)
	{
		for (int j=0;j<OUTPUT_LAYER_NODE_CNTS;++j)
		{
			weight2To3Layer[i][j]=randInt();
		}
	}
	for (int i=0;i<HIDDEN_LAYER_NODE_CNTS;++i)
	{
		 bias1To2Layer[i]=randInt();
	}
	for (int i=0;i<OUTPUT_LAYER_NODE_CNTS;++i)
	{
		bias2To3Layer[i]=randInt();
	}
}

void myBP::SaveTrainedModel(const string &dstFilePath)			// 保存已训练好的模型  
{
	ofstream outfile(dstFilePath,ios::out);
	if(!outfile)//文件未打开不保存
	{
		cout<<"save Error"<<endl;
		outfile.close();
	}else //将权值保存为txt格式
	{
		for (int i=0;i<INPUT_LAYER_NODE_CNTS;++i)
		{
			for (int j=0;j<HIDDEN_LAYER_NODE_CNTS;++j)
			{
				outfile<<weight1To2Layer[i][j]<<endl;
			}
		}
		for (int i=0;i<HIDDEN_LAYER_NODE_CNTS;++i)
		{
			for (int j=0;j<OUTPUT_LAYER_NODE_CNTS;++j)
			{
				outfile<<weight2To3Layer[i][j]<<endl;
			}
		}
		for (int i=0;i<HIDDEN_LAYER_NODE_CNTS;++i)
		{
			outfile<<bias1To2Layer[i]<<endl;
		}
		for (int i=0;i<OUTPUT_LAYER_NODE_CNTS;++i)
		{
			outfile<<bias2To3Layer[i]<<endl;
		}
		outfile.close();
	}
}

void myBP::LoadTrainedModel(const string &srcFilePath)			// 加载已训练好的模型 
{
	ifstream inFile(srcFilePath,ios::in);
	if (!inFile)
	{
		cout<<"load Error"<<endl;
		inFile.close();
	}
	else
	{
		for (int i=0;i<INPUT_LAYER_NODE_CNTS;++i)
		{
			for (int j=0;j<HIDDEN_LAYER_NODE_CNTS;++j)
			{
				inFile>>weight1To2Layer[i][j];
			}
		}
		for (int i=0;i<HIDDEN_LAYER_NODE_CNTS;++i)
		{
			for (int j=0;j<OUTPUT_LAYER_NODE_CNTS;++j)
			{
				inFile>>weight2To3Layer[i][j];
			}
		}
		for (int i=0;i<HIDDEN_LAYER_NODE_CNTS;++i)
		{
			inFile>>bias1To2Layer[i];
		}
		for (int i=0;i<OUTPUT_LAYER_NODE_CNTS;++i)
		{
			inFile>>bias2To3Layer[i];
		}
		inFile.close();
	}
}

void myBP::forward(vector<vector<int>> &dataVec,int index) //前向更新权值
{
	vector<int> pixelValsOfOneTrainingImage = dataVec[index]; //处理一个图片
	for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)				// 2nd：64
	{
		for (int i = 0; i < INPUT_LAYER_NODE_CNTS; ++i)				// 1st: 784
		{
			realOutput2Layer[h] += (weight1To2Layer[i][h] * pixelValsOfOneTrainingImage[i]);	// w * x
		}
		realOutput2Layer[h] += bias1To2Layer[h];														// (w * x) + b 偏置默认是1
		realOutput2Layer[h] =  sigmoid(realOutput2Layer[h]);											// S((w * x) + b)
	}
	for (int o = 0; o < OUTPUT_LAYER_NODE_CNTS; ++o)		// 3rd：10
	{
		for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)	// 2nd：100
		{
			realOutput3Layer[o] += (weight2To3Layer[h][o] * realOutput2Layer[h]);			// w * x
		}
		realOutput3Layer[o] += bias2To3Layer[o];											// (w * x) + b
		realOutput3Layer[o] = sigmoid(realOutput3Layer[o]);									// S((w * x) + b)
	}
}

void myBP::backward(vector<vector<int>> &dataVec,vector<int> & labelVec,int index)
{
	//先把标签转换成向量
    double TempLabelVal[10];
	for (int l=0;l<OUTPUT_LAYER_NODE_CNTS;++l)
	{
		TempLabelVal[l]=0.0;
	}
	TempLabelVal[labelVec[index]]=1.0;
	//计算误差
	for (int OutputLayerNodeIdx = 0; OutputLayerNodeIdx < OUTPUT_LAYER_NODE_CNTS; ++OutputLayerNodeIdx) //求第三层输出的误差
	{
		double OutputVal = realOutput3Layer[OutputLayerNodeIdx];
		delta3Layer[OutputLayerNodeIdx] = OutputVal * (1.0 - OutputVal) * (OutputVal - TempLabelVal[OutputLayerNodeIdx]);
	}
	for (int HiddenLayerNodeIdx = 0; HiddenLayerNodeIdx < HIDDEN_LAYER_NODE_CNTS; ++HiddenLayerNodeIdx) //求第二层的误差
	{
		double tempSum = 0.0;
		for (int OutputLayerNodeIdx = 0; OutputLayerNodeIdx < OUTPUT_LAYER_NODE_CNTS; ++OutputLayerNodeIdx)
		{
			tempSum += weight2To3Layer[HiddenLayerNodeIdx][OutputLayerNodeIdx] * delta3Layer[OutputLayerNodeIdx];	
		}
		double HiddenVal = realOutput2Layer[HiddenLayerNodeIdx];
		delta2Layer[HiddenLayerNodeIdx] = HiddenVal * (1.0 - HiddenVal) * tempSum;											
	}
	//更新权值 其实更新次序没啥关系 每个权重的梯度都等于与其相连的前一层节点的输出（即xi和θ(s1i)）乘以与其相连的后一层的反向传播的输出（即δ1j和δ2j）
	for (int HiddenLayerNodeIdx = 0; HiddenLayerNodeIdx < HIDDEN_LAYER_NODE_CNTS; ++HiddenLayerNodeIdx) //更新2到1层的权值
	{
		bias1To2Layer[HiddenLayerNodeIdx] -= LEARNING_RATE * delta2Layer[HiddenLayerNodeIdx];
		for (int InputLayerNodeIdx = 0; InputLayerNodeIdx < INPUT_LAYER_NODE_CNTS; ++InputLayerNodeIdx)
		{
			weight1To2Layer[InputLayerNodeIdx][HiddenLayerNodeIdx] -= LEARNING_RATE * dataVec[index][InputLayerNodeIdx] * delta2Layer[HiddenLayerNodeIdx];
		}
	}
	for (int OutputLayerNodeIdx = 0; OutputLayerNodeIdx < OUTPUT_LAYER_NODE_CNTS; ++OutputLayerNodeIdx) //更新3到2层的权值
	{
		bias2To3Layer[OutputLayerNodeIdx] -= LEARNING_RATE * delta3Layer[OutputLayerNodeIdx];
		for (int HiddenLayerNodeIdx = 0; HiddenLayerNodeIdx < HIDDEN_LAYER_NODE_CNTS; ++HiddenLayerNodeIdx)
		{
			weight2To3Layer[HiddenLayerNodeIdx][OutputLayerNodeIdx] -= LEARNING_RATE *  realOutput2Layer[HiddenLayerNodeIdx] * delta3Layer[OutputLayerNodeIdx];
		}
	}
}

int myBP::testForOne(vector<vector<int>> &dataVec,int index)
{
	//先根据权值计算输出值
	vector<int> pixelValsOfOneTestingImage = dataVec[index];
	for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)				// 2nd：100
	{
		for (int i = 0; i < INPUT_LAYER_NODE_CNTS; ++i)				// 1st: 784
		{
			realOutput2Layer[h] += (weight1To2Layer[i][h] * pixelValsOfOneTestingImage[i]);	// w * x
		}
		realOutput2Layer[h] += bias1To2Layer[h];														// (w * x) + b
		realOutput2Layer[h] = sigmoid(realOutput2Layer[h]);											// S((w * x) + b)
	}
	for (int o = 0; o < OUTPUT_LAYER_NODE_CNTS; ++o)		// 3rd：10
	{
		for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)	// 2nd：100
		{
			realOutput3Layer[o] += (weight2To3Layer[h][o] * realOutput2Layer[h]);			// w * x
		}
		realOutput3Layer[o] += bias2To3Layer[o];											// (w * x) + b
		realOutput3Layer[o] = sigmoid(realOutput3Layer[o]);									// S((w * x) + b)
	}
	//对输出值进行筛选,找出最大的一个值
	int outLabel=0;
	double tempMax=-99999;
	for (int o = 0; o < OUTPUT_LAYER_NODE_CNTS; ++o) 
	{
		if (realOutput3Layer[o]>tempMax)
		{
			tempMax=realOutput3Layer[o];
			outLabel=o;
		}
	}
	return outLabel;
}

void myBP::useBpNN(vector<vector<int>> &dataVec,vector<int> & labelVec,int flag)
{
	if (dataVec.size()!=labelVec.size())
	{
		cout<<"数据和标签数量不相符"<<endl;
	}
	else
	{
		int num=dataVec.size();//得到数据的大小  
			switch (flag)
		{
			case TRAIN:  //不负责计算准确率，如果要计算准确率可以把模型保存好，再用输入去训练
				{
					Initialize();//先对数据进行初始化
				//采用随机梯度法 每次更新
				time_t startTime=time(0);
				for (int it=0;it<TRAIN_NUM;++it) //训练多次
				{
					time_t tempTimeForNum=time(0);
					cout<<"开始第"<<it+1<<"轮训练,已用时"<<(tempTimeForNum-startTime)<<"秒"<<endl;
					vector<int> randomOrder;
					for (int n=0;n<num;++n) //把数据读入开始处理
					{ 
						randomOrder.push_back(n);
					}
					random_shuffle(randomOrder.begin(),randomOrder.end());		//由于训练集是按顺序来排序的，这里将其打乱顺序,每次采用不同的顺序
					for(int n=0;n<num;++n)
					{
						forward(dataVec,randomOrder[n]); //前向传播
						backward(dataVec,labelVec,randomOrder[n]);//后向更新权值
						if (n%1000==0)
						{
							cout<<"已经训练了"<<n<<"张图片"<<endl;
						}
					}
				}
				time_t endTime=time(0);
				cout<<"训练完成,已用时"<<(endTime-startTime)<<"秒"<<endl;
				}
				break;
			case TEST:
				{
					double rate=0.0;
				    double tempSum=0;
				for (int n=0;n<num;++n) //将数据带入进行测试
				{
					if (labelVec[n]==testForOne(dataVec,n))
					{
						++tempSum;
					}
				}
				rate=tempSum/num*100;
				cout<<"测试完成,"<<"准确率为"<<rate<<endl;
				}
				break;
		}
	}
}