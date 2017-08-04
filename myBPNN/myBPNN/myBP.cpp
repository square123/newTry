#include "myBP.h"

myBP::myBP()
{
	//�Ƚ��洢����Ͳ������
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
	return ((rand()/(double)(RAND_MAX)*2*0.076)-0.076); //���� sqrt(6)/sqrt(N_in+N_out) 
}

void myBP::Initialize()				//������������ʼ��   �������ѵ���Ͳ���Ҫ��ʼ����
{
	//Ȩֵ�����ʼ��
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

void myBP::SaveTrainedModel(const string &dstFilePath)			// ������ѵ���õ�ģ��  
{
	ofstream outfile(dstFilePath,ios::out);
	if(!outfile)//�ļ�δ�򿪲�����
	{
		cout<<"save Error"<<endl;
		outfile.close();
	}else //��Ȩֵ����Ϊtxt��ʽ
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

void myBP::LoadTrainedModel(const string &srcFilePath)			// ������ѵ���õ�ģ�� 
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

void myBP::forward(vector<vector<int>> &dataVec,int index) //ǰ�����Ȩֵ
{
	vector<int> pixelValsOfOneTrainingImage = dataVec[index]; //����һ��ͼƬ
	for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)				// 2nd��64
	{
		for (int i = 0; i < INPUT_LAYER_NODE_CNTS; ++i)				// 1st: 784
		{
			realOutput2Layer[h] += (weight1To2Layer[i][h] * pixelValsOfOneTrainingImage[i]);	// w * x
		}
		realOutput2Layer[h] += bias1To2Layer[h];														// (w * x) + b ƫ��Ĭ����1
		realOutput2Layer[h] =  sigmoid(realOutput2Layer[h]);											// S((w * x) + b)
	}
	for (int o = 0; o < OUTPUT_LAYER_NODE_CNTS; ++o)		// 3rd��10
	{
		for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)	// 2nd��100
		{
			realOutput3Layer[o] += (weight2To3Layer[h][o] * realOutput2Layer[h]);			// w * x
		}
		realOutput3Layer[o] += bias2To3Layer[o];											// (w * x) + b
		realOutput3Layer[o] = sigmoid(realOutput3Layer[o]);									// S((w * x) + b)
	}
}

void myBP::backward(vector<vector<int>> &dataVec,vector<int> & labelVec,int index)
{
	//�Ȱѱ�ǩת��������
    double TempLabelVal[10];
	for (int l=0;l<OUTPUT_LAYER_NODE_CNTS;++l)
	{
		TempLabelVal[l]=0.0;
	}
	TempLabelVal[labelVec[index]]=1.0;
	//�������
	for (int OutputLayerNodeIdx = 0; OutputLayerNodeIdx < OUTPUT_LAYER_NODE_CNTS; ++OutputLayerNodeIdx) //���������������
	{
		double OutputVal = realOutput3Layer[OutputLayerNodeIdx];
		delta3Layer[OutputLayerNodeIdx] = OutputVal * (1.0 - OutputVal) * (OutputVal - TempLabelVal[OutputLayerNodeIdx]);
	}
	for (int HiddenLayerNodeIdx = 0; HiddenLayerNodeIdx < HIDDEN_LAYER_NODE_CNTS; ++HiddenLayerNodeIdx) //��ڶ�������
	{
		double tempSum = 0.0;
		for (int OutputLayerNodeIdx = 0; OutputLayerNodeIdx < OUTPUT_LAYER_NODE_CNTS; ++OutputLayerNodeIdx)
		{
			tempSum += weight2To3Layer[HiddenLayerNodeIdx][OutputLayerNodeIdx] * delta3Layer[OutputLayerNodeIdx];	
		}
		double HiddenVal = realOutput2Layer[HiddenLayerNodeIdx];
		delta2Layer[HiddenLayerNodeIdx] = HiddenVal * (1.0 - HiddenVal) * tempSum;											
	}
	//����Ȩֵ ��ʵ���´���ûɶ��ϵ ÿ��Ȩ�ص��ݶȶ���������������ǰһ��ڵ���������xi�ͦ�(s1i)���������������ĺ�һ��ķ��򴫲������������1j�ͦ�2j��
	for (int HiddenLayerNodeIdx = 0; HiddenLayerNodeIdx < HIDDEN_LAYER_NODE_CNTS; ++HiddenLayerNodeIdx) //����2��1���Ȩֵ
	{
		bias1To2Layer[HiddenLayerNodeIdx] -= LEARNING_RATE * delta2Layer[HiddenLayerNodeIdx];
		for (int InputLayerNodeIdx = 0; InputLayerNodeIdx < INPUT_LAYER_NODE_CNTS; ++InputLayerNodeIdx)
		{
			weight1To2Layer[InputLayerNodeIdx][HiddenLayerNodeIdx] -= LEARNING_RATE * dataVec[index][InputLayerNodeIdx] * delta2Layer[HiddenLayerNodeIdx];
		}
	}
	for (int OutputLayerNodeIdx = 0; OutputLayerNodeIdx < OUTPUT_LAYER_NODE_CNTS; ++OutputLayerNodeIdx) //����3��2���Ȩֵ
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
	//�ȸ���Ȩֵ�������ֵ
	vector<int> pixelValsOfOneTestingImage = dataVec[index];
	for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)				// 2nd��100
	{
		for (int i = 0; i < INPUT_LAYER_NODE_CNTS; ++i)				// 1st: 784
		{
			realOutput2Layer[h] += (weight1To2Layer[i][h] * pixelValsOfOneTestingImage[i]);	// w * x
		}
		realOutput2Layer[h] += bias1To2Layer[h];														// (w * x) + b
		realOutput2Layer[h] = sigmoid(realOutput2Layer[h]);											// S((w * x) + b)
	}
	for (int o = 0; o < OUTPUT_LAYER_NODE_CNTS; ++o)		// 3rd��10
	{
		for (int h = 0; h < HIDDEN_LAYER_NODE_CNTS; ++h)	// 2nd��100
		{
			realOutput3Layer[o] += (weight2To3Layer[h][o] * realOutput2Layer[h]);			// w * x
		}
		realOutput3Layer[o] += bias2To3Layer[o];											// (w * x) + b
		realOutput3Layer[o] = sigmoid(realOutput3Layer[o]);									// S((w * x) + b)
	}
	//�����ֵ����ɸѡ,�ҳ�����һ��ֵ
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
		cout<<"���ݺͱ�ǩ���������"<<endl;
	}
	else
	{
		int num=dataVec.size();//�õ����ݵĴ�С  
			switch (flag)
		{
			case TRAIN:  //���������׼ȷ�ʣ����Ҫ����׼ȷ�ʿ��԰�ģ�ͱ���ã���������ȥѵ��
				{
					Initialize();//�ȶ����ݽ��г�ʼ��
				//��������ݶȷ� ÿ�θ���
				time_t startTime=time(0);
				for (int it=0;it<TRAIN_NUM;++it) //ѵ�����
				{
					time_t tempTimeForNum=time(0);
					cout<<"��ʼ��"<<it+1<<"��ѵ��,����ʱ"<<(tempTimeForNum-startTime)<<"��"<<endl;
					vector<int> randomOrder;
					for (int n=0;n<num;++n) //�����ݶ��뿪ʼ����
					{ 
						randomOrder.push_back(n);
					}
					random_shuffle(randomOrder.begin(),randomOrder.end());		//����ѵ�����ǰ�˳��������ģ����ｫ�����˳��,ÿ�β��ò�ͬ��˳��
					for(int n=0;n<num;++n)
					{
						forward(dataVec,randomOrder[n]); //ǰ�򴫲�
						backward(dataVec,labelVec,randomOrder[n]);//�������Ȩֵ
						if (n%1000==0)
						{
							cout<<"�Ѿ�ѵ����"<<n<<"��ͼƬ"<<endl;
						}
					}
				}
				time_t endTime=time(0);
				cout<<"ѵ�����,����ʱ"<<(endTime-startTime)<<"��"<<endl;
				}
				break;
			case TEST:
				{
					double rate=0.0;
				    double tempSum=0;
				for (int n=0;n<num;++n) //�����ݴ�����в���
				{
					if (labelVec[n]==testForOne(dataVec,n))
					{
						++tempSum;
					}
				}
				rate=tempSum/num*100;
				cout<<"�������,"<<"׼ȷ��Ϊ"<<rate<<endl;
				}
				break;
		}
	}
}