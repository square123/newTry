#include "mnistRead.h"
#include "bp.h"
#include <algorithm>


#define trainDataPath "train-images.idx3-ubyte"
#define trainLabelPath "train-labels.idx1-ubyte"
#define testDataPath "t10k-images.idx3-ubyte"
#define testLabelPath "t10k-labels.idx1-ubyte"


int main(){
	//ѵ������
	Mnist data;
	bp bpTrain;
	bpTrain.initialize();//ѵ��ʱ�ȳ�ʼ��
	cout<<"begin reading, the procedure maybe long, please wait....."<<endl;
	data.readData(trainDataPath,trainLabelPath);
	vector<int> randNum;
	int tempNum=data.PixelOfImages.size();
	for (int i=0;i<tempNum;++i)
	{
		randNum.push_back(i);
	}
	int target[10];
	time_t startTime=time(0);
	cout<<"the data is loaded, the train is beginning:"<<endl;
	for (int k=0;k<100;++k) //ѵ������
	{
	cout<<"The No."<<k+1<<" train procedure"<<endl;
		random_shuffle(randNum.begin(),randNum.end());//����˳��
		for (int l=0;l<tempNum;++l)
		{
			memcpy(bpTrain.input,&data.PixelOfImages[randNum[l]][0],sizeof(bpTrain.input));//���鿽�� ��ʱ�벻�����õĴ��� ȷʵ�ܿ죬һ��ѵ���������28�룬��vectorѰַҪ��
			memset(target,0,sizeof(target));
			target[data.LabelOfImages[randNum[l]]]=1;
			memcpy(bpTrain.target,target,sizeof(bpTrain.target));
			bpTrain.training();
			if (l%1000==0)
			{
				cout<<l<<endl;
			}
		}
		time_t endTime=time(0);
		cout<<"have used time "<<endTime-startTime<<" seconds"<<endl;
	}
	bpTrain.saveModel("haha.txt");
}
