#include "mnistRead.h"
#include "bp.h"
#include <algorithm>


#define trainDataPath "train-images.idx3-ubyte"
#define trainLabelPath "train-labels.idx1-ubyte"
#define testDataPath "t10k-images.idx3-ubyte"
#define testLabelPath "t10k-labels.idx1-ubyte"


int main(){

	//���Բ���
	Mnist data;
	bp bpTest;
	bpTest.loadModel("100-9735-50.txt");//����ʱ�ȶ���ģ��
	cout<<"The model is loaded, begin reading,The procedure maybe slow,please wait...."<<endl;
	data.readData(testDataPath,testLabelPath);
	cout<<"The data is loaded. Test is beginning...."<<endl;
	time_t startTime=time(0);
	double test_num = 0.0;//ͳ����ȷ����
	int tempNum=data.PixelOfImages.size();
	for (int l=0;l<tempNum;++l)
	{
		memcpy(bpTest.input,&data.PixelOfImages[l][0],sizeof(bpTest.input)); //����ʱ����˳�����ݶ���
		if (data.LabelOfImages[l]==bpTest.testing())
		{
			++test_num;
		}
	}
	double test_success_count = test_num/tempNum;
	time_t endTime=time(0);
	cout<<"Success Accurate:"<<test_success_count<<" use time:"<<endTime-startTime<<"seconds"<<endl;
}
