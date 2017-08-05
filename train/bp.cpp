#include "bp.h"
void bp::saveModel(const string &dstFilePath)
{
	ofstream outfile(dstFilePath,ios::out);
	if(!outfile)//文件未打开不保存
	{
		cout<<"save Error"<<endl;
		outfile.close();
	}else
	{
		for (int i=0;i<inputNum;++i)
		{
			for (int j=0;j<hidderNum;++j)
			{
				outfile<<weight1To2[i][j]<<endl;
			}
		}
		for (int i=0;i<hidderNum;++i)
		{
			for (int j=0;j<outputNum;++j)
			{
				outfile<<weight2To3[i][j]<<endl;
			}
		}
		for (int i=0;i<hidderNum;++i)
		{
			outfile<<bias1[i]<<endl;
		}
		for (int i=0;i<outputNum;++i)
		{
			outfile<<bias2[i]<<endl;
		}
		outfile.close();
	}
}

void bp::loadModel(const string &srcFilePath)
{
	ifstream inFile(srcFilePath,ios::in);
	if (!inFile)
	{
		cout<<"load Error"<<endl;
		inFile.close();
	}
	else
	{
		for (int i=0;i<inputNum;++i)
		{
			for (int j=0;j<hidderNum;++j)
			{
				inFile>>weight1To2[i][j];
			}
		}
		for (int i=0;i<hidderNum;++i)
		{
			for (int j=0;j<outputNum;++j)
			{
				inFile>>weight2To3[i][j];
			}
		}
		for (int i=0;i<hidderNum;++i)
		{
			inFile>>bias1[i];
		}
		for (int i=0;i<outputNum;++i)
		{
			inFile>>bias2[i];
		}
		inFile.close();
	}
}

inline double bp::sigmoid(double x){
	return 1.0 / (1.0 + exp(-x));
}

inline double bp::randInt()
{
	return ((rand()/(double)(RAND_MAX)*2*0.076)-0.076); //经验 sqrt(6)/sqrt(N_in+N_out) 
}

void bp::forward1To2(){
	for (int j = 0; j < hidderNum; ++j){
		double sigma = 0;
		for (int i = 0; i < inputNum; ++i){
			sigma += input[i] * weight1To2[i][j]; 
		}
		double x = sigma + bias1[j];
		outputLayer2[j] = sigmoid(x);
	}
}

void bp::forward2To3(){
	for (int k = 0; k < outputNum; ++k){
		double sigma = 0;
		for (int j = 0; j < hidderNum; ++j){
			sigma += outputLayer2[j] * weight2To3[j][k];
		}
		double x = sigma + bias2[k];
		outputLayer3[k] = sigmoid(x);
	}
}

void bp::backward3To2(){
	for (int k = 0; k < outputNum; ++k){
		deltaLayer3[k] = (outputLayer3[k]) * (1.0 - outputLayer3[k]) * (outputLayer3[k] - target[k]);
	}
}

void bp::backward2To1(){
	for (int j = 0; j < hidderNum; ++j){
		double sigma = 0;
		for (int k = 0; k < outputNum; ++k){
			sigma += weight2To3[j][k] * deltaLayer3[k];
		}
		deltaLayer2[j] = (outputLayer2[j]) * (1.0 - outputLayer2[j]) * sigma;
	}
}

void bp::renew1To2(){
	for (int j = 0; j < hidderNum; ++j){
		bias1[j] = bias1[j] - learnRate * deltaLayer2[j];
		for (int i = 0; i < inputNum; ++i){
			weight1To2[i][j] = weight1To2[i][j] - learnRate * input[i] * deltaLayer2[j];
		}
	}
}

void bp::renew2To3(){
	for (int k = 0; k < outputNum; ++k){
		bias2[k] = bias2[k] - learnRate * deltaLayer3[k];
		for (int j = 0; j < hidderNum; ++j){
			weight2To3[j][k] = weight2To3[j][k] - learnRate * outputLayer2[j] * deltaLayer3[k];
		}
	}
}

void bp::initialize(){
	for (int i = 0; i < inputNum; ++i){
		for (int j = 0; j < hidderNum; ++j){
			weight1To2[i][j] = randInt();
		}
	}	
	for (int j = 0; j < hidderNum; ++j){
		for (int k = 0; k < outputNum; ++k){
			weight2To3[j][k] = randInt();
		}
	}

	for (int j = 0; j < hidderNum; ++j){
		bias1[j] = randInt();
	}
	for (int k = 0; k < outputNum; ++k){
		bias2[k] = randInt();
	}
}

void bp::training(){//每次只是更改，减少了之前的寻址操作

	//FILE *image_train;
	//FILE *image_label;
	//image_train = fopen("../tc/train-images.idx3-ubyte", "rb");
	//image_label = fopen("../tc/train-labels.idx1-ubyte", "rb");
	//if (image_train == NULL || image_label == NULL){
	//	cout << "can't open the file!" << endl;
	//	exit(0);
	//}
	//unsigned char image_buf[784];
	//unsigned char label_buf[10];
	//int useless[1000];
	//fread(useless, 1, 16, image_train);//跳过前面文件头
	//fread(useless, 1, 8, image_label);
	//int cnt = 0;
	//cout << "Start training..." << endl;
	////60000 times
	//while (!feof(image_train) && !feof(image_label)){
	//	memset(image_buf, 0, 784);
	//	memset(label_buf, 0, 10);
	//	fread(image_buf, 1, 784, image_train);
	//	fread(label_buf, 1, 1, image_label);
	//	//initialize the input by 28 x 28 (0,1)matrix of the images
	//	for (int i = 0; i < 784; i++){
	//		if ((unsigned int)image_buf[i] < 128){
	//			input[i] = 0;
	//		}
	//		else{
	//			input[i] = 1;
	//		}
	//	}
	//	//initialize the target output
	//	int target_value = (unsigned int)label_buf[0];
	//	for (int k = 0; k < outputNum; k++){
	//		target[k] = 0;
	//	}
	//	target[target_value] = 1;
	//一次处理

	forward1To2();
	forward2To3();
	backward3To2();
	backward2To1();
	renew1To2();
	renew2To3();
}


int bp::testing(){ //应该也是可以拆开的 把里面的函数单独拿出来

	//double test_num = 0.0;
	//double test_success_count = 0.0;
	//FILE *image_test;
	//FILE *image_test_label;
	//image_test = fopen("../tc/t10k-images.idx3-ubyte", "rb");
	//image_test_label = fopen("../tc/t10k-labels.idx1-ubyte", "rb");
	//if (image_test == NULL || image_test_label == NULL){
	//	cout << "can't open the file!" << endl;
	//	exit(0);
	//}
	//unsigned char image_buf[784];
	//unsigned char label_buf[10];
	//int useless[1000];
	//fread(useless, 1, 16, image_test);
	//fread(useless, 1, 8, image_test_label);
	//while (!feof(image_test) && !feof(image_test_label)){
	//	memset(image_buf, 0, 784);
	//	memset(label_buf, 0, 10);
	//	fread(image_buf, 1, 784, image_test);
	//	fread(label_buf, 1, 1, image_test_label);
	//	//initialize the input by 28 x 28 (0,1)matrix of the images
	//	for (int i = 0; i < 784; i++){
	//		if ((unsigned int)image_buf[i] < 128){
	//			input[i] = 0;
	//		}
	//		else{
	//			input[i] = 1;
	//		}
	//	}
	//	//initialize the target output
	//	for (int k = 0; k < outputNum; k++){
	//		target[k] = 0;
	//	}
	//	int target_value = (unsigned int)label_buf[0];
	//	target[target_value] = 1;

		//get the ouput and compare with the targe
		forward1To2();
		forward2To3();
		double max_value = -99999;
		int max_index = 0;
		for (int k = 0; k < outputNum; k++){
			if (outputLayer3[k] > max_value){
				max_value = outputLayer3[k];
				max_index = k;
			}
		}
		return max_index;
//		//output == target
//		if (target[max_index] == 1){
//			test_success_count ++;
//		}
//
//		test_num ++;
//
//		if ((int)test_num % 1000 == 0){
//			cout << "test num: " << test_num << "  success: " << test_success_count << endl;
//		}
//	}
//	cout << endl;
//	cout << "The success rate: " << test_success_count / test_num << endl;
}
