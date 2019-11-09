#include <iostream>
#include <cstdio>
#include <cmath>

#include "CBackProp.h"

using namespace std;

void Pause()
{
    printf("Press Enter key to continue...");
    fgetc(stdin);
}

int main()
{
    long i=0;
	// 七段顯示器轉二進位
	double data[][11]={
                            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                            1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                            1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
                            0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0,
                            1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
                            0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
                            1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                            1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1};
                          //a, b, c, d, e, f, g, 8, 4, 2, 1

	// prepare test data
	double testData[][7]={
                            1, 1, 1, 1, 1, 1, 0,
                            0, 1, 1, 0, 0, 0, 0,
                            1, 1, 0, 1, 1, 0, 1,
                            1, 1, 1, 1, 0, 0, 1,
                            0, 1, 1, 0, 0, 1, 1,
                            1, 0, 1, 1, 0, 1, 1,
                            0, 0, 1, 1, 1, 1, 1,
                            1, 1, 1, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 0, 0, 1, 1};

    //*
    //---
    //ANN model Training

	// defining a net with 4 layers having 3,3,3, and 1 neuron respectively,
	// the first layer is input layer i.e. simply holder for the input parameters
	// and has to be the same size as the no of input parameters, in out example 3
	int numLayers = 3, lSz[3] = {7,17,4};


	// Learing rate - beta
	// momentum - alpha
	// Threshhold - thresh (value of target mse, training stops once it is achieved)
	double beta = 0.1, alpha = 0.1, Thresh = 0.000001;


	// maximum no of iterations during training
	long num_iter = 100000000;


	// Creating the net
	CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha);

	cout<< endl <<  "Now training the network...." << endl;
	for (i=0; i<num_iter ; i++)
	{

		bp->bpgt(data[i%10], &data[i%10][7]);

		if( bp->mse(&data[i%10][7]) < Thresh) {
			cout << endl << "Network Trained. Threshold value achieved in " << i << " iterations." << endl;
			cout << "MSE:  " << bp->mse(&data[i%10][7])
				 <<  endl <<  endl;
			break;
		}
		if ( i%(num_iter/10) == 0 )
			cout<<  endl <<  "MSE:  " << bp->mse(&data[i%10][7])
				<< "... Training..." << endl;

	}

	if ( i == num_iter )
		cout << endl << i << " iterations completed..."
		<< "MSE: " << bp->mse(&data[(i-1)%10][7]) << endl;

	cout<< "Now using the trained network to make predctions on test data...." << endl << endl;
	for ( i = 0 ; i < 10 ; i++ )
	{
		bp->ffwd(testData[i]);
		cout << testData[i][0]<< "  " << testData[i][1]<< "  "  << testData[i][2]<< "  " << testData[i][3]<< "  " << testData[i][4]<< "  "  << testData[i][5]<< "  " << testData[i][6] << " -> " << round(bp->Out(0))  << "  " << round(bp->Out(1))  << "  " << round(bp->Out(2))  << "  " << round(bp->Out(3)) << endl;
	}
    bp->saveVar("20191108.var");
    //---ANN model Training
    //*/

	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "fp output..." << endl;
    CBackProp *fp = new CBackProp();
    fp->loadVar("20191108.var");
	for ( i = 0 ; i < 10 ; i++ )
	{
		fp->ffwd(testData[i]);
		cout << testData[i][0]<< "  " << testData[i][1]<< "  "  << testData[i][2]<< "  " << testData[i][3]<< "  " << testData[i][4]<< "  "  << testData[i][5]<< "  " << testData[i][6] << " -> " << round(fp->Out(0))  << "  " << round(fp->Out(1))  << "  " << round(fp->Out(2))  << "  " << round(fp->Out(3)) << endl;
	}
	Pause();
    return 0;
}
