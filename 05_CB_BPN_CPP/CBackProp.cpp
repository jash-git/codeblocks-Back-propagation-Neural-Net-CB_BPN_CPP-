#include "CBackProp.h"

#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>       /* exp */
#include <string.h>       /*strlen*/

CBackProp::CBackProp()
{
    //ctor
}

//	initializes and allocates memory on heap
CBackProp::CBackProp(int nl,int *sz,double b,double a):beta(b),alpha(a)
{

	//	set no of layers and their sizes
	numl=nl;
	lsize=new int[numl];

	for(int i=0;i<numl;i++){
		lsize[i]=sz[i];
	}

	//	allocate memory for output of each neuron
	out = new double*[numl];

	for(int i=0;i<numl;i++){
		out[i]=new double[lsize[i]];
	}

	//	allocate memory for delta
	delta = new double*[numl];

	for(int i=1;i<numl;i++){
		delta[i]=new double[lsize[i]];
	}

	//	allocate memory for weights
	weight = new double**[numl];

	for(int i=1;i<numl;i++){
		weight[i]=new double*[lsize[i]];
	}
	for(int i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			weight[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	allocate memory for previous weights
	prevDwt = new double**[numl];

	for(int i=1;i<numl;i++){
		prevDwt[i]=new double*[lsize[i]];

	}
	for(int i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			prevDwt[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	seed and assign random weights
	srand((unsigned)(time(NULL)));
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;//32767

	//	initialize previous weights to 0 for first iteration
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				prevDwt[i][j][k]=(double)0.0;

// Note that the following variables are unused,
//
// delta[0]
// weight[0]
// prevDwt[0]

//  I did this intentionaly to maintains consistancy in numbering the layers.
//  Since for a net having n layers, input layer is refered to as 0th layer,
//  first hidden layer as 1st layer and the nth layer as output layer. And
//  first (0th) layer just stores the inputs hence there is no delta or weigth
//  values corresponding to it.
}



CBackProp::~CBackProp()
{
	//	free out
	for(int i=0;i<numl;i++)
		delete[] out[i];
	delete[] out;

	//	free delta
	for(int i=1;i<numl;i++)
		delete[] delta[i];
	delete[] delta;

	//	free weight
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] weight[i][j];
	for(int i=1;i<numl;i++)
		delete[] weight[i];
	delete[] weight;

	//	free prevDwt
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] prevDwt[i][j];
	for(int i=1;i<numl;i++)
		delete[] prevDwt[i];
	delete[] prevDwt;

	//	free layer info
	delete[] lsize;
}

//	sigmoid function
double CBackProp::sigmoid(double in)
{
		return (double)(1/(1+exp(-in)));
}

//	mean square error
double CBackProp::mse(double *tgt) const
{
	double mse=0;
	for(int i=0;i<lsize[numl-1];i++){
		mse+=(tgt[i]-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}
	return mse/2;
}


//	returns i'th output of the net
double CBackProp::Out(int i) const
{
	return out[numl-1][i];
}

//jash add_start
void CBackProp::saveVar(char *name)
{
    FILE *pfsave=fopen(name,"w");
    fprintf(pfsave,"%d\n",numl);
    for(int i=0;i<numl;i++)
    {
        fprintf(pfsave,"%d\n",lsize[i]);
    }
    fprintf(pfsave,"%f\n",beta);
    fprintf(pfsave,"%f\n",alpha);
    fprintf(pfsave,"%~~~~~\n");
    for(int i=1;i<numl;i++){
        for(int j=0;j<lsize[i];j++){
            for(int k=0;k<lsize[i-1];k++){
                fprintf(pfsave,"%f\n",weight[i][j][k]);
            }
        }
    }
    fclose(pfsave);
}
void CBackProp::loadVar(char *name)
{
    int l=0;
    int m=0;
    int n=0;
    float *fweight;//ANN神經元權重一維陣列變數
    int wsize=0;//ANN神經元權重一維陣列變數個數計算
    bool blnSeparate=false;//判斷是否到間隔符號旗標

    FILE *pfload=fopen(name,"r");
    char buf[513];
    memset(buf,0,sizeof(buf));
    while(fgets(buf,512,pfload) != NULL)
    {
        buf[strlen(buf)-1]='\0';
        if(l==0)//物件內的變數動態配置初始化第一階段
        {
            numl = atoi(buf);
            lsize = new int[numl];

            //	allocate memory for output of each neuron
            out = new double*[numl];

            //	allocate memory for delta
            delta = new double*[numl];

            //	allocate memory for weights
            weight = new double**[numl];

            //	allocate memory for previous weights
            prevDwt = new double**[numl];

            l++;
        }
        else
        {
            if(strcmp("~~~~~", buf) == 0)//物件內的變數動態配置初始化第二階段
            {
                blnSeparate=true;

                for(int i=0;i<numl;i++){
                    out[i]=new double[lsize[i]];
                }
                for(int i=1;i<numl;i++){
                    delta[i]=new double[lsize[i]];
                }
                for(int i=1;i<numl;i++){
                    weight[i]=new double*[lsize[i]];
                }
                for(int i=1;i<numl;i++){
                    for(int j=0;j<lsize[i];j++){
                        weight[i][j]=new double[lsize[i-1]+1];
                    }
                }
                for(int i=1;i<numl;i++){
                    prevDwt[i]=new double*[lsize[i]];

                }
                for(int i=1;i<numl;i++){
                    for(int j=0;j<lsize[i];j++){
                        prevDwt[i][j]=new double[lsize[i-1]+1];
                    }
                }

                //	seed and assign random weights
                srand((unsigned)(time(NULL)));
                for(int i=1;i<numl;i++)
                    for(int j=0;j<lsize[i];j++)
                        for(int k=0;k<lsize[i-1]+1;k++)
                            weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;//32767

                //	initialize previous weights to 0 for first iteration
                for(int i=1;i<numl;i++)
                    for(int j=0;j<lsize[i];j++)
                        for(int k=0;k<lsize[i-1]+1;k++)
                            prevDwt[i][j][k]=(double)0.0;

                fweight=new float[wsize];
                continue;
            }
            if(!blnSeparate)//設定物件內基礎變數值
            {
                if(m<numl)
                {
                   lsize[m] = atoi(buf);
                   if(m>0)
                   {
                       wsize+=lsize[m]*lsize[m-1];//計算權重變數個數
                   }
                }
                else
                {
                    if(m==numl)
                    {
                        beta = atof(buf);
                    }
                    else
                    {
                        alpha = atof(buf);
                    }
                }
                m++;
            }
            else//抓取所有權重變數值放到站存區
            {
                if(strlen(buf)>2)
                {
                    fweight[n] = atof(buf);
                }
                else
                {
                    break;
                }
                n++;
            }
        }
        memset(buf,0,sizeof(buf));
    }

    n=0;
    for(int i=1;i<numl;i++){
        for(int j=0;j<lsize[i];j++){
            for(int k=0;k<lsize[i-1];k++){
                weight[i][j][k]=fweight[n];//設定所有權重變數值
                n++;
            }
        }
    }

    fclose(pfload);
    delete[] fweight;
}
//jash add_end

// feed forward one set of input
void CBackProp::ffwd(double *in)
{
	double sum;

	//	assign content to input layer
	for(int i=0;i<lsize[0];i++)
		out[0][i]=in[i];  // output_from_neuron(i,j) Jth neuron in Ith Layer

	//	assign output(activation) value
	//	to each neuron usng sigmoid func
	for(int i=1;i<numl;i++){				// For each layer
		for(int j=0;j<lsize[i];j++){		// For each neuron in current layer
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		// For input from each neuron in preceeding layer
				sum+= out[i-1][k]*weight[i][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=weight[i][j][lsize[i-1]];		// Apply bias
			out[i][j]=sigmoid(sum);				// Apply sigmoid function
		}
	}
}


//	backpropogate errors from output
//	layer uptill the first hidden layer
void CBackProp::bpgt(double *in,double *tgt)
{
	double sum;

	//	update output values for each neuron
	ffwd(in);

	//	find delta for output layer
	for(int i=0;i<lsize[numl-1];i++){
		delta[numl-1][i]=out[numl-1][i]*
		(1-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}

	//	find delta for hidden layers
	for(int i=numl-2;i>0;i--){
		for(int j=0;j<lsize[i];j++){
			sum=0.0;
			for(int k=0;k<lsize[i+1];k++){
				sum+=delta[i+1][k]*weight[i+1][k][j];
			}
			delta[i][j]=out[i][j]*(1-out[i][j])*sum;
		}
	}

	//	apply momentum ( does nothing if alpha=0 )
	for(int i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				weight[i][j][k]+=alpha*prevDwt[i][j][k];
			}
			weight[i][j][lsize[i-1]]+=alpha*prevDwt[i][j][lsize[i-1]];
		}
	}

	//	adjust weights usng steepest descent
	for(int i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				prevDwt[i][j][k]=beta*delta[i][j]*out[i-1][k];
				weight[i][j][k]+=prevDwt[i][j][k];
			}
			prevDwt[i][j][lsize[i-1]]=beta*delta[i][j];
			weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
		}
	}
}
