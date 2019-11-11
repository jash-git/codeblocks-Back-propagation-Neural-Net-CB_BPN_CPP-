#include <iostream>
#include "BpNet.h"

using namespace std;

///*
//---
//七段->二進位
//输入样本
double X[trainsample][innode]= {
    {1, 1, 1, 1, 1, 1, 0},
    {0, 1, 1, 0, 0, 0, 0},
    {1, 1, 0, 1, 1, 0, 1},
    {1, 1, 1, 1, 0, 0, 1},
    {0, 1, 1, 0, 0, 1, 1},
    {1, 0, 1, 1, 0, 1, 1},
    {0, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 0, 0, 1, 1}
    };
//期望输出样本
double Y[trainsample][outnode]={
    {0,0,0,0},
    {0,0,0,1},
    {0,0,1,0},
    {0,0,1,1},
    {0,1,0,0},
    {0,1,0,1},
    {0,1,1,0},
    {0,1,1,1},
    {1,0,0,0},
    {1,0,0,1}
    };
//---七段->二進位
//*/

/*
//---
//二進位->七段
//输入样本
double X[trainsample][innode]= {
    {0,0,0,0},
    {0,0,0,1},
    {0,0,1,0},
    {0,0,1,1},
    {0,1,0,0},
    {0,1,0,1},
    {0,1,1,0},
    {0,1,1,1},
    {1,0,0,0},
    {1,0,0,1}
    };
//期望输出样本
double Y[trainsample][outnode]={
    {1, 1, 1, 1, 1, 1, 0},
    {0, 1, 1, 0, 0, 0, 0},
    {1, 1, 0, 1, 1, 0, 1},
    {1, 1, 1, 1, 0, 0, 1},
    {0, 1, 1, 0, 0, 1, 1},
    {1, 0, 1, 1, 0, 1, 1},
    {0, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 0, 0, 1, 1}
    };
//---二進位->七段
//*/
void Pause()
{
    printf("Press Enter key to continue...");
    fgetc(stdin);
}
int main()
{
    bool blntrainning = false;
    BpNet bp;
    BpNet bp_f;
    bp.init();
    bp_f.init();
    long times=0;
    if(blntrainning)
    {
        while((bp.error>0.0001)&&(times<100000000))
        {
            bp.e=0.0;
            times++;
            bp.train(X,Y);
            if((times%100000)==0)
            {
                cout<<"Times="<<times<<" error="<<bp.error<<endl;
            }
        }
        if(bp.error<0.0001)
        {
            cout<<"trainning complete...OK"<<endl;
            cout<<"Times="<<times<<" error="<<bp.error<<endl;
            bp.writetrain();
            bp_f.readtrain();
        }
        else
        {
            cout<<"trainning complete...Fail"<<endl;
        }
    }
    else
    {
        bp_f.readtrain();
    }

    for(int j=0;j<trainsample;j++)
    {
        for(int i=0;i<innode;++i)
        {
            cout<< X[j][i] <<" ";
        }
        cout<<" -> ";
        double *r=bp_f.recognize(X[j]);
        for(int i=0;i<outnode;++i)
        {
           cout<<round(bp_f.result[i])<<" ";

        }
        cout<<endl;
    }

    Pause();
    return 0;
}
