#ifndef BPNET_H
#define BPNET_H

//http://www.voidcn.com/article/p-qoklaglm-vu.html
#include<cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>

///*
//---
//七段->二進位
#define  innode 7  //输入结点数
#define  hidenode 17//隐含结点数
#define  outnode 4 //输出结点数
//---七段->二進位
//*/

/*
//---
//二進位->七段
#define  innode 4  //输入结点数
#define  hidenode 22//隐含结点数
#define  outnode 7 //输出结点数
//---二進位->七段
//*/

#define  trainsample 10//BP训练样本数

class BpNet
{
public:
    void train(double p[trainsample][innode],double t[trainsample][outnode]);//Bp训练
    double p[trainsample][innode];     //输入的样本
    double t[trainsample][outnode];    //样本要输出的

    double *recognize(double *p);//Bp识别

    void writetrain(); //写训练完的权值
    void readtrain(); //读训练好的权值，这使的不用每次去训练了，只要把训练最好的权值存下来就OK

    BpNet();
    virtual ~BpNet();

public:
    void init();
    double w[innode][hidenode];//隐含结点权值
    double w1[hidenode][outnode];//输出结点权值
    double b1[hidenode];//隐含结点阀值
    double b2[outnode];//输出结点阀值

    double rate_w; //权值学习率（输入层-隐含层)
    double rate_w1;//权值学习率 (隐含层-输出层)
    double rate_b1;//隐含层阀值学习率
    double rate_b2;//输出层阀值学习率

    double e;//误差计算
    double error;//允许的最大误差
    double result[outnode];// Bp输出
};

#endif // BPNET_H
