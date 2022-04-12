#include<iostream>
#include <sys/time.h>
#include<arm_neon.h>
using namespace std;
float A[2000][2000],B[2000][2000],C[2000][2000],D[2000][2000];
void Initialize(int n)//初始化
{
	int i,j,k;
	for(i=0;i<n;i++)
	{
		for(j=0;j<i;j++){
			A[i][j]=0;//下三角元素初始化为零
			B[i][j]=0;
			C[i][j]=0;
			D[i][j]=0;
		}
		A[i][i]=1.0;//对角线元素初始化为1
		B[i][i]=1.0;
		C[i][i]=1.0;
		D[i][i]=1.0;
		for(j=i+1;j<n;j++){
			A[i][j]=rand();//上三角元素初始化为随机数
			B[i][j]=A[i][j];
			C[i][j]=A[i][j];
			D[i][j]=A[i][j];
		}
	}
	for(k=0;k<n;k++)
		for(i=k+1;i<n;i++)
			for(j=0;j<n;j++){
				A[i][j]+=A[k][j];//最终每一行的值是上一行的值与这一行的值之和
				B[i][j]+=B[k][j];
				C[i][j]+=C[k][j];
				D[i][j]+=D[k][j];
			}
}
void Gauss_normal(int n){//串行算法
	int i,j,k;
	for(k=0;k<n;k++)
    {
		float tmp=A[k][k];
		for(j=k;j<n;j++)
			A[k][j]/=tmp;//可以进行向量化，用SIMD 扩展指令进行并行优化
		for(i=k+1;i<n;i++)\
		{
			float tmp2=A[i][k];
			for(j=k+1;j<n;j++)
				A[i][j]-=tmp2*A[k][j];//可以进行向量化，用SIMD 扩展指令进行并行优化
			A[i][k]=0;
		}
	}
}
void Gauss_para1(int n){ //对第一个部分进行向量化的SSE并行算法
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float32_t tmp[4]={B[k][k],B[k][k],B[k][k],B[k][k]};
        t1=vld1q_f32(tmp); //加载到t1向量寄存器
        for(j=k+1;j+4<=n;j+=4)
        {
            t2=vld1q_f32(B[k]+j); //把内存中从B[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=vdivq_f32(t2,t1); //相除结果放到t3寄存器
            vst1q_f32(B[k]+j,t3); //把t3寄存器的值放回内存
        }
        for(j;j<n;j++) //处理剩下的不能被4整除的
            B[k][j]/=B[k][k];
        B[k][k]=1.0;
        //以上完成了对第一个部分的向量化

        for(i=k+1;i<n;i++)
        {
			float tmp2=B[i][k];
			for(j=k+1;j<n;j++)
				B[i][j]-=tmp2*B[k][j];//可以进行向量化，用SIMD 扩展指令进行并行优化
			B[i][k]=0;
        }
    }
}
void Gauss_para2(int n){ //对第二个部分进行向量化的SSE并行算法
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float tmp=C[k][k];
		for(j=k;j<n;j++)
			C[k][j]/=tmp;//可以进行向量化，用SIMD 扩展指令进行并行优化

        for(i=k+1;i<n;i++)
        {
            float32_t tmp2[4]={C[i][k],C[i][k],C[i][k],C[i][k]};
            t1=vld1q_f32(tmp2);
            for(j=k+1;j+4<=n;j+=4)
            {
                t2=vld1q_f32(C[k]+j);
                t3=vld1q_f32(C[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(C[i]+j,t3);
            }
            for(j=j;j<n;j++)
                C[i][j]-=C[i][k]*C[k][j];
            C[i][k]=0;
        }
    }
}
void Gauss_para(int n)
{
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float32_t tmp[4]={D[k][k],D[k][k],D[k][k],D[k][k]};
        t1=vld1q_f32(tmp); //加载到t1向量寄存器
        for(j=k+1;j+4<=n;j+=4)
        {
            t2=vld1q_f32(D[k]+j); //把内存中从B[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=vdivq_f32(t2,t1); //相除结果放到t3寄存器
            vst1q_f32(D[k]+j,t3); //把t3寄存器的值放回内存
        }
        for(j;j<n;j++) //处理剩下的不能被4整除的
            D[k][j]/=D[k][k];
        D[k][k]=1.0;
        //以上完成了对第一个部分的向量化

        for(i=k+1;i<n;i++)
        {
            float32_t tmp2[4]={D[i][k],D[i][k],D[i][k],D[i][k]};
            t1=vld1q_f32(tmp2);
            for(j=k+1;j+4<=n;j+=4)
            {
                t2=vld1q_f32(D[k]+j);
                t3=vld1q_f32(D[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(D[i]+j,t3);
            }
            for(j=j;j<n;j++)
                D[i][j]-=D[i][k]*D[k][j];
            D[i][k]=0;
        }
    }
}
void Print(int n,float m[][2000]){//打印结果
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++)
			cout<<m[i][j]<<" ";
		cout<<endl;
	}
}
int main(){
	int n,N,count,cycle;
	N=1024;
	struct timeval beg1,end1,beg2,end2;
	for(n=2;n<=N;n*=2){
		Initialize(n);
		count=1;
		if(n<=30)
            cycle=1000;
        else
            if(n<=70)
                cycle=100;
            else
                if(n>=300)
                    cycle=10;

        gettimeofday(&beg1, NULL);
		while(count<cycle){
			Gauss_normal(n);
            count++;}
        gettimeofday(&end1, NULL);
		cout <<n<<" "<<count<<" "<< (long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec;

        count=1;
		gettimeofday(&beg2, NULL);
		while(count<cycle){
			Gauss_para(n);
            count++;}
        gettimeofday(&end2, NULL);
		cout <<" "<< (long long)1000000*end2.tv_sec+(long long)end2.tv_usec- (long long)1000000*beg2.tv_sec-(long long)beg2.tv_usec<<"ms"<<endl;
	}
}
