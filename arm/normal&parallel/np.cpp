#include<iostream>
#include <sys/time.h>
#include<arm_neon.h>
using namespace std;
float A[2000][2000],B[2000][2000],C[2000][2000],D[2000][2000];
void Initialize(int n)//��ʼ��
{
	int i,j,k;
	for(i=0;i<n;i++)
	{
		for(j=0;j<i;j++){
			A[i][j]=0;//������Ԫ�س�ʼ��Ϊ��
			B[i][j]=0;
			C[i][j]=0;
			D[i][j]=0;
		}
		A[i][i]=1.0;//�Խ���Ԫ�س�ʼ��Ϊ1
		B[i][i]=1.0;
		C[i][i]=1.0;
		D[i][i]=1.0;
		for(j=i+1;j<n;j++){
			A[i][j]=rand();//������Ԫ�س�ʼ��Ϊ�����
			B[i][j]=A[i][j];
			C[i][j]=A[i][j];
			D[i][j]=A[i][j];
		}
	}
	for(k=0;k<n;k++)
		for(i=k+1;i<n;i++)
			for(j=0;j<n;j++){
				A[i][j]+=A[k][j];//����ÿһ�е�ֵ����һ�е�ֵ����һ�е�ֵ֮��
				B[i][j]+=B[k][j];
				C[i][j]+=C[k][j];
				D[i][j]+=D[k][j];
			}
}
void Gauss_normal(int n){//�����㷨
	int i,j,k;
	for(k=0;k<n;k++)
    {
		float tmp=A[k][k];
		for(j=k;j<n;j++)
			A[k][j]/=tmp;//���Խ�������������SIMD ��չָ����в����Ż�
		for(i=k+1;i<n;i++)\
		{
			float tmp2=A[i][k];
			for(j=k+1;j<n;j++)
				A[i][j]-=tmp2*A[k][j];//���Խ�������������SIMD ��չָ����в����Ż�
			A[i][k]=0;
		}
	}
}
void Gauss_para1(int n){ //�Ե�һ�����ֽ�����������SSE�����㷨
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //����4�������Ĵ���
    for(k=0;k<n;k++)
    {
        float32_t tmp[4]={B[k][k],B[k][k],B[k][k],B[k][k]};
        t1=vld1q_f32(tmp); //���ص�t1�����Ĵ���
        for(j=k+1;j+4<=n;j+=4)
        {
            t2=vld1q_f32(B[k]+j); //���ڴ��д�B[k][j]��ʼ���ĸ������ȸ��������ص�t2�Ĵ���
            t3=vdivq_f32(t2,t1); //�������ŵ�t3�Ĵ���
            vst1q_f32(B[k]+j,t3); //��t3�Ĵ�����ֵ�Ż��ڴ�
        }
        for(j;j<n;j++) //����ʣ�µĲ��ܱ�4������
            B[k][j]/=B[k][k];
        B[k][k]=1.0;
        //��������˶Ե�һ�����ֵ�������

        for(i=k+1;i<n;i++)
        {
			float tmp2=B[i][k];
			for(j=k+1;j<n;j++)
				B[i][j]-=tmp2*B[k][j];//���Խ�������������SIMD ��չָ����в����Ż�
			B[i][k]=0;
        }
    }
}
void Gauss_para2(int n){ //�Եڶ������ֽ�����������SSE�����㷨
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //����4�������Ĵ���
    for(k=0;k<n;k++)
    {
        float tmp=C[k][k];
		for(j=k;j<n;j++)
			C[k][j]/=tmp;//���Խ�������������SIMD ��չָ����в����Ż�

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
    float32x4_t t1,t2,t3,t4; //����4�������Ĵ���
    for(k=0;k<n;k++)
    {
        float32_t tmp[4]={D[k][k],D[k][k],D[k][k],D[k][k]};
        t1=vld1q_f32(tmp); //���ص�t1�����Ĵ���
        for(j=k+1;j+4<=n;j+=4)
        {
            t2=vld1q_f32(D[k]+j); //���ڴ��д�B[k][j]��ʼ���ĸ������ȸ��������ص�t2�Ĵ���
            t3=vdivq_f32(t2,t1); //�������ŵ�t3�Ĵ���
            vst1q_f32(D[k]+j,t3); //��t3�Ĵ�����ֵ�Ż��ڴ�
        }
        for(j;j<n;j++) //����ʣ�µĲ��ܱ�4������
            D[k][j]/=D[k][k];
        D[k][k]=1.0;
        //��������˶Ե�һ�����ֵ�������

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
void Print(int n,float m[][2000]){//��ӡ���
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
