#include<iostream>
#include <windows.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
#include<ammintrin.h>
using namespace std;
float A[2000][2000],B[2000][2000],C[2000][2000],D[2000][2000];
void Initialize(int n)//��ʼ��
{
	int i,j,k;
	for(i=0;i<n;i++)
	{
		for(j=0;j<i;j++){

			B[i][j]=0;
			C[i][j]=0;
			D[i][j]=0;
		}

		B[i][i]=1.0;
		C[i][i]=1.0;
		D[i][i]=1.0;
		for(j=i+1;j<n;j++){
			B[i][j]=rand();//������Ԫ�س�ʼ��Ϊ�����
			C[i][j]=B[i][j];
			D[i][j]=B[i][j];
		}
	}
	for(k=0;k<n;k++)
		for(i=k+1;i<n;i++)
			for(j=0;j<n;j++){
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

void Gauss_SSE(int n)
{
    int i,j,k;
    __m128 t1,t2,t3,t4; //����4�������Ĵ���
    for(k=0;k<n;k++)
    {
        float tmp[4]={B[k][k],B[k][k],B[k][k],B[k][k]};
        t1=_mm_loadu_ps(tmp); //���ص�t1�����Ĵ���
        for(j=k+1;j+4<=n;j+=4)
        {
            t2=_mm_loadu_ps(B[k]+j); //���ڴ��д�B[k][j]��ʼ���ĸ������ȸ��������ص�t2�Ĵ���
            t3=_mm_div_ps(t2,t1); //�������ŵ�t3�Ĵ���
            _mm_storeu_ps(B[k]+j,t3); //��t3�Ĵ�����ֵ�Ż��ڴ�
        }
        for(j;j<n;j++) //����ʣ�µĲ��ܱ�4������
            B[k][j]/=B[k][k];
        B[k][k]=1.0;
        //��������˶Ե�һ�����ֵ�������

        for(i=k+1;i<n;i++)
        {
            float tmp2[4]={B[i][k],B[i][k],B[i][k],B[i][k]};
            t1=_mm_loadu_ps(tmp2);
            for(j=k+1;j+4<=n;j+=4)
            {
                t2=_mm_loadu_ps(B[k]+j);
                t3=_mm_loadu_ps(B[i]+j);
                t4=_mm_mul_ps(t1,t2);
                t3=_mm_sub_ps(t3,t4);
                _mm_storeu_ps(B[i]+j,t3);
            }
            for(j=j;j<n;j++)
                B[i][j]-=B[i][k]*B[k][j];
            B[i][k]=0;
        }
    }
}
void Gauss_AVX(int n)
{
    int i,j,k;
    __m256_u t1,t2,t3,t4; //����4�������Ĵ���
    for(k=0;k<n;k++)
    {
        float tmp[8]={C[k][k],C[k][k],C[k][k],C[k][k],C[k][k],C[k][k],C[k][k],C[k][k]};
        t1=_mm256_loadu_ps(tmp); //���ص�t1�����Ĵ���
        for(j=k+1;j+8<=n;j+=8)
        {
            t2=_mm256_loadu_ps(C[k]+j); //���ڴ��д�B[k][j]��ʼ���ĸ������ȸ��������ص�t2�Ĵ���
            t3=_mm256_div_ps(t2,t1); //�������ŵ�t3�Ĵ���
            _mm256_storeu_ps(C[k]+j,t3); //��t3�Ĵ�����ֵ�Ż��ڴ�
        }
        for(j;j<n;j++) //����ʣ�µĲ��ܱ�4������
            C[k][j]/=C[k][k];
        C[k][k]=1.0;
        //��������˶Ե�һ�����ֵ�������

        for(i=k+1;i<n;i++)
        {
            float tmp2[8]={C[i][k],C[i][k],C[i][k],C[i][k],C[i][k],C[i][k],C[i][k],C[i][k]};
            t1=_mm256_loadu_ps(tmp2);
            for(j=k+1;j+8<=n;j+=8)
            {
                t2=_mm256_loadu_ps(C[k]+j);
                t3=_mm256_loadu_ps(C[i]+j);
                t4=_mm256_mul_ps(t1,t2);
                t3=_mm256_sub_ps(t3,t4);
                _mm256_storeu_ps(C[i]+j,t3);
            }
            for(j=j;j<n;j++)
                C[i][j]-=C[i][k]*C[k][j];
            C[i][k]=0;
        }
    }
}
void Gauss_AVX512(int n)
{
    int i,j,k;
    __m512 t1,t2,t3,t4; //����4�������Ĵ���
    for(k=0;k<n;k++)
    {
        float tmp[16]={D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k],D[k][k]};
        t1=_mm512_loadu_ps(tmp); //���ص�t1�����Ĵ���
        for(j=k+1;j+16<=n;j+=16)
        {
            t2=_mm512_loadu_ps(D[k]+j); //���ڴ��д�B[k][j]��ʼ���ĸ������ȸ��������ص�t2�Ĵ���
            t3=_mm512_div_ps(t2,t1); //�������ŵ�t3�Ĵ���
            _mm512_storeu_ps(D[k]+j,t3); //��t3�Ĵ�����ֵ�Ż��ڴ�
        }
        for(j;j<n;j++) //����ʣ�µĲ��ܱ�4������
            D[k][j]/=D[k][k];
        D[k][k]=1.0;
        //��������˶Ե�һ�����ֵ�������

        for(i=k+1;i<n;i++)
        {
            float tmp2[16]={D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k],D[i][k]};
            t1=_mm512_loadu_ps(tmp2);
            for(j=k+1;j+16<=n;j+=16)
            {
                t2=_mm512_loadu_ps(D[k]+j);
                t3=_mm512_loadu_ps(D[i]+j);
                t4=_mm512_mul_ps(t1,t2);
                t3=_mm512_sub_ps(t3,t4);
                _mm512_storeu_ps(D[i]+j,t3);
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
	LARGE_INTEGER t1,t2,tc1,t3,t4,tc2,t5,t6,tc3;
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

		QueryPerformanceFrequency(&tc1);
		QueryPerformanceCounter(&t1);
		while(count<cycle){
			Gauss_SSE(n);
            count++;}
        QueryPerformanceCounter(&t2);
		cout<<n<<" "<<count<<" "<<((t2.QuadPart - t1.QuadPart)*1000.0 / tc1.QuadPart);

        count=1;
		QueryPerformanceFrequency(&tc2);
		QueryPerformanceCounter(&t3);
		while(count<cycle){
			Gauss_AVX(n);
            count++;}
        QueryPerformanceCounter(&t4);
		cout<<" "<<((t4.QuadPart - t3.QuadPart)*1000.0 / tc2.QuadPart);

		count=1;
		QueryPerformanceFrequency(&tc3);
		QueryPerformanceCounter(&t5);
		while(count<cycle){
			Gauss_AVX512(n);
            count++;}
        QueryPerformanceCounter(&t6);
		cout<<" "<<((t6.QuadPart - t5.QuadPart)*1000.0 / tc3.QuadPart);
	}
}
