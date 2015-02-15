#include "separableEdgePreservingFilter.h"
#include "util.h"
using namespace std;
double psnr(Mat& src1, Mat& src2)
{
	Mat y1,y2;
	cvtColor(src1,y1,CV_BGR2GRAY);
	cvtColor(src2,y2,CV_BGR2GRAY);
	return PSNR(y1,y2);
}

#include <fstream>

void test()
{
	float sigma_space = 16.33;
	int r = cvRound(sigma_space*3.0)/2;
	int d = 2*r+1;
	
	
	double data[130][15];
	for (int j=0;j<130;j++)
	for (int i=0;i<15;i++) 
		data[j][i]=0.0;


	
for (int n=1;n<25;n++)
	{
		cout<<n<<endl;
		ofstream out(format("out%02d.csv",n));
		Mat src = imread(format("img/kodim%02d.png",n));
		Mat ref;
		Mat dest;
		out<<"sigma "<<"naive "<<"1.0 "<<"0.9 "<<"0.8 "<<"0.7 "<<"0.6 "<<"0.5 "<<endl;
		for(int i=1;i<30;i+=2)
		{
			int count=0;double sn=0.0;
			float sigma_color = 60;

			float sigma_space = i;
			int r = cvRound(sigma_space*3.0)/2;
			int d = 2*r+1;

			out<<i<<" ";
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);

			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, FILTER_SEPARABLE);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			int rate = 100;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 90;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 80;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 70;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 60;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 50;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			out<<endl;
		}
	}

	ofstream out(format("ave_space.csv"));
	for(int i=1;i<30;i+=2)
	{
		int count=0;
		float sigma_color = i;
		out<<i<<" ";
		
		out<<data[i][count++]/24.0<<" ";//naive
		out<<data[i][count++]/24.0<<" ";//1
		out<<data[i][count++]/24.0<<" ";//0.9
		out<<data[i][count++]/24.0<<" ";//0.8
		out<<data[i][count++]/24.0<<" ";//0.7
		out<<data[i][count++]/24.0<<" ";//0.6
		out<<data[i][count++]/24.0<<" ";//0.5
		
		out<<endl;
	}

	/*
	for (int n=1;n<25;n++)
	{
		cout<<n<<endl;
		ofstream out(format("out%02d.csv",n));
		Mat src = imread(format("img/kodim%02d.png",n));
		Mat ref;
		Mat dest;
		out<<"sigma "<<"naive "<<"1.0 "<<"0.9 "<<"0.8 "<<"0.7 "<<"0.6 "<<"0.5 "<<endl;
		for(int i=5;i<130;i+=5)
		{
			int count=0;double sn=0.0;
			float sigma_color = i;
			out<<i<<" ";
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);

			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, FILTER_SEPARABLE);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			int rate = 100;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 90;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 80;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 70;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 60;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			rate = 50;
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, rate/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			out<<endl;
		}
	}

	ofstream out(format("ave.csv"));
	for(int i=10;i<130;i+=5)
	{
		int count=0;
		float sigma_color = i;
		out<<i<<" ";
		
		out<<data[i][count++]/24.0<<" ";//naive
		out<<data[i][count++]/24.0<<" ";//1
		out<<data[i][count++]/24.0<<" ";//0.9
		out<<data[i][count++]/24.0<<" ";//0.8
		out<<data[i][count++]/24.0<<" ";//0.7
		out<<data[i][count++]/24.0<<" ";//0.6
		out<<data[i][count++]/24.0<<" ";//0.5
		
		out<<endl;
	}
	*/
	/*
	for (int n=1;n<25;n++)
	{
		cout<<n<<endl;
		ofstream out(format("out%02d.csv",n));
		Mat src = imread(format("img/kodim%02d.png",n));
		Mat ref;
		Mat dest;
		//out<<"sigma "<<"15-4 "<<"15-16"<<"15-32"<<"60-4 "<<"60-16 "<<"60-32 "<<"1000-16 "<<endl;
		for(int i=0;i<=100;i+=5)
		{
			cout<<i<<endl;
			int count=0;double sn=0.0;

			out<<i<<" ";

			float sigma_color = 15.0;
			float sigma_space = 4;
			int r = cvRound(sigma_space*3.0)/2;
			int d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			sigma_color = 15;
			sigma_space = 16;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;


			sigma_color = 15.0;
			sigma_space = 32;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;


			sigma_color = 30.0;
			sigma_space = 4;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;


			sigma_color = 30.0;
			sigma_space = 16;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;


			sigma_color = 30.0;
			sigma_space = 32;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;


			sigma_color = 120.0;
			sigma_space = 4;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			sigma_color = 120.0;
			sigma_space = 16;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;

			sigma_color = 120.0;
			sigma_space = 32;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;


			sigma_color = 6000.0;
			sigma_space = 16;
			r = cvRound(sigma_space*3.0)/2;
			d = 2*r+1;
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space, FILTER_RECTANGLE);
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, i/100.0,DUAL_KERNEL_HV);
			sn=psnr(ref, dest);
			out<<sn<<" ";
			data[i][count++]+=sn;


			out<<endl;
		}
	}
	ofstream out(format("ave2.csv"));
	for(int i=0;i<=100;i+=5)
	{
		int count=0;
		float sigma_color = i;
		out<<i<<" ";
		
		for(int n=0;n<10;n++)
			out<<data[i][count++]/24.0<<" ";//naive
		
		
		out<<endl;
	}
	*/
}