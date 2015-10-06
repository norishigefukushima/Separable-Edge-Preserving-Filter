#include <opencv2/opencv.hpp>
#include "util.h"
#include "opencp.hpp"

using namespace cv;
using namespace std;

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_imgcodecs"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_imgcodecs"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER".lib")
#endif

void onMouse( int event, int x_, int y_, int flag, void* data)
{
	Point* pt = (Point*) data;
	if( flag == EVENT_FLAG_LBUTTON)
	{
		pt->x=x_;
		pt->y=y_;
	}
}

class VizKernel
{
	Mat ref;
	Mat gray;
	Mat show;
	Size imsize;
	string wname;
	Point pt;

	int alpha;

	bool isGrid;
	int color;


public:
	Mat kernel;

	void bilateralWeight(Mat& src, Size size, float sigma_color, float sigma_space)
	{
		const double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
		const double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

		kernel.setTo(0);
		const int rh = size.width/2;
		const int rv = size.height/2;

		const double cb = (double)src.at<uchar>(pt.y,3*pt.x+0);
		const double cg = (double)src.at<uchar>(pt.y,3*pt.x+1);
		const double cr = (double)src.at<uchar>(pt.y,3*pt.x+2);

		for(int j=-rv;j<=rv;j++)
		{
			for(int i=-rv;i<=rv;i++)
			{
				if(pt.x+i>=0 && pt.x+i<imsize.width &&
					pt.y+j>=0 && pt.y+j<imsize.height)
				{

					const double r = sqrt(i*i+j*j);
					const double tb = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+0);
					const double tg = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+1);
					const double tr = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+2);
					const double cd = abs(cb-tb) + abs(cg-tg) + abs(cr-tr);

					double s = exp( r* r*gauss_space_coeff);
					double c = exp(cd*cd*gauss_color_coeff);

					kernel.at<float>(pt.y+j,pt.x+i) = (float)(c*s);
				}
			}
		}
	}
	void bilateralWeightSP_OldHV(Mat& src, Mat& srcH, Size size, float sigma_color, float sigma_space)
	{
		const double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
		const double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

		kernel.setTo(0);
		const int rh = size.width/2;
		const int rv = size.height/2;


		const double ccbH = (double)srcH.at<uchar>(pt.y,3*pt.x+0);
		const double ccgH = (double)srcH.at<uchar>(pt.y,3*pt.x+1);
		const double ccrH = (double)srcH.at<uchar>(pt.y,3*pt.x+2);

		for(int j=-rv;j<=rv;j++)
		{
			for(int i=-rv;i<=rv;i++)
			{
				if(pt.x+i>=0 && pt.x+i<imsize.width &&
					pt.y+j>=0 && pt.y+j<imsize.height)
				{
					//V
					const double cbH = (double)srcH.at<uchar>(pt.y+j, 3*pt.x+0);
					const double cgH = (double)srcH.at<uchar>(pt.y+j, 3*pt.x+1);
					const double crH = (double)srcH.at<uchar>(pt.y+j, 3*pt.x+2);

					const double ccd = abs(cbH-ccbH) + abs(cgH-ccgH) + abs(crH-ccrH);
					double sv = exp(j*j*gauss_space_coeff);
					double cv = exp(ccd*ccd*gauss_color_coeff);

					//H
					const double cb = (double)src.at<uchar>(pt.y+j, 3*pt.x+0);
					const double cg = (double)src.at<uchar>(pt.y+j, 3*pt.x+1);
					const double cr = (double)src.at<uchar>(pt.y+j, 3*pt.x+2);

					const double tb = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+0);
					const double tg = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+1);
					const double tr = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+2);
					const double cd = abs(cb-tb) + abs(cg-tg) + abs(cr-tr);

					double s = exp(i*i*gauss_space_coeff);
					double c = exp(cd*cd*gauss_color_coeff);

					kernel.at<float>(pt.y+j,pt.x+i) = (float)(c*s*sv*cv);
				}
			}
		}
	}
	void bilateralWeightSP_DualHV(Mat& src, Size size, float sigma_color, float sigma_space, float rate)
	{
		const double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
		const double gauss_color_coeff2 = -0.5/(sigma_color*rate*sigma_color*rate);
		const double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

		kernel.setTo(0);
		const int rh = size.width/2;
		const int rv = size.height/2;


		const double ccb = (double)src.at<uchar>(pt.y, 3*pt.x+0);
		const double ccg = (double)src.at<uchar>(pt.y, 3*pt.x+1);
		const double ccr = (double)src.at<uchar>(pt.y, 3*pt.x+2);

		for(int j=-rv;j<=rv;j++)
		{
			for(int i=-rv;i<=rv;i++)
			{
				if(pt.x+i>=0 && pt.x+i<imsize.width &&
					pt.y+j>=0 && pt.y+j<imsize.height)
				{
					//V
					const double cb = (double)src.at<uchar>(pt.y+j, 3*pt.x+0);
					const double cg = (double)src.at<uchar>(pt.y+j, 3*pt.x+1);
					const double cr = (double)src.at<uchar>(pt.y+j, 3*pt.x+2);

					const double ccd = abs(cb-ccb) + abs(cg-ccg) + abs(cr-ccr);
					double sv = exp(j*j*gauss_space_coeff);
					double cv = exp(ccd*ccd*gauss_color_coeff2);

					//H
					const double tb = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+0);
					const double tg = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+1);
					const double tr = (double)src.at<uchar>(pt.y+j,3*(pt.x+i)+2);
					const double cd = abs(cb-tb) + abs(cg-tg) + abs(cr-tr);

					double s = exp(i*i*gauss_space_coeff);
					double c = exp(cd*cd*gauss_color_coeff);

					kernel.at<float>(pt.y+j,pt.x+i) = (float)(c*s*sv*cv);
				}
			}
		}
	}
	void bilateralWeightSP_DualVH(Mat& src, Size size, float sigma_color, float sigma_space, float rate)
	{
		const double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
		const double gauss_color_coeff2 = -0.5/(sigma_color*rate*sigma_color*rate);
		const double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

		kernel.setTo(0);
		const int rh = size.width/2;
		const int rv = size.height/2;


		const double ccb = (double)src.at<uchar>(pt.y, 3*pt.x+0);
		const double ccg = (double)src.at<uchar>(pt.y, 3*pt.x+1);
		const double ccr = (double)src.at<uchar>(pt.y, 3*pt.x+2);

		for(int i=-rv;i<=rv;i++)
		{
			for(int j=-rv;j<=rv;j++)
			{
				if(pt.x+i>=0 && pt.x+i<imsize.width &&
					pt.y+j>=0 && pt.y+j<imsize.height)
				{
					//H
					const double cb = (double)src.at<uchar>(pt.y, 3*(pt.x+i)+0);
					const double cg = (double)src.at<uchar>(pt.y, 3*(pt.x+i)+1);
					const double cr = (double)src.at<uchar>(pt.y, 3*(pt.x+i)+2);

					const double ccd = abs(cb-ccb) + abs(cg-ccg) + abs(cr-ccr);
					double sv = exp(i*i*gauss_space_coeff);
					double cv = exp(ccd*ccd*gauss_color_coeff2);

					//V
					const double tb = (double)src.at<uchar>(pt.y+j,3*(pt.x)+0);
					const double tg = (double)src.at<uchar>(pt.y+j,3*(pt.x)+1);
					const double tr = (double)src.at<uchar>(pt.y+j,3*(pt.x)+2);
					const double cd = abs(cb-tb) + abs(cg-tg) + abs(cr-tr);

					double s = exp(j*j*gauss_space_coeff);
					double c = exp(cd*cd*gauss_color_coeff);

					kernel.at<float>(pt.y+j,pt.x+i) = (float)(c*s*sv*cv);
				}
			}
		}
	}

	void showKernel(int key=0)
	{
		double minv,maxv;
		minMaxLoc(kernel, &minv, &maxv);
		kernel-=(float)minv;
		kernel*=(float)(1.0/maxv);
		//normalize(kernel, kernel, 1.0,0.0, NORM_MINMAX);


		kernel.convertTo(gray,CV_8U,255);
		applyColorMap(gray,show,color);

		alphaBlend(ref,show,alpha/100.0,show);

		if(isGrid)
		{
			line(show,Point(0,pt.y),Point(imsize.width,pt.y), Scalar(0,255,0,0));
			line(show,Point(pt.x,0),Point(pt.x,imsize.height), Scalar(0,255,0,0));
		}

		imshow(wname,show);

		setPoint();

		if(key=='g')
		{
			isGrid = isGrid ? false:true;
		}
		if(key=='f')
		{
			alpha = (alpha==0) ? 100 : 0;
		}
	}

	void init(string window_name, Size size)
	{

		isGrid = true;

		imsize = size;
		wname = window_name;

		kernel = Mat::zeros(size,CV_32F);
		ref = Mat::zeros(size,CV_8UC3);
		namedWindow(wname);

		pt = Point(size.width/2,size.height/2);
		alpha = 0;
		createTrackbar("a",wname,&alpha,100);
		createTrackbar("x",wname,&pt.x,size.width-1);
		createTrackbar("y",wname,&pt.y,size.height-1);


		color=1;
		createTrackbar("color",wname,&color,11);


		setMouseCallback(wname, onMouse, &pt);
		updateTrackbar();
	}

	VizKernel(string window_name, Mat& src)
	{
		init(window_name, src.size());
		setImage(src);
	}

	VizKernel(string window_name, Size size)
	{
		init(window_name, size);
	}

	void setImage(Mat& src)
	{
		src.copyTo(ref);
	}

	void updateTrackbar()
	{
		setTrackbarPos("x",wname,pt.x);
		setTrackbarPos("y",wname,pt.y);
		setTrackbarPos("a",wname,alpha);
	}
	void vcopy()
	{
		const int x = pt.x;
		for(int j=0;j<imsize.height;j++)
		{
			const float val = kernel.at<float>(j,x);
			float* data = kernel.ptr<float>(j);
			for(int i=0;i<imsize.width;i++)
				data[i]=val;
		}

	}
	void setLine(int d, bool isHorizon)
	{
		kernel.setTo(0);
		int r = d/2;
		if(isHorizon)
		{
			for(int i=-r;i<=r;i++)
			{
				if(pt.x+i>=0 && pt.x+i<imsize.width)
				{
					kernel.at<float>(pt.y,pt.x+i)=1.f;
				}
			}
		}
		else
		{
			for(int i=-r;i<=r;i++)
			{
				if(pt.y+i>=0 && pt.y+i<imsize.height)
				{
					kernel.at<float>(pt.y+i,pt.x)=1.f;
				}
			}
		}	
	}
	void setPoint()
	{
		kernel.setTo(0);
		kernel.at<float>(pt)=1.f;
	}
};

void guiSeparableBilateralFilterTest(Mat& src)
{
	Mat srcf; src.convertTo(srcf,CV_32F);
	Mat dest;

	string wname = "bilateral filter SP";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 6);
	//int r = 20; createTrackbar("r",wname,&r,200);
	int space = 300; createTrackbar("space",wname,&space,2000);
	int color = 500; createTrackbar("color",wname,&color,2550);
	int rate = 100; createTrackbar("color rate",wname,&rate,100);
	int scale = 10; createTrackbar("scale",wname,&scale,20);


	int x=src.cols/2;
	int y=src.rows/2;
	Mat kernel = Mat::zeros(src.size(),CV_32F);

	class VizKernel vk("kernel",src);
	Mat ref;
	{
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int r = cvRound(sigma_space*3.0)/2;
		int d = 2*r+1;
		bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
	}
	ConsoleImage ci;
	bool isKernelBF = false;
	Mat show;
	int key = 0;
	while(key!='q')
	{
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int r = cvRound(sigma_space*3.0);
		int d = 2*r+1;

		//		double ssims = s/10.0;

		if(key=='r')
		{
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
		}

		if(sw==0)
		{
			CalcTime t("bilateral filter: opencv");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
			vk.bilateralWeight(src, Size(d,d), sigma_color,sigma_space);

		}
		else if(sw==1)
		{
			CalcTime t("bilateral filter: opencv sp");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_SEPARABLE);

			Mat srch;
			bilateralFilter(src, srch, Size(d,1), sigma_color, sigma_space,FILTER_RECTANGLE);
			vk.bilateralWeightSP_OldHV(src, srch,Size(d,d), sigma_color,sigma_space);
		}
		else if(sw==2)
		{
			CalcTime t("bilateral filter: opencv sp HV");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_HV);

			vk.bilateralWeightSP_DualHV(src, Size(d,d), sigma_color,sigma_space,rate/100.f);
		}
		else if(sw==3)
		{
			CalcTime t("bilateral filter: opencv sp VH");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_VH);
			vk.bilateralWeightSP_DualVH(src, Size(d,d), sigma_color,sigma_space,rate/100.f);
			//separableJointBilateralFilter(vk.kernel, srcf, vk.kernel, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_VH);

			//CalcTime t("bilateral filter: opencv sp HVVH");
			//separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_HVVH);
		}
		else if(sw==4)
		{
			CalcTime t("bilateral filter: opencv sp HVVH");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_HVVH);

			vk.bilateralWeightSP_DualHV(src, Size(d,d), sigma_color,sigma_space,rate/100.f);
			Mat temp = vk.kernel.clone();
			vk.bilateralWeightSP_DualVH(src, Size(d,d), sigma_color,sigma_space,rate/100.f);

			alphaBlend(vk.kernel, temp, 0.5, vk.kernel);
			//			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_CROSS);
		}
		else if(sw==5)
		{
			CalcTime t("bilateral filter: opencv sp HVVH");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_CROSSCROSS);
		}
		else if(sw==6)
		{
			//CalcTime t("bilateral filter: opencv sp HVVH");
			//separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate1/100.0,DUAL_KERNEL_CROSSCROSS);
		}

		if(key=='f')
		{
			a = (a==0) ? 100 : 0;
			setTrackbarPos("a",wname,a);
		}
		ci(format("%f dB",PSNR(ref,dest)));


		Mat g1,g2; cvtColor(ref,g1,COLOR_BGR2GRAY); cvtColor(dest,g2,COLOR_BGR2GRAY);
		diffshow("diff", g1, g2, (float)scale);
		//ci(format("%f dB",SSIM(ref,dest,ssims)));
		//ci(format("%f %f",calcTV(dest),calcTV(ref)));


		alphaBlend(ref, dest,a/100.0, show);
		imshow(wname,show);

		ci.flush();
		if(key=='k')
		{
			isKernelBF = isKernelBF ? false:true;
		}
		if(isKernelBF)
		{
			cout<<"BF kernel"<<endl;
			vk.bilateralWeight(src, Size(d,d), sigma_color,sigma_space);
		}
		vk.showKernel(key);
		key = waitKey(1);
	}
}

void guiSeparableNonLocalMeans(Mat& src)
{
	Mat srcf; src.convertTo(srcf,CV_32F);
	Mat dest;

	string wname = "non-local means filter SP";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 6);
	int tr = 1; createTrackbar("tr",wname,&tr,10);
	int sr = 3; createTrackbar("sr",wname,&sr,50);

	int h = 500; createTrackbar("h",wname,&h,2550);
	int rate = 100; createTrackbar("color rate",wname,&rate,100);


	Mat ref;
	{
		float sigma_h = h/10.f;
		nonLocalMeansFilter(src, ref, 2*tr+1, 2*sr+1, sigma_h, 0);
	}
	ConsoleImage ci;
	bool isKernelBF = false;
	Mat show;
	int key = 0;
	while(key!='q')
	{
		float sigma_h = h/10.f;

		//		double ssims = s/10.0;

		if(key=='r')
		{
			nonLocalMeansFilter(src, ref, 2*tr+1, 2*sr+1, sigma_h, 0);
		}

		if(sw==0)
		{
			CalcTime t("non-local means filter: opencv");
			fastNlMeansDenoisingColored(src, dest, sigma_h, sigma_h, 2*tr+1, 2*sr+1);
		}
		else if(sw==1)
		{
			CalcTime t("non-local means filter: my");
			nonLocalMeansFilter(src, dest, 2*tr+1, 2*sr+1, sigma_h, 0);
		}
		else if(sw==2)
		{
			CalcTime t("non-local means filter: separable conv");
			nonLocalMeansFilter(src, dest, 2*tr+1, 2*sr+1, sigma_h, 0, FILTER_SEPARABLE);
		}
		else if(sw==3)
		{
			CalcTime t("non-local means filter: separable prop");
			separableNonLocalMeansFilter(src, dest, 2*tr+1, 2*sr+1, sigma_h, 0, rate/100.0, DUAL_KERNEL_HV);
		}
		/*		else if(sw==3)
		{
		CalcTime t("bilateral filter: opencv sp VH");
		separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_VH);
		vk.bilateralWeightSP_DualVH(src, Size(d,d), sigma_color,sigma_space,rate/100.f);
		//separableJointBilateralFilter(vk.kernel, srcf, vk.kernel, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_VH);

		//CalcTime t("bilateral filter: opencv sp HVVH");
		//separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_HVVH);
		}
		else if(sw==4)
		{
		CalcTime t("bilateral filter: opencv sp HVVH");
		separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_HVVH);

		vk.bilateralWeightSP_DualHV(src, Size(d,d), sigma_color,sigma_space,rate/100.f);
		Mat temp = vk.kernel.clone();
		vk.bilateralWeightSP_DualVH(src, Size(d,d), sigma_color,sigma_space,rate/100.f);

		alphaBlend(vk.kernel, temp, 0.5, vk.kernel);
		//			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_CROSS);
		}
		else if(sw==5)
		{
		CalcTime t("bilateral filter: opencv sp HVVH");
		separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_CROSSCROSS);
		}
		else if(sw==6)
		{
		//CalcTime t("bilateral filter: opencv sp HVVH");
		//separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate1/100.0,DUAL_KERNEL_CROSSCROSS);
		}*/

		if(key=='f')
		{
			a = (a==0) ? 100 : 0;
			setTrackbarPos("a",wname,a);
		}
		ci(format("%f dB",PSNR(ref,dest)));


		//Mat g1,g2;
		//cvtColor(ref,g1,COLOR_BGR2GRAY);
		//cvtColor(dest,g2,COLOR_BGR2GRAY);
		//diffshow("diff", g1, g2, (float)scale);
		//ci(format("%f dB",SSIM(ref,dest,ssims)));
		//ci(format("%f %f",calcTV(dest),calcTV(ref)));

		alphaBlend(ref, dest,a/100.0, show);
		imshow(wname,show);

		ci.flush();

		key = waitKey(1);
	}
}

void guiSeparableDualBilateralFilter(Mat& src, Mat& guide)
{
	Mat srcf; src.convertTo(srcf,CV_32F);
	Mat dest;

	string wname = "dual bilateral filter SP";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 4; createTrackbar("switch",wname,&sw, 6);
	int r = 20; createTrackbar("r",wname,&r,20);
	int sigma_s = 50; createTrackbar("sigma_s",wname,&sigma_s,300);
	int sigma_c1 = 1500; createTrackbar("sigma_c1",wname,&sigma_c1,2550);
	int sigma_c2 = 150; createTrackbar("sigma_c2",wname,&sigma_c2,2550);
	int a1 = 100; createTrackbar("a1",wname,&a1,100);
	int a2 = 100; createTrackbar("a2",wname,&a2,100);

	int scale = 2; createTrackbar("scale",wname,&scale,20);
	Mat ref;
	{
		double ss = sigma_s/10.0;
		double sc1 = sigma_c1/10.0;
		double sc2 = sigma_c2/10.0;
		dualBilateralFilter(src, guide, ref,Size(2*r+1,2*r+1), sc1, sc2, ss, FILTER_DEFAULT);
	}

	ConsoleImage ci;
	
	Mat show;
	int key = 0;
	while(key!='q')
	{
		double ss = sigma_s/10.0;
		double sc1 = sigma_c1/10.0;
		double sc2 = sigma_c2/10.0;

		if(key=='r')
		{
			dualBilateralFilter(src, guide, ref,Size(2*r+1,2*r+1), sc1, sc2, ss, FILTER_DEFAULT);
		}

		if(sw==0)
		{
			CalcTime t("bilateral filter");
			bilateralFilter(src, dest, Size(2*r+1,2*r+1), sc1, ss, FILTER_DEFAULT);
		}
		else if(sw==1)
		{
			CalcTime t("joint bilateral filter");
			jointBilateralFilter(src, guide, dest, Size(2*r+1,2*r+1), sc2, ss, FILTER_DEFAULT);
		}
		else if(sw==2)
		{
			CalcTime t("dual bilateral filter");
			dualBilateralFilter(src, guide, dest, Size(2*r+1,2*r+1), sc1, sc2, ss, FILTER_DEFAULT);
		}
		else if(sw==3)
		{
			CalcTime t("dual bilateral filter: conv sp");
			dualBilateralFilter(src, guide, dest, Size(2*r+1,2*r+1), sc1, sc2, ss, FILTER_SEPARABLE);
		}
		else if(sw==4)
		{
			CalcTime t("dual bilateral filter: prpop sp");
			separableDualBilateralFilter(src, guide, dest, Size(2*r+1, 2*r+1), sc1, sc2, ss, a1/100.0, a2/100.0, DUAL_KERNEL_HV);
		}

		if(key=='f')
		{
			a = (a==0) ? 100 : 0;
			setTrackbarPos("a",wname,a);
		}
		ci(format("%f dB",PSNR(ref,dest)));

		Mat g1,g2; cvtColor(ref,g1,COLOR_BGR2GRAY); cvtColor(dest,g2,COLOR_BGR2GRAY);
		diffshow("diff", g1, g2, (float)scale);
		//ci(format("%f dB",SSIM(ref,dest,ssims)));
		//ci(format("%f %f",calcTV(dest),calcTV(ref)));

		alphaBlend(ref, dest,a/100.0, show);
		imshow(wname,show);

		ci.flush();

		key = waitKey(1);
	}
}
int main(int argc, char** argv)
{
	//Mat img = imread("imgbig/artificial.png");
	Mat img = imread("img/kodim21.png");
	//Mat re;resize(img,re,Size(1024,1024));
	guiSeparableBilateralFilterTest(img);
	//guiSeparableNonLocalMeans(img);
	Mat fls = imread("img/cave-flash.png");
	Mat nfls = imread("img/cave-noflash.png");
	//guiSeparableDualBilateralFilter(nfls,fls);
}
