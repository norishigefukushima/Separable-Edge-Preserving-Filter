#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;


class ConsoleImage
{
private:
	int count;
	string windowName;
	std::vector<std::string> strings;
	bool isLineNumber;
public:
	void setIsLineNumber(bool isLine = true);
	bool getIsLineNumber();
	cv::Mat show;

	void init(Size size, string wname);
	ConsoleImage();
	ConsoleImage(cv::Size size, string wname = "console");
	~ConsoleImage();

	void printData();
	void clear();

	void operator()(string src);
	void operator()(const char *format, ...);
	void operator()(cv::Scalar color, const char *format, ...);

	void flush(bool isClear=true);
};



enum
{
	TIME_AUTO=0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};
class CalcTime
{
	int64 pre;
	string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode();
	vector<string> lap_mes;
public:
	
	void start();
	void setMode(int mode);
	void setMessage(string src);
	void restart();
	double getTime();
	void show();
	void show(string message);
	void lap(string message);
	void init(string message, int mode, bool isShow);

	CalcTime(string message, int mode=TIME_AUTO, bool isShow=true);
	CalcTime();

	~CalcTime();
};

void alphaBlend(const Mat& src1, const Mat& src2, double alpha, Mat& dest);
void guiAlphaBlend(const Mat& src1, const Mat& src2);
void showMatInfo(InputArray src_, string name="Mat");

double SSIM(Mat& src, Mat& ref, double sigma = 1.5);
double calcTV(Mat& src);

void diffshow(string wname, InputArray src, InputArray ref, const float scale=1.f);