#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

void splitBGRLineInterleave( const Mat& src, Mat& dest);

void set1DSpaceKernel45(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle);
void set1DSpaceKernel135(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle);
void setSpaceKernel(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle);

void set1DSpaceKernel135(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle);
void set1DSpaceKernel45(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle);
void setSpaceKernel(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle);

void jointBilateralFilter_direction_8u( const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, cv::Size kernelSize , double sigma_color, double sigma_space, int borderType,  int direction, bool isRectangle);

enum
{
	FILTER_DEFAULT = 0,
	FILTER_CIRCLE,
	FILTER_RECTANGLE,
	FILTER_SEPARABLE,
	FILTER_SLOWEST,// for just comparison.
};

enum SeparableMethod
{
	DUAL_KERNEL_HV=0,
	DUAL_KERNEL_VH,
	DUAL_KERNEL_HVVH,
	DUAL_KERNEL_CROSS,
	DUAL_KERNEL_CROSSCROSS,
};


void bilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=FILTER_DEFAULT, int borderType=cv::BORDER_REPLICATE);
void jointBilateralFilter(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=FILTER_DEFAULT, int borderType=cv::BORDER_REPLICATE);

void separableBilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, double alpha, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
void separableJointBilateralFilter(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, double alpha, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);