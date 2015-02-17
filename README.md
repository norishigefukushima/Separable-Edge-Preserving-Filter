# Separable-Edge-Preserving-Filter


This is implimentation of the paper:
N. Fukushima, S. Fujita, Y. Ishibashi, "Switching Dual Kernels for Separable Edge-Preserving Filtering," in Proc. ICASSP2015, Apr. 2015. 
http://fukushima.web.nitech.ac.jp/research/separable.html

    @inproceedings{fukushima2015icassp,
     author  = {N. Fukushima and S. Fujita and Y. Ishibashi},
     title   = {Switching Dual Kernels  for Separable Edge-Preserving Filtering},
     booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
     year    = {2015},
    }

The code is a subset of OpenCP(https://github.com/norishigefukushima/OpenCP). 

    enum SeparableMethod
    {
    	DUAL_KERNEL_HV=0,
    	DUAL_KERNEL_VH,
    	DUAL_KERNEL_HVVH,
    	DUAL_KERNEL_CROSS,
    	DUAL_KERNEL_CROSSCROSS,
    };
    

    void separableBilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, double alpha, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
    void separableJointBilateralFilter(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, double alpha, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
    void separableNonLocalMeansFilter(Mat& src, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma=-1.0, double alpha=1.0, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
    void separableNonLocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma=-1.0, double alpha=1.0, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
    void separableDualBilateralFilter(const Mat& src, const Mat& guide, Mat& dst, Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1=1.0, double alpha2=1.0, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
    void separableDualBilateralFilter(const Mat& src, const Mat& guide, Mat& dst, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1=1.0, double alpha2=1.0, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
    void separableJointDualBilateralFilter(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1=1.0, double alpha2=1.0, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);
    void separableJointDualBilateralFilter(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1=1.0, double alpha2=1.0, int method=DUAL_KERNEL_HV, int borderType=cv::BORDER_REPLICATE);

