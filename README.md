# Separable-Edge-Preserving-Filter
http://fukushima.web.nitech.ac.jp/research/separable.html

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
