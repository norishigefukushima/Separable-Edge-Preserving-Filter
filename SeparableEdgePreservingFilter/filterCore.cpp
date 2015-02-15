#include "separableEdgePreservingFilter.h"

//8u
void splitBGRLineInterleave_8u( const Mat& src, Mat& dest)
{

	const int size = src.size().area();
	dest.create(Size(src.cols,src.rows*3),CV_8U);
	const int dstep = src.cols*3;
	const int sstep = src.cols*3;

	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(1);
	uchar* R = dest.ptr<uchar>(2);

	//BGR BGR BGR BGR BGR B
	//GR BGR BGR BGR BGR BG
	//R BGR BGR BGR BGR BGR
	//BBBBBBGGGGGRRRRR shuffle
	const __m128i mask1 = _mm_setr_epi8(0,3,6,9,12,15,1,4,7,10,13,2,5,8,11,14);
	//GGGGGBBBBBBRRRRR shuffle
	const __m128i smask1 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
	const __m128i ssmask1 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

	//GGGGGGBBBBBRRRRR shuffle
	const __m128i mask2 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);
	//const __m128i smask2 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
	const __m128i ssmask2 = _mm_setr_epi8(0,1,2,3,4,11,12,13,14,15,5,6,7,8,9,10);

	//RRRRRRGGGGGBBBBB shuffle -> same mask2
	//__m128i mask3 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);

	//const __m128i smask3 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,6,7,8,9,10);
	//const __m128i ssmask3 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

	const __m128i bmask1 = _mm_setr_epi8
		(255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0);

	const __m128i bmask2 = _mm_setr_epi8
		(255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0);

	const __m128i bmask3 = _mm_setr_epi8
		(255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0);

	const __m128i bmask4 = _mm_setr_epi8
		(255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0);	

	__m128i a,b,c;

	for(int j=0;j<src.rows;j++)
	{
		int i=0;
		for(;i<src.cols;i+=16)
		{
			a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i)),mask1);
			b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i+16)),mask2);
			c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i+32)),mask2);
			_mm_stream_si128((__m128i*)(B+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask1),bmask2));

			a = _mm_shuffle_epi8(a,smask1);
			b = _mm_shuffle_epi8(b,smask1);
			c = _mm_shuffle_epi8(c,ssmask1);
			_mm_stream_si128((__m128i*)(G+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask2));

			a = _mm_shuffle_epi8(a,ssmask1);
			c = _mm_shuffle_epi8(c,ssmask1);
			b = _mm_shuffle_epi8(b,ssmask2);

			_mm_stream_si128((__m128i*)(R+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask4));
		}
		R+=dstep;
		G+=dstep;
		B+=dstep;
		s+=sstep;
	}
}

void splitBGRLineInterleave_32f( const Mat& src, Mat& dest)
{

	const int size = src.size().area();
	dest.create(Size(src.cols,src.rows*3),CV_32F);
	const int dstep = src.cols*3;
	const int sstep = src.cols*3;

	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(1);
	float* R = dest.ptr<float>(2);

	for(int j=0;j<src.rows;j++)
	{
		int i=0;
		for(;i<src.cols;i+=4)
		{
			__m128 a = _mm_load_ps((s+3*i));
			__m128 b = _mm_load_ps((s+3*i+4));
			__m128 c = _mm_load_ps((s+3*i+8));

			__m128 aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,2,3,0));
			aa=_mm_blend_ps(aa,b,4);
			__m128 cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(1,3,2,0));
			aa=_mm_blend_ps(aa,cc,8);
			_mm_stream_ps((B+i),aa);

			aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,2,0,1));
			__m128 bb = _mm_shuffle_ps(b,b,_MM_SHUFFLE(2,3,0,1));
			bb=_mm_blend_ps(bb,aa,1);
			cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(2,3,1,0));
			bb=_mm_blend_ps(bb,cc,8);
			_mm_stream_ps((G+i),bb);

			aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
			bb=_mm_blend_ps(aa,b,2);
			cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(3,0,1,2));
			cc=_mm_blend_ps(bb,cc,12);
			_mm_stream_ps((R+i),cc);

		}
		R+=dstep;
		G+=dstep;
		B+=dstep;
		s+=sstep;
	}
}

void splitBGRLineInterleave_32fcast( const Mat& src, Mat& dest)
{
	Mat a,b;
	src.convertTo(a,CV_32F);
	splitBGRLineInterleave_32f(a,b);
	b.convertTo(dest,src.type());
}

void splitBGRLineInterleave( const Mat& src, Mat& dest)
{
	if(src.type()==CV_MAKE_TYPE(CV_8U,3))
	{
		CV_Assert(src.cols%16==0);
		splitBGRLineInterleave_8u(src,dest);
	}
	else if(src.type()==CV_MAKE_TYPE(CV_32F,3))
	{
		CV_Assert(src.cols%4==0);
		splitBGRLineInterleave_32f(src,dest);
	}
	else
	{
		CV_Assert(src.cols%4==0);
		splitBGRLineInterleave_32fcast(src,dest);
	}
}

void set1DSpaceKernel45(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=j) continue;

			//printf("45: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void set1DSpaceKernel135(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=-j) continue;
			//printf("135: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;

			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;

			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk] = (int)(i*imstep1 + j);
			space_guide_ofs[maxk++] = (int)(i*imstep2 + j);
		}
	}
}

void set1DSpaceKernel45(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle)
	{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=j) continue;

			//printf("45: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk] = (int)(i*imstep1 + j);
			space_guide_ofs[maxk++] = (int)(i*imstep2 + j);
		}
	}
}

void set1DSpaceKernel135(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle)
	{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=-j) continue;

			//printf("135: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk] = (int)(i*imstep1 + j);
			space_guide_ofs[maxk++] = (int)(i*imstep2 + j);
		}
	}
}
