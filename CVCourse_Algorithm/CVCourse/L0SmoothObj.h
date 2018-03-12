#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;



// 该类用于并行完成src_中每一层矩阵的离散傅里叶变换
class ParallelDft : public ParallelLoopBody
{
private:
	vector<Mat> src_;
public:
	ParallelDft(vector<Mat> &s){
		src_ = s;
	}
	void operator() (const Range& range) const{
		for (int i = range.start; i != range.end; i++){
			dft(src_[i], src_[i]);
		}
	}
};
// 离散傅里叶逆变换
class ParallelIdft : public ParallelLoopBody
{
private:
	vector<Mat> src_;
public:
	ParallelIdft(vector<Mat> &s){
		src_ = s;
	}
	void operator() (const Range& range) const{
		for (int i = range.start; i != range.end; i++){
			idft(src_[i], src_[i], DFT_SCALE);
		}
	}
};

// 该类用于并行完成numer_逐元素除以denom_的计算，numer_与denom_尺寸相同且可以为多层
class ParallelDivComplexByReal : public ParallelLoopBody
{
private:
	vector<Mat> numer_;
	vector<Mat> denom_;
	vector<Mat> dst_;

public:
	ParallelDivComplexByReal(vector<Mat> &numer, vector<Mat> &denom, vector<Mat> &dst)
	{
		numer_ = numer;
		denom_ = denom;
		dst_ = dst;
	}
	void operator() (const Range& range) const;
};


class CL0SmoothObj
{
public:
	CL0SmoothObj() {};
	~CL0SmoothObj() {};

	void L0Smooth(InputArray src, OutputArray dst, double lambda, double kappa);
private:
	// 补充虚部后进行离散傅里叶变换
	void fft(InputArray src, OutputArray dst);
	// 将src向右下平移（右shift_x个像素，下shift_y个像素），得到dst
	void shift(InputArray src, OutputArray dst, int shift_x, int shift_y);
	// 仿写matlab函数psf2otf（点扩散函数src转为光学传递函数dst）
	void psf2otf(InputArray src, OutputArray dst, int height, int width);
	// src是复数矩阵（m*n*2），返回每个元素模的平方构成的矩阵
	Mat pow2absComplex(InputArray src);

	// 并行执行多层矩阵中每一层的离散傅里叶变换
	void dftMultiChannel(InputArray src, vector<Mat> &dst);
	// 并行执行多层矩阵中每一层的离散傅里叶逆变换
	void idftMultiChannel(const vector<Mat> &src, OutputArray dst);

	// 并行执行多层复数矩阵的除法（逐元素相除）
	void divComplexByRealMultiChannel(vector<Mat> &numer,
		vector<Mat> &denom, vector<Mat> &dst);
};

