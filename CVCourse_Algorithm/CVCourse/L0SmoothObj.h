#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;



// �������ڲ������src_��ÿһ��������ɢ����Ҷ�任
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
// ��ɢ����Ҷ��任
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

// �������ڲ������numer_��Ԫ�س���denom_�ļ��㣬numer_��denom_�ߴ���ͬ�ҿ���Ϊ���
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
	// �����鲿�������ɢ����Ҷ�任
	void fft(InputArray src, OutputArray dst);
	// ��src������ƽ�ƣ���shift_x�����أ���shift_y�����أ����õ�dst
	void shift(InputArray src, OutputArray dst, int shift_x, int shift_y);
	// ��дmatlab����psf2otf������ɢ����srcתΪ��ѧ���ݺ���dst��
	void psf2otf(InputArray src, OutputArray dst, int height, int width);
	// src�Ǹ�������m*n*2��������ÿ��Ԫ��ģ��ƽ�����ɵľ���
	Mat pow2absComplex(InputArray src);

	// ����ִ�ж�������ÿһ�����ɢ����Ҷ�任
	void dftMultiChannel(InputArray src, vector<Mat> &dst);
	// ����ִ�ж�������ÿһ�����ɢ����Ҷ��任
	void idftMultiChannel(const vector<Mat> &src, OutputArray dst);

	// ����ִ�ж�㸴������ĳ�������Ԫ�������
	void divComplexByRealMultiChannel(vector<Mat> &numer,
		vector<Mat> &denom, vector<Mat> &dst);
};

