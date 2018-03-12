#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "edgeaware_filters_common.hpp"
using namespace cv;
using namespace std;
using namespace cv::ximgproc::intrinsics;

// 二维矩阵作为操作单元的结构体
template <typename T>
struct SymArray2D
{
	vector<T> vec;
	int sz;

	SymArray2D()
	{
		sz = 0;
	}

	void create(int sz_)
	{
		CV_DbgAssert(sz_ > 0);
		sz = sz_;
		vec.resize(total());
	}

	inline T& operator()(int i, int j)
	{
		CV_DbgAssert(i >= 0 && i < sz && j >= 0 && j < sz);
		if (i < j) std::swap(i, j);
		return vec[i*(i + 1) / 2 + j];
	}

	inline T& operator()(int i)
	{
		return vec[i];
	}

	int total() const
	{
		return sz*(sz + 1) / 2;
	}

	void release()
	{
		vec.clear();
		sz = 0;
	}
};


// 图像数据和基本信息类，便于在并行计算中统一使用
class CGuidedImgObj
{
public:
	InputArray guide;
	InputArray src;
	int radius;
	double eps;
	int h, w;
	int gCnNum;
	int dstDepth;
	vector<Mat> guideCn;
	vector<Mat> guideCnMean;
	SymArray2D<Mat> covarsInv;

public:
	CGuidedImgObj(InputArray guide_, InputArray src_, int radius_, double eps_, int dstDepth_=-1);

	// 将图像像素的id号转为行列号
	void getWalkPattern(int eid, int &cn1, int &cn2) const;

	// 给定窗口的均值滤波函数，便于并行执行
	inline void meanFilter(Mat& src, Mat& dst)
	{
		boxFilter(src, dst, CV_32F, Size(2 * radius + 1, 2 * radius + 1), cv::Point(-1, -1), true, BORDER_REFLECT);
	}
	// 图像类型转换函数，便于并行执行
	inline void convertToWorkType(Mat& src, Mat& dst)
	{
		src.convertTo(dst, CV_32F);
	}
	
};


class CGuidedFilterObj
{
public:
	// 使用图像数据和基本信息对象进行初始化，便于获取和记录图像处理结果
	CGuidedFilterObj(CGuidedImgObj imgObj_);
	~CGuidedFilterObj();

	// 获得滤波结果
	void GetFilterDst(OutputArray dst);
private:
	CGuidedImgObj m_imgObj;

	// 一系列用于执行并行运算的结构体
	// 波段间乘法运算，用于求协方差矩阵
	struct MulChannelsGuide_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		SymArray2D<Mat> &covars;
		MulChannelsGuide_ParBody(CGuidedImgObj& imgObj_, SymArray2D<Mat>& covars_)
			: imgObj(imgObj_),covars(covars_) {}
		void operator () (const Range& range) const;
	};

	struct ComputeCovGuideFromChannelsMul_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		SymArray2D<Mat> &covars;
		ComputeCovGuideFromChannelsMul_ParBody(CGuidedImgObj& imgObj_, SymArray2D<Mat>& covars_)
			: imgObj(imgObj_),covars(covars_) {}
		void operator () (const Range& range) const;
	};

	struct ComputeCovGuideInv_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		SymArray2D<Mat> &covars;
		ComputeCovGuideInv_ParBody(CGuidedImgObj& imgObj_, SymArray2D<Mat>& covars_);
		void operator () (const Range& range) const;
	};

	struct MulChannelsGuideAndSrc_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		vector<vector<Mat> > &cov;
		vector<Mat> &srcCn;
		MulChannelsGuideAndSrc_ParBody(CGuidedImgObj& imgObj_, vector<Mat>& srcCn_, vector<vector<Mat> >& cov_)
			: imgObj(imgObj_),cov(cov_), srcCn(srcCn_) {}

		void operator () (const Range& range) const;
	};


	typedef void(CGuidedImgObj::*TransformFunc)(Mat& src, Mat& dst);
	struct GFTransform_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		mutable vector<Mat*> src;
		mutable vector<Mat*> dst;
		TransformFunc func;
		GFTransform_ParBody(CGuidedImgObj &imgObj_, vector<Mat>& srcv, vector<Mat>& dstv, TransformFunc func_);
		GFTransform_ParBody(CGuidedImgObj &imgObj_, vector<vector<Mat> >& srcvv, vector<vector<Mat> >& dstvv, TransformFunc func_);
		void operator () (const Range& range) const;
		Range getRange() const
		{
			return Range(0, (int)src.size());
		}
	};


	struct ComputeCovFromSrcChannelsMul_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		vector<vector<Mat> > &cov;
		vector<Mat> &srcCnMean;
		ComputeCovFromSrcChannelsMul_ParBody(CGuidedImgObj& imgObj_,vector<Mat>& srcCnMean_, vector<vector<Mat> >& cov_)
			: imgObj(imgObj_), cov(cov_), srcCnMean(srcCnMean_) {}

		void operator () (const Range& range) const;
	};

	struct ComputeAlpha_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		vector<vector<Mat> > &alpha;
		vector<vector<Mat> > &covSrc;

		ComputeAlpha_ParBody(CGuidedImgObj& imgObj_, vector<vector<Mat> >& alpha_, vector<vector<Mat> >& covSrc_)
			: imgObj(imgObj_), alpha(alpha_), covSrc(covSrc_) {}

		void operator () (const Range& range) const;
	};

	struct ComputeBeta_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		vector<vector<Mat> > &alpha;
		vector<Mat> &srcCnMean;
		vector<Mat> &beta;
		ComputeBeta_ParBody(CGuidedImgObj& imgObj_, vector<vector<Mat> >& alpha_, vector<Mat>& srcCnMean_, vector<Mat>& beta_)
			: imgObj(imgObj_), alpha(alpha_), srcCnMean(srcCnMean_), beta(beta_) {}
		void operator () (const Range& range) const;
	};


	struct ApplyTransform_ParBody : public ParallelLoopBody
	{
		CGuidedImgObj &imgObj;
		vector<vector<Mat> > &alpha;
		vector<Mat> &beta;

		ApplyTransform_ParBody(CGuidedImgObj& imgObj_, vector<vector<Mat> >& alpha_, vector<Mat>& beta_)
			: imgObj(imgObj_), alpha(alpha_), beta(beta_) {}

		void operator () (const Range& range) const;
	};


private:
	// 执行并行运算结构体的函数
	void runParBody(const ParallelLoopBody& pb);

	template<typename V>
	void parMeanFilter(V &src, V &dst)
	{
		GFTransform_ParBody pb(m_imgObj, src, dst, &CGuidedImgObj::meanFilter);
		parallel_for_(pb.getRange(), pb);
	}


	template<typename V>
	void parConvertToWorkType(V &src, V &dst)
	{
		GFTransform_ParBody pb(m_imgObj, src, dst, &CGuidedImgObj::convertToWorkType);
		parallel_for_(pb.getRange(), pb);
	}

	void computeCovGuideAndSrc(vector<Mat>& srcCn,
		vector<Mat>& srcCnMean, vector<vector<Mat> >& cov);

};

