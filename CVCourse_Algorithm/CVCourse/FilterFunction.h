#pragma once
#include <opencv2\opencv.hpp>
#include "L0SmoothObj.h"
#include "GuidedFilterObj.h"

#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "opencv2/opencv.hpp"

namespace py = boost::python;
namespace np = boost::python::numpy;


using namespace std;
using namespace cv;

enum AlgId { Guided, L0, Bil };

class CFilterFunction
{
public:
	CFilterFunction();
	~CFilterFunction();
private:
	Mat src;
	// 增强后的图像，需返回
	Mat dst;
	// 增强后图像的梯度反转比例，需返回
	double reverseRate;

	void create(Mat &src_);
	void filterFun(int algId = 0, double enhanceRatio = 3);
private:
	bool readPara(string filename, double &para1, double &para2);
	// 记录是否需要执行滤波算法，同一幅图像只需要执行一次
	bool m_bNeedGuided;
	bool m_bNeedL0;
	bool m_bNeedBil;
	// 原图像浮点类型
	Mat m_srcFloat;
	// 记录已经得到的滤波结果
	Mat m_guidedBase;
	Mat m_L0Base;
	Mat m_bilBase;
	Mat m_guidedDetail;
	Mat m_L0Detail;
	Mat m_bilDetail;
	// 执行算法得到滤波结果
	void runFilterAlg(int algId = 0);

	// 计算梯度反向比例相关
	int dWidth, dheight;
	Mat dx;
	Mat dy;
	void calcuReserveRate(Mat enhanceImg);

public:
	void initData(np::ndarray& array);
	np::ndarray filter(int algId, double enhanceRatio);

	double getRate();

};

