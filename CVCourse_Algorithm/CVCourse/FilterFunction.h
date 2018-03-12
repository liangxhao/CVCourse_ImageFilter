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
	// ��ǿ���ͼ���践��
	Mat dst;
	// ��ǿ��ͼ����ݶȷ�ת�������践��
	double reverseRate;

	void create(Mat &src_);
	void filterFun(int algId = 0, double enhanceRatio = 3);
private:
	bool readPara(string filename, double &para1, double &para2);
	// ��¼�Ƿ���Ҫִ���˲��㷨��ͬһ��ͼ��ֻ��Ҫִ��һ��
	bool m_bNeedGuided;
	bool m_bNeedL0;
	bool m_bNeedBil;
	// ԭͼ�񸡵�����
	Mat m_srcFloat;
	// ��¼�Ѿ��õ����˲����
	Mat m_guidedBase;
	Mat m_L0Base;
	Mat m_bilBase;
	Mat m_guidedDetail;
	Mat m_L0Detail;
	Mat m_bilDetail;
	// ִ���㷨�õ��˲����
	void runFilterAlg(int algId = 0);

	// �����ݶȷ���������
	int dWidth, dheight;
	Mat dx;
	Mat dy;
	void calcuReserveRate(Mat enhanceImg);

public:
	void initData(np::ndarray& array);
	np::ndarray filter(int algId, double enhanceRatio);

	double getRate();

};

