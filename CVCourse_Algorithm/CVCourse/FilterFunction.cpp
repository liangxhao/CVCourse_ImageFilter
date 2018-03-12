#include "FilterFunction.h"



CFilterFunction::CFilterFunction()
{
	m_bNeedGuided = true;
	m_bNeedL0 = true;
	m_bNeedBil = true;
	// 初始化为浮点类型，便于后续计算
	m_guidedBase = Mat(0, 0, CV_32FC3);
	m_L0Base = Mat(0, 0, CV_32FC3);
}


CFilterFunction::~CFilterFunction()
{
}


void CFilterFunction::create(Mat &src_)
{
	m_bNeedGuided = true;
	m_bNeedL0 = true;
	m_bNeedBil = true;
	src = src_;

	// 保存浮点类型的图像数据，注意这里限定取值为0-255，而不是0-1
	if (src.depth() == CV_8U)
	{
		src.convertTo(m_srcFloat, CV_32F); 
	}
	else if (src.depth() == CV_16U)
	{
		src.convertTo(m_srcFloat, CV_32F, 1 / 255.0f); 
	}
	else if (src.depth() == CV_64F)
	{
		src.convertTo(m_srcFloat, CV_32F);
	}
	
	// 计算并保存原始图像的梯度信息
	dWidth = m_srcFloat.cols - 2;
	dheight = m_srcFloat.rows - 2;
	dx = m_srcFloat(Rect(0, 0, dWidth, dheight + 2)) - m_srcFloat(Rect(2, 0, dWidth, dheight + 2));
	dy = m_srcFloat(Rect(0, 0, dWidth + 2, dheight)) - m_srcFloat(Rect(0, 2, dWidth + 2, dheight));

	//Sobel(m_srcFloat, dx, CV_32F, 1, 0, 1, 1, 0, BORDER_DEFAULT); //X方向  
	//Sobel(m_srcFloat, dy, CV_32F, 0, 1, 1, 1, 0, BORDER_DEFAULT); //Y方向  
}

void CFilterFunction::filterFun(int algId, double enhanceRatio)
{
	Mat baseNow;
	Mat detailNow;
	if (AlgId(algId) == Guided) {
		// 若尚未执行过所需的算法，则先执行相应算法
		if (m_bNeedGuided){
			runFilterAlg(algId);
			m_bNeedGuided = false;
		}
		baseNow = m_guidedBase;
		detailNow = m_guidedDetail;
	}
	else if (AlgId(algId) == L0) {
		if (m_bNeedL0) {
			runFilterAlg(algId);
			m_bNeedL0 = false;
		}
		baseNow = m_L0Base;
		detailNow = m_L0Detail;
	}
	else if (AlgId(algId) == Bil){
		if (m_bNeedBil) {
			runFilterAlg(algId);
			m_bNeedBil = false;
		}
		baseNow = m_bilBase;
		detailNow = m_bilDetail;
	}
	// 得到增强的图像
	Mat enhanceImg;
	threshold(baseNow + detailNow*enhanceRatio, enhanceImg, 255, 255, THRESH_TRUNC );
	threshold(enhanceImg, enhanceImg, 0, 255, THRESH_TOZERO);
	calcuReserveRate(enhanceImg);
	
	// 转为默认格式并记录在成员变量中
	enhanceImg.convertTo(dst, src.depth());
}


void CFilterFunction::calcuReserveRate(Mat enhanceImg)
{
	Mat dxNew, dyNew;
	dxNew = enhanceImg(Rect(0, 0, dWidth, dheight + 2)) - enhanceImg(Rect(2, 0, dWidth, dheight + 2));
	dyNew = enhanceImg(Rect(0, 0, dWidth + 2, dheight)) - enhanceImg(Rect(0, 2, dWidth + 2, dheight)); 
	
	//Sobel(enhanceImg, dxNew, CV_32F, 1, 0, 1, 1, 0, BORDER_DEFAULT); //X方向  
	//Sobel(enhanceImg, dyNew, CV_32F, 0, 1, 1, 1, 0, BORDER_DEFAULT); //Y方向  
	Mat changeX = dxNew - dx;
	Mat changeY = dyNew - dy;
	Mat absX = abs(changeX);
	Mat absY = abs(changeY);

	// 为了达到<而不是<=的效果，minChange取很小的负值
	double minChange = -0.99;
	// xRevMask：x方向梯度方向发生反向变化的二值图
	Mat xRevMask,yRevMask;
	threshold(dx.mul(changeX), xRevMask, minChange, 1, THRESH_BINARY_INV);
	threshold(dy.mul(changeY), yRevMask, minChange, 1, THRESH_BINARY_INV);

	// 所有反向的梯度变化，占全部梯度变化的比例
	Scalar xRev = sum(sum(xRevMask.mul(absX)));
	Scalar xAll = sum(sum(absX));
	Scalar yRev = sum(sum(yRevMask.mul(absY)));
	Scalar yAll = sum(sum(absY));
	reverseRate = (xRev(0) + yRev(0)) / (xAll(0) + yAll(0) + 1e-10);
}

void CFilterFunction::runFilterAlg(int algId)
{
	// 算法1使用引导滤波
	if (AlgId(algId) == Guided) {
		double rTemp, eps;
		readPara("configGuided.txt", rTemp, eps);
		int radius = int(rTemp + 0.5);
		// 引导图像与原图像为同一幅图像，先创建图像数据对象
		CGuidedImgObj imgObj(src, src, radius, eps, m_guidedBase.depth());
		// 再构建滤波对象
		CGuidedFilterObj guideObj(imgObj);
		// 得到滤波结果，即图像的基本分量
		guideObj.GetFilterDst(m_guidedBase);

		m_guidedDetail = m_srcFloat - m_guidedBase;
	}
	else if (AlgId(algId) == L0) {
		double lambda, kappa;
		readPara("configL0.txt", lambda, kappa);
		// 构建滤波对象
		CL0SmoothObj L0Obj;
		// 得到滤波结果，即图像的基本分量
		L0Obj.L0Smooth(src, m_L0Base, lambda, kappa);

		m_L0Base = m_L0Base * 255;
		m_L0Detail = m_srcFloat - m_L0Base;
	}
	else{
		double sigmaColor, sigmaSpace;
		readPara("configBil.txt", sigmaColor, sigmaSpace);

		bilateralFilter(src, m_bilBase, 0, sigmaColor, sigmaSpace);
		m_bilBase.convertTo(m_bilBase, CV_32F);
		m_bilDetail = m_srcFloat - m_bilBase;
	}
	
}


bool CFilterFunction::readPara(string filename, double &para1, double &para2)
{
	ifstream infile;
	infile.open(filename.data());
	if (!infile.is_open()) {
		return false;
	}

	string s;
	// 跳过第一行（参数1的说明），记录下第一个参数
	getline(infile, s);
	getline(infile, s);
	para1 = atof(s.data());
	// 类似得到第二个参数
	getline(infile, s);
	getline(infile, s);
	para2 = atof(s.data());

	infile.close();
	return true;
}


void CFilterFunction::initData(np::ndarray& array)
{
	int rows = array.shape(0);
	int cols = array.shape(1);
	int channel = array.shape(2);
	uchar *row_iter = reinterpret_cast<uchar*>(array.get_data());

	Mat image;

	if (channel == 3)
	{
		image.create(rows, cols, CV_8UC3);
	}
	else
	{
		image.create(rows, cols, CV_8U);
	}

	image.data = (uchar *)row_iter;

	this->create(image);
}


np::ndarray CFilterFunction::filter(int algId, double enhanceRatio)
{
	this->filterFun(algId, enhanceRatio);
	

	np::dtype dt = np::dtype::get_builtin<uchar>();
	py::tuple shape;
	py::tuple stride;
	if (this->dst.channels() == 3)
	{
		shape = py::make_tuple(this->dst.rows, this->dst.cols, this->dst.channels());
		stride = py::make_tuple(this->dst.cols*this->dst.channels() * sizeof(uchar), this->dst.channels() * sizeof(uchar), sizeof(uchar));
	}
	else
	{
		shape = py::make_tuple(this->dst.rows, this->dst.cols);
		stride = py::make_tuple(this->dst.cols * sizeof(uchar), sizeof(uchar));
		//stride = py::make_tuple(this->dst.rows * sizeof(uchar), sizeof(uchar));
	}

	py::object own;

	np::ndarray output = np::from_data(this->dst.data, dt, shape, stride, own);
	
	
	return output;
}

double CFilterFunction::getRate()
{

	return this->reverseRate;
}

BOOST_PYTHON_MODULE(CVCourse)
{

	np::initialize();
	py::class_<CFilterFunction>("Filter")
		.def("initData", &CFilterFunction::initData)
		.def("filter", &CFilterFunction::filter)
		.def("getRate", &CFilterFunction::getRate);

}