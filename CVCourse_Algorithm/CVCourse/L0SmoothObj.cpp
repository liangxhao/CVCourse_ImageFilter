#include "L0SmoothObj.h"


void ParallelDivComplexByReal::operator() (const Range& range) const
{
	for (int i = range.start; i != range.end; i++)
	{
		Mat aPanels[2];
		split(numer_[i], aPanels);
		//Mat bPanels[2];
		//split(denom_[i], bPanels);

		Mat realPart;
		Mat imaginaryPart;

		divide(aPanels[0], denom_[i], realPart);
		divide(aPanels[1], denom_[i], imaginaryPart);

		aPanels[0] = realPart;
		aPanels[1] = imaginaryPart;

		merge(aPanels, 2, dst_[i]);
	}
}


// 需要先补充虚部，再进行离散傅里叶变换
void CL0SmoothObj::fft(InputArray src, OutputArray dst)
{
	Mat S = src.getMat();
	Mat planes[] = { S.clone(), Mat::zeros(S.size(), S.type()) };
	Mat x;
	merge(planes, 2, dst);

	// compute the result
	dft(dst, dst);
}

// 将src向右下平移（右shift_x个像素，下shift_y个像素），得到dst
void CL0SmoothObj::shift(InputArray src, OutputArray dst, int shift_x, int shift_y)
{
	Mat S = src.getMat();
	Mat D = dst.getMat();

	if (S.data == D.data)
	{
		S = S.clone();
	}

	D.create(S.size(), S.type());

	Mat s0(S, Rect(0, 0, S.cols - shift_x, S.rows - shift_y));
	Mat s1(S, Rect(S.cols - shift_x, 0, shift_x, S.rows - shift_y));
	Mat s2(S, Rect(0, S.rows - shift_y, S.cols - shift_x, shift_y));
	Mat s3(S, Rect(S.cols - shift_x, S.rows - shift_y, shift_x, shift_y));

	Mat d0(D, Rect(shift_x, shift_y, S.cols - shift_x, S.rows - shift_y));
	Mat d1(D, Rect(0, shift_y, shift_x, S.rows - shift_y));
	Mat d2(D, Rect(shift_x, 0, S.cols - shift_x, shift_y));
	Mat d3(D, Rect(0, 0, shift_x, shift_y));

	s0.copyTo(d0);
	s1.copyTo(d1);
	s2.copyTo(d2);
	s3.copyTo(d3);
}

// 仿写matlab函数psf2otf（点扩散函数src转为光学传递函数dst），要求dst的尺寸height,width不小于src的尺寸
// 基本的过程是先将src补0扩充为dst的尺寸，并通过循环平移将src原本的中心放置于左上像素[0,0]
// 最后对该平移所得函数进行离散傅里叶变换
void CL0SmoothObj::psf2otf(InputArray src, OutputArray dst, int height, int width)
{
	Mat S = src.getMat();
	Mat D = dst.getMat();

	Mat padded;

	if (S.data == D.data) {
		S = S.clone();
	}

	// 在右侧和下方增加补充0元素，使得padded的尺寸满足目标尺寸
	copyMakeBorder(S, padded, 0, height - S.rows, 0, width - S.cols,
		BORDER_CONSTANT, Scalar::all(0));

	// 向右下平移，使得原来src的中心像素平移到左上位置（即[0,0]）
	shift(padded, padded, width - S.cols / 2, height - S.rows / 2);

	// 得到离散傅里叶变换结果
	fft(padded, dst);
}


// src是复数矩阵（m*n*2），返回每个元素模的平方构成的矩阵
Mat CL0SmoothObj::pow2absComplex(InputArray src)
{
	Mat S = src.getMat();

	Mat sPanels[2];
	split(S, sPanels);

	Mat mag;
	magnitude(sPanels[0], sPanels[1], mag);
	pow(mag, 2, mag);

	return mag;
}

// 并行执行多层矩阵中每一层的离散傅里叶变换
void CL0SmoothObj::dftMultiChannel(InputArray src, vector<Mat> &dst)
{
	Mat S = src.getMat();

	split(S, dst);

	for (int i = 0; i < S.channels(); i++) {
		Mat planes[] = { dst[i].clone(), Mat::zeros(dst[i].size(), dst[i].type()) };
		merge(planes, 2, dst[i]);
	}

	parallel_for_(cv::Range(0, S.channels()), ParallelDft(dst));
}

// 并行执行多层矩阵中每一层的离散傅里叶逆变换
void CL0SmoothObj::idftMultiChannel(const vector<Mat> &src, OutputArray dst)
{
	vector<Mat> channels(src);

	parallel_for_(Range(0, int(src.size())), ParallelIdft(channels));

	// 这里只保留了逆变换结果的实部
	for (int i = 0; unsigned(i) < src.size(); i++) {
		Mat panels[2];
		split(channels[i], panels);
		channels[i] = panels[0];
	}

	Mat D;
	merge(channels, D);
	D.copyTo(dst);
}


// 并行执行多层复数矩阵的除法（逐元素相除）
void CL0SmoothObj::divComplexByRealMultiChannel(vector<Mat> &numer,
	vector<Mat> &denom, vector<Mat> &dst)
{

	for (int i = 0; unsigned(i) < numer.size(); i++)
	{
		dst[i].create(numer[i].size(), numer[i].type());
	}
	parallel_for_(Range(0, int(numer.size())), ParallelDivComplexByReal(numer, denom, dst));

}


void CL0SmoothObj::L0Smooth(InputArray src, OutputArray dst, double lambda, double kappa)
{
	Mat S = src.getMat();

	// 验证输入图像的格式
	CV_Assert(!S.empty());
	CV_Assert(S.depth() == CV_8U || S.depth() == CV_16U
		|| S.depth() == CV_32F || S.depth() == CV_64F);


	// 转为浮点型Mat，便于后续计算
	if (S.depth() == CV_8U)
	{
		S.convertTo(S, CV_32F, 1 / 255.0f);
	}
	else if (S.depth() == CV_16U)
	{
		S.convertTo(S, CV_32F, 1 / 65535.0f);
	}
	else if (S.depth() == CV_64F)
	{
		S.convertTo(S, CV_32F);
	}

	// 参数beta的上限
	const double betaMax = 100000;

	// 得到差分求导窗口的离散傅里叶变换结果
	Mat otfFx, otfFy;
	float kernel[2] = { -1, 1 };
	float kernel_inv[2] = { 1,-1 };
	// 利用点扩散函数到光学传递函数的变换，完成差分窗口的傅里叶变换
	psf2otf(Mat(1, 2, CV_32FC1, kernel_inv), otfFx, S.rows, S.cols);
	psf2otf(Mat(2, 1, CV_32FC1, kernel_inv), otfFy, S.rows, S.cols);

	// 根据论文中公式8，每次优化S时分母denomConst是固定的，可以提前计算出
	vector<Mat> denomConst;
	Mat tmp = pow2absComplex(otfFx) + pow2absComplex(otfFy);

	for (int i = 0; i < S.channels(); i++)
	{
		denomConst.push_back(tmp);
	}

	// 这里的S就是原始图像I，其傅里叶变换结果是公式8中固定的一项，因此提前计算出
	// 考虑到S可能有多层，对每一层分别进行傅里叶变换
	vector<Mat> numerConst;
	dftMultiChannel(S, numerConst);
	// 控制辅助变量与梯度的一致性的变量beta，初始化为lambda*2
	// 这是考虑公式9中隐含了一个要求，即需要保证lambda/beta<1，这里相当于初始化为lambda/beta=0.5
	double beta = 2 * lambda;
	while (beta < betaMax) {
		// 固定目标图像S，首先求解优化h和v的子问题
		Mat h, v;
		// 对图像利用差分窗口得到水平和竖直方向上的梯度
		filter2D(S, h, -1, Mat(1, 2, CV_32FC1, kernel), Point(0, 0),
			0, BORDER_REPLICATE);
		filter2D(S, v, -1, Mat(2, 1, CV_32FC1, kernel), Point(0, 0),
			0, BORDER_REPLICATE);
		// 平方和即为梯度幅值的平方
		Mat hvMag = h.mul(h) + v.mul(v);

		Mat mask;
		// 根据阈值lambda/beta，得到需要保持梯度不变的掩膜图mask，即mask为0的位置需要将梯度置0
		if (S.channels() == 1)
		{
			threshold(hvMag, mask, lambda / beta, 1, THRESH_BINARY);
		}
		else if (S.channels() > 1)
		{
			vector<Mat> channels(S.channels());
			split(hvMag, channels);
			hvMag = channels[0];

			for (int i = 1; i < S.channels(); i++)
			{
				hvMag = hvMag + channels[i];
			}

			threshold(hvMag, mask, lambda / beta, 1, THRESH_BINARY);

			Mat in[] = { mask, mask, mask };
			merge(in, 3, mask);
		}
		// 将mask为0的位置梯度置0（即公式12），h和v的优化完成
		h = h.mul(mask);
		v = v.mul(mask);

		// 固定辅助变量h和v，优化目标图像S
		// denom即公式8中的分母部分，每次迭代中beta会增大，因此每次迭代都需要计算一次
		vector<Mat> denom(S.channels());
		for (int i = 0; i < S.channels(); i++)
		{
			denom[i] = beta * denomConst[i] + 1;
		}
		// 根据公式8的分子部分，需要对h和v分别进行水平和竖直的差分
		// 这里的计算方式与公式8的傅里叶形式一致，其实也可以直接进行矩阵的相减运算
		Mat hGrad, vGrad;
		filter2D(h, hGrad, -1, Mat(1, 2, CV_32FC1, kernel_inv));
		filter2D(v, vGrad, -1, Mat(2, 1, CV_32FC1, kernel_inv));
		// 得到分子部分
		vector<Mat> hvGradFreq;
		dftMultiChannel(hGrad + vGrad, hvGradFreq);
		vector<Mat> numer(S.channels());
		for (int i = 0; i < S.channels(); i++)
		{
			numer[i] = numerConst[i] + hvGradFreq[i] * beta;
		}

		// 多层复数矩阵的除法，注意是逐元素相除，但这里除数只有实部，因此结果只有实部有意义
		vector<Mat> sFreq(S.channels());
		divComplexByRealMultiChannel(numer, denom, sFreq);

		// 对每一层进行逆傅里叶变换，得到S的优化结果
		idftMultiChannel(sFreq, S);

		// 增大beta，进入下次循环
		beta = beta * kappa;
	}


	// 按相同的尺寸创建目标图像
	dst.create(S.size(), S.type());
	Mat D = dst.getMat();
	if (D.depth() == CV_8U)
	{
		S.convertTo(D, CV_8U, 255);
	}
	else if (D.depth() == CV_16U)
	{
		S.convertTo(D, CV_16U, 65535);
	}
	else if (D.depth() == CV_64F)
	{
		S.convertTo(D, CV_64F);
	}
	else
	{
		S.copyTo(D);
	}
}

