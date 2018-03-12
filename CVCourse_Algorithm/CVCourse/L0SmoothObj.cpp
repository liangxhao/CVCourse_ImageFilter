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


// ��Ҫ�Ȳ����鲿���ٽ�����ɢ����Ҷ�任
void CL0SmoothObj::fft(InputArray src, OutputArray dst)
{
	Mat S = src.getMat();
	Mat planes[] = { S.clone(), Mat::zeros(S.size(), S.type()) };
	Mat x;
	merge(planes, 2, dst);

	// compute the result
	dft(dst, dst);
}

// ��src������ƽ�ƣ���shift_x�����أ���shift_y�����أ����õ�dst
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

// ��дmatlab����psf2otf������ɢ����srcתΪ��ѧ���ݺ���dst����Ҫ��dst�ĳߴ�height,width��С��src�ĳߴ�
// �����Ĺ������Ƚ�src��0����Ϊdst�ĳߴ磬��ͨ��ѭ��ƽ�ƽ�srcԭ�������ķ�������������[0,0]
// ���Ը�ƽ�����ú���������ɢ����Ҷ�任
void CL0SmoothObj::psf2otf(InputArray src, OutputArray dst, int height, int width)
{
	Mat S = src.getMat();
	Mat D = dst.getMat();

	Mat padded;

	if (S.data == D.data) {
		S = S.clone();
	}

	// ���Ҳ���·����Ӳ���0Ԫ�أ�ʹ��padded�ĳߴ�����Ŀ��ߴ�
	copyMakeBorder(S, padded, 0, height - S.rows, 0, width - S.cols,
		BORDER_CONSTANT, Scalar::all(0));

	// ������ƽ�ƣ�ʹ��ԭ��src����������ƽ�Ƶ�����λ�ã���[0,0]��
	shift(padded, padded, width - S.cols / 2, height - S.rows / 2);

	// �õ���ɢ����Ҷ�任���
	fft(padded, dst);
}


// src�Ǹ�������m*n*2��������ÿ��Ԫ��ģ��ƽ�����ɵľ���
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

// ����ִ�ж�������ÿһ�����ɢ����Ҷ�任
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

// ����ִ�ж�������ÿһ�����ɢ����Ҷ��任
void CL0SmoothObj::idftMultiChannel(const vector<Mat> &src, OutputArray dst)
{
	vector<Mat> channels(src);

	parallel_for_(Range(0, int(src.size())), ParallelIdft(channels));

	// ����ֻ��������任�����ʵ��
	for (int i = 0; unsigned(i) < src.size(); i++) {
		Mat panels[2];
		split(channels[i], panels);
		channels[i] = panels[0];
	}

	Mat D;
	merge(channels, D);
	D.copyTo(dst);
}


// ����ִ�ж�㸴������ĳ�������Ԫ�������
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

	// ��֤����ͼ��ĸ�ʽ
	CV_Assert(!S.empty());
	CV_Assert(S.depth() == CV_8U || S.depth() == CV_16U
		|| S.depth() == CV_32F || S.depth() == CV_64F);


	// תΪ������Mat�����ں�������
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

	// ����beta������
	const double betaMax = 100000;

	// �õ�����󵼴��ڵ���ɢ����Ҷ�任���
	Mat otfFx, otfFy;
	float kernel[2] = { -1, 1 };
	float kernel_inv[2] = { 1,-1 };
	// ���õ���ɢ��������ѧ���ݺ����ı任����ɲ�ִ��ڵĸ���Ҷ�任
	psf2otf(Mat(1, 2, CV_32FC1, kernel_inv), otfFx, S.rows, S.cols);
	psf2otf(Mat(2, 1, CV_32FC1, kernel_inv), otfFy, S.rows, S.cols);

	// ���������й�ʽ8��ÿ���Ż�Sʱ��ĸdenomConst�ǹ̶��ģ�������ǰ�����
	vector<Mat> denomConst;
	Mat tmp = pow2absComplex(otfFx) + pow2absComplex(otfFy);

	for (int i = 0; i < S.channels(); i++)
	{
		denomConst.push_back(tmp);
	}

	// �����S����ԭʼͼ��I���丵��Ҷ�任����ǹ�ʽ8�й̶���һ������ǰ�����
	// ���ǵ�S�����ж�㣬��ÿһ��ֱ���и���Ҷ�任
	vector<Mat> numerConst;
	dftMultiChannel(S, numerConst);
	// ���Ƹ����������ݶȵ�һ���Եı���beta����ʼ��Ϊlambda*2
	// ���ǿ��ǹ�ʽ9��������һ��Ҫ�󣬼���Ҫ��֤lambda/beta<1�������൱�ڳ�ʼ��Ϊlambda/beta=0.5
	double beta = 2 * lambda;
	while (beta < betaMax) {
		// �̶�Ŀ��ͼ��S����������Ż�h��v��������
		Mat h, v;
		// ��ͼ�����ò�ִ��ڵõ�ˮƽ����ֱ�����ϵ��ݶ�
		filter2D(S, h, -1, Mat(1, 2, CV_32FC1, kernel), Point(0, 0),
			0, BORDER_REPLICATE);
		filter2D(S, v, -1, Mat(2, 1, CV_32FC1, kernel), Point(0, 0),
			0, BORDER_REPLICATE);
		// ƽ���ͼ�Ϊ�ݶȷ�ֵ��ƽ��
		Mat hvMag = h.mul(h) + v.mul(v);

		Mat mask;
		// ������ֵlambda/beta���õ���Ҫ�����ݶȲ������Ĥͼmask����maskΪ0��λ����Ҫ���ݶ���0
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
		// ��maskΪ0��λ���ݶ���0������ʽ12����h��v���Ż����
		h = h.mul(mask);
		v = v.mul(mask);

		// �̶���������h��v���Ż�Ŀ��ͼ��S
		// denom����ʽ8�еķ�ĸ���֣�ÿ�ε�����beta���������ÿ�ε�������Ҫ����һ��
		vector<Mat> denom(S.channels());
		for (int i = 0; i < S.channels(); i++)
		{
			denom[i] = beta * denomConst[i] + 1;
		}
		// ���ݹ�ʽ8�ķ��Ӳ��֣���Ҫ��h��v�ֱ����ˮƽ����ֱ�Ĳ��
		// ����ļ��㷽ʽ�빫ʽ8�ĸ���Ҷ��ʽһ�£���ʵҲ����ֱ�ӽ��о�����������
		Mat hGrad, vGrad;
		filter2D(h, hGrad, -1, Mat(1, 2, CV_32FC1, kernel_inv));
		filter2D(v, vGrad, -1, Mat(2, 1, CV_32FC1, kernel_inv));
		// �õ����Ӳ���
		vector<Mat> hvGradFreq;
		dftMultiChannel(hGrad + vGrad, hvGradFreq);
		vector<Mat> numer(S.channels());
		for (int i = 0; i < S.channels(); i++)
		{
			numer[i] = numerConst[i] + hvGradFreq[i] * beta;
		}

		// ��㸴������ĳ�����ע������Ԫ����������������ֻ��ʵ������˽��ֻ��ʵ��������
		vector<Mat> sFreq(S.channels());
		divComplexByRealMultiChannel(numer, denom, sFreq);

		// ��ÿһ������渵��Ҷ�任���õ�S���Ż����
		idftMultiChannel(sFreq, S);

		// ����beta�������´�ѭ��
		beta = beta * kappa;
	}


	// ����ͬ�ĳߴ紴��Ŀ��ͼ��
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

