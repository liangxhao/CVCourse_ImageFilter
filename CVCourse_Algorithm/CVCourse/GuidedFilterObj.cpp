#include "GuidedFilterObj.h"

CGuidedImgObj::CGuidedImgObj(InputArray guide_, InputArray src_, int radius_, double eps_, int dstDepth_):
	guide(guide_), src(src_)
{
	// 记录下算法参数
	radius = radius_;
	eps = eps_;

	// 分离引导图像的各波段
	split(guide, guideCn);
	// 图像属性
	gCnNum = (int)guideCn.size();
	h = guideCn[0].rows;
	w = guideCn[0].cols;

	// 要求输出的图像类型
	dstDepth = dstDepth_;
}

void CGuidedImgObj::getWalkPattern(int eid, int &cn1, int &cn2) const
{
	static int wdata[] = {
		0, -1, -1, -1, -1, -1,
		0, -1, -1, -1, -1, -1,

		0,  0,  1, -1, -1, -1,
		0,  1,  1, -1, -1, -1,

		0,  0,  0,  2,  1,  1,
		0,  1,  2,  2,  2,  1,
	};

	cn1 = wdata[6 * 2 * (gCnNum - 1) + eid];
	cn2 = wdata[6 * 2 * (gCnNum - 1) + 6 + eid];
}



CGuidedFilterObj::CGuidedFilterObj(CGuidedImgObj imgObj_):m_imgObj(imgObj_)
{
}

CGuidedFilterObj::~CGuidedFilterObj()
{
}


// 执行并行运算结构体的函数
void CGuidedFilterObj::runParBody(const ParallelLoopBody& pb)
{
	parallel_for_(Range(0, m_imgObj.h), pb);
}




void CGuidedFilterObj::MulChannelsGuide_ParBody::operator()(const Range& range) const
{
	int total = covars.total();

	for (int i = range.start; i < range.end; i++)
	{
		int c1, c2;
		float *cov, *guide1, *guide2;

		for (int k = 0; k < total; k++)
		{
			imgObj.getWalkPattern(k, c1, c2);

			guide1 = imgObj.guideCn[c1].ptr<float>(i);
			guide2 = imgObj.guideCn[c2].ptr<float>(i);
			cov = covars(c1, c2).ptr<float>(i);

			mul(cov, guide1, guide2, imgObj.w);
		}
	}
}


void CGuidedFilterObj::ComputeCovGuideFromChannelsMul_ParBody::operator () (const Range& range) const
{
	int total = covars.total();
	float diagSummand = (float)(imgObj.eps);

	for (int i = range.start; i < range.end; i++)
	{
		int c1, c2;
		float *cov, *guide1, *guide2;

		for (int k = 0; k < total; k++)
		{
			imgObj.getWalkPattern(k, c1, c2);

			guide1 = imgObj.guideCnMean[c1].ptr<float>(i);
			guide2 = imgObj.guideCnMean[c2].ptr<float>(i);
			cov = covars(c1, c2).ptr<float>(i);

			if (c1 != c2)
			{
				sub_mul(cov, guide1, guide2, imgObj.w);
			}
			else
			{
				sub_mad(cov, guide1, guide2, -diagSummand, imgObj.w);
			}
		}
	}
}



CGuidedFilterObj::ComputeCovGuideInv_ParBody::ComputeCovGuideInv_ParBody(CGuidedImgObj& imgObj_,SymArray2D<Mat>& covars_)
	: imgObj(imgObj_),covars(covars_)
{
	imgObj.covarsInv.create(imgObj.gCnNum);

	if (imgObj.gCnNum == 3)
	{
		for (int k = 0; k < 2; k++)
			for (int l = 0; l < 3; l++)
				imgObj.covarsInv(k, l).create(imgObj.h, imgObj.w, CV_32FC1);

		////trick to avoid memory allocation
		imgObj.covarsInv(2, 0).create(imgObj.h, imgObj.w, CV_32FC1);
		imgObj.covarsInv(2, 1) = covars(2, 1);
		imgObj.covarsInv(2, 2) = covars(2, 2);

		return;
	}

	if (imgObj.gCnNum == 2)
	{
		imgObj.covarsInv(0, 0) = covars(1, 1);
		imgObj.covarsInv(0, 1) = covars(0, 1);
		imgObj.covarsInv(1, 1) = covars(0, 0);
		return;
	}

	if (imgObj.gCnNum == 1)
	{
		imgObj.covarsInv(0, 0) = covars(0, 0);
		return;
	}
}


void CGuidedFilterObj::ComputeCovGuideInv_ParBody::operator()(const Range& range) const
{
	if (imgObj.gCnNum == 3)
	{
		vector<float> covarsDet(imgObj.w);
		float *det = &covarsDet[0];

		for (int i = range.start; i < range.end; i++)
		{
			for (int k = 0; k < 3; k++)
				for (int l = 0; l <= k; l++)
				{
					float *dst = imgObj.covarsInv(k, l).ptr<float>(i);

					float *a00 = covars((k + 1) % 3, (l + 1) % 3).ptr<float>(i);
					float *a01 = covars((k + 1) % 3, (l + 2) % 3).ptr<float>(i);
					float *a10 = covars((k + 2) % 3, (l + 1) % 3).ptr<float>(i);
					float *a11 = covars((k + 2) % 3, (l + 2) % 3).ptr<float>(i);

					det_2x2(dst, a00, a01, a10, a11, imgObj.w);
				}

			for (int k = 0; k < 3; k++)
			{
				register float *a = covars(k, 0).ptr<float>(i);
				register float *ac = imgObj.covarsInv(k, 0).ptr<float>(i);

				if (k == 0)
					mul(det, a, ac, imgObj.w);
				else
					add_mul(det, a, ac, imgObj.w);
			}

			if (imgObj.eps < 1e-2)
			{
				for (int j = 0; j < imgObj.w; j++)
					if (abs(det[j]) < 1e-6f)
						det[j] = 1.f;
			}

			for (int k = 0; k < imgObj.covarsInv.total(); k += 1)
			{
				div_1x(imgObj.covarsInv(k).ptr<float>(i), det, imgObj.w);
			}
		}
		return;
	}

	if (imgObj.gCnNum == 2)
	{
		for (int i = range.start; i < range.end; i++)
		{
			float *a00 = imgObj.covarsInv(0, 0).ptr<float>(i);
			float *a10 = imgObj.covarsInv(1, 0).ptr<float>(i);
			float *a11 = imgObj.covarsInv(1, 1).ptr<float>(i);

			div_det_2x2(a00, a10, a11, imgObj.w);
		}
		return;
	}

	if (imgObj.gCnNum == 1)
	{
		for (int i = range.start; i < range.end; i++)
		{
			float *res = covars(0, 0).ptr<float>(i);
			inv_self(res, imgObj.w);
		}
		return;
	}
}

void CGuidedFilterObj::MulChannelsGuideAndSrc_ParBody::operator()(const Range& range) const
{
	int srcCnNum = (int)srcCn.size();

	for (int i = range.start; i < range.end; i++)
	{
		for (int si = 0; si < srcCnNum; si++)
		{
			int step = (si % 2) * 2 - 1;
			int start = (si % 2) ? 0 : imgObj.gCnNum - 1;
			int end = (si % 2) ? imgObj.gCnNum : -1;

			float *srcLine = srcCn[si].ptr<float>(i);

			for (int gi = start; gi != end; gi += step)
			{
				float *guideLine = imgObj.guideCn[gi].ptr<float>(i);
				float *dstLine = cov[si][gi].ptr<float>(i);

				mul(dstLine, srcLine, guideLine, imgObj.w);
			}
		}
	}
}


CGuidedFilterObj::GFTransform_ParBody::GFTransform_ParBody(CGuidedImgObj &imgObj_, 
	vector<Mat>& srcv, vector<Mat>& dstv, TransformFunc func_)
	: imgObj(imgObj_), func(func_)
{
	CV_DbgAssert(srcv.size() == dstv.size());
	src.resize(srcv.size());
	dst.resize(srcv.size());

	for (int i = 0; i < (int)srcv.size(); i++)
	{
		src[i] = &srcv[i];
		dst[i] = &dstv[i];
	}
}

CGuidedFilterObj::GFTransform_ParBody::GFTransform_ParBody(CGuidedImgObj &imgObj_, 
	vector<vector<Mat> >& srcvv, vector<vector<Mat> >& dstvv, TransformFunc func_)
	: imgObj(imgObj_), func(func_)
{
	CV_DbgAssert(srcvv.size() == dstvv.size());
	int n = (int)srcvv.size();
	int total = 0;

	for (int i = 0; i < n; i++)
	{
		CV_DbgAssert(srcvv[i].size() == dstvv[i].size());
		total += (int)srcvv[i].size();
	}

	src.resize(total);
	dst.resize(total);

	int k = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < (int)srcvv[i].size(); j++)
		{
			src[k] = &srcvv[i][j];
			dst[k] = &dstvv[i][j];
			k++;
		}
	}
}

void CGuidedFilterObj::GFTransform_ParBody::operator()(const Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		(imgObj.*func)(*src[i], *dst[i]);
	}
}


void CGuidedFilterObj::ComputeCovFromSrcChannelsMul_ParBody::operator()(const Range& range) const
{
	int srcCnNum = (int)srcCnMean.size();

	for (int i = range.start; i < range.end; i++)
	{
		for (int si = 0; si < srcCnNum; si++)
		{
			int step = (si % 2) * 2 - 1;
			int start = (si % 2) ? 0 : imgObj.gCnNum - 1;
			int end = (si % 2) ? imgObj.gCnNum : -1;

			register float *srcMeanLine = srcCnMean[si].ptr<float>(i);

			for (int gi = start; gi != end; gi += step)
			{
				float *guideMeanLine = imgObj.guideCnMean[gi].ptr<float>(i);
				float *covLine = cov[si][gi].ptr<float>(i);

				sub_mul(covLine, srcMeanLine, guideMeanLine, imgObj.w);
			}
		}
	}
}


void CGuidedFilterObj::ComputeAlpha_ParBody::operator()(const Range& range) const
{
	int srcCnNum = (int)covSrc.size();

	for (int i = range.start; i < range.end; i++)
	{
		for (int si = 0; si < srcCnNum; si++)
		{
			for (int gi = 0; gi < imgObj.gCnNum; gi++)
			{
				float *y, *A, *dstAlpha;

				dstAlpha = alpha[si][gi].ptr<float>(i);
				for (int k = 0; k < imgObj.gCnNum; k++)
				{
					y = covSrc[si][k].ptr<float>(i);
					A = imgObj.covarsInv(gi, k).ptr<float>(i);

					if (k == 0)
					{
						mul(dstAlpha, A, y, imgObj.w);
					}
					else
					{
						add_mul(dstAlpha, A, y, imgObj.w);
					}
				}
			}
		}
	}
}


void CGuidedFilterObj::ComputeBeta_ParBody::operator()(const Range& range) const
{
	int srcCnNum = (int)srcCnMean.size();
	CV_DbgAssert(&srcCnMean == &beta);

	for (int i = range.start; i < range.end; i++)
	{
		float *_g[4];
		for (int gi = 0; gi < imgObj.gCnNum; gi++)
			_g[gi] = imgObj.guideCnMean[gi].ptr<float>(i);

		float *betaDst, *g, *a;
		for (int si = 0; si < srcCnNum; si++)
		{
			betaDst = beta[si].ptr<float>(i);
			for (int gi = 0; gi < imgObj.gCnNum; gi++)
			{
				a = alpha[si][gi].ptr<float>(i);
				g = _g[gi];

				sub_mul(betaDst, a, g, imgObj.w);
			}
		}
	}
}

void CGuidedFilterObj::ApplyTransform_ParBody::operator()(const Range& range) const
{
	int srcCnNum = (int)alpha.size();

	for (int i = range.start; i < range.end; i++)
	{
		float *_g[4];
		for (int gi = 0; gi < imgObj.gCnNum; gi++)
			_g[gi] = imgObj.guideCn[gi].ptr<float>(i);

		float *betaDst, *g, *a;
		for (int si = 0; si < srcCnNum; si++)
		{
			betaDst = beta[si].ptr<float>(i);
			for (int gi = 0; gi < imgObj.gCnNum; gi++)
			{
				a = alpha[si][gi].ptr<float>(i);
				g = _g[gi];

				add_mul(betaDst, a, g, imgObj.w);
			}
		}
	}
}



void CGuidedFilterObj::computeCovGuideAndSrc(vector<Mat>& srcCn, vector<Mat>& srcCnMean, vector<vector<Mat> >& cov)
{
	int srcCnNum = (int)srcCn.size();

	cov.resize(srcCnNum);
	for (int i = 0; i < srcCnNum; i++)
	{
		cov[i].resize(m_imgObj.gCnNum);
		for (int j = 0; j < m_imgObj.gCnNum; j++)
			cov[i][j].create(m_imgObj.h, m_imgObj.w, CV_32FC1);
	}

	// 计算交叉相乘结果
	runParBody(MulChannelsGuideAndSrc_ParBody(m_imgObj, srcCn, cov));
	// 交叉相乘结果窗口均值
	parMeanFilter(cov, cov);
	// 计算原图像窗口均值
	parMeanFilter(srcCn, srcCnMean);
	// 引导图像的窗口均值已知，根据上述结果可以计算协方差
	runParBody(ComputeCovFromSrcChannelsMul_ParBody(m_imgObj, srcCnMean, cov));
}



void CGuidedFilterObj::GetFilterDst(OutputArray dst)
{
	CV_Assert(!m_imgObj.guide.empty() && m_imgObj.radius >= 0 && m_imgObj.eps >= 0);
	CV_Assert((m_imgObj.guide.depth() == CV_32F || m_imgObj.guide.depth() == CV_8U || 
		m_imgObj.guide.depth() == CV_16U) && (m_imgObj.guide.channels() <= 3));

	// 转为浮点类型图像
	parConvertToWorkType(m_imgObj.guideCn, m_imgObj.guideCn);

	// 各波段进行均值滤波
	m_imgObj.guideCnMean.resize(m_imgObj.gCnNum);
	parMeanFilter(m_imgObj.guideCn, m_imgObj.guideCnMean);


	SymArray2D<Mat> covars;
	covars.create(m_imgObj.gCnNum);
	for (int i = 0; i < covars.total(); i++)
		covars(i).create(m_imgObj.h, m_imgObj.w, CV_32FC1);

	// 计算各像素邻域内协方差矩阵，第一步：波段之间相乘
	runParBody(MulChannelsGuide_ParBody(m_imgObj, covars));
	// 第二步：均值滤波，得到协方差公式被减数一项
	parMeanFilter(covars.vec, covars.vec);
	// 第三步：结合各波段的均值滤波结果（减数一项）
	// 得到每个像素邻域内协方差矩阵（若只有一个波段，则为方差）
	runParBody(ComputeCovGuideFromChannelsMul_ParBody(m_imgObj, covars));

	// 得到并记录协方差逆矩阵
	runParBody(ComputeCovGuideInv_ParBody(m_imgObj, covars));
	covars.release();


	CV_Assert(!m_imgObj.src.empty() && (m_imgObj.src.depth() == CV_32F || m_imgObj.src.depth() == CV_8U));
	if (m_imgObj.src.rows() != m_imgObj.h || m_imgObj.src.cols() != m_imgObj.w)
	{
		CV_Error(Error::StsBadSize, "Size of filtering image must be equal to size of guide image");
		return;
	}

	if (m_imgObj.dstDepth == -1) m_imgObj.dstDepth = m_imgObj.src.depth();
	int srcCnNum = m_imgObj.src.channels();

	vector<Mat> srcCn(srcCnNum);
	vector<Mat>& srcCnMean = srcCn;
	split(m_imgObj.src, srcCn);

	// 原图像的类型转换
	if (m_imgObj.src.depth() != CV_32F)
	{
		parConvertToWorkType(srcCn, srcCn);
	}

	// 计算引导图像和原图像之间的协方差
	vector<vector<Mat> > covSrcGuide(srcCnNum);
	computeCovGuideAndSrc(srcCn, srcCnMean, covSrcGuide);

	vector<vector<Mat> > alpha(srcCnNum);
	for (int si = 0; si < srcCnNum; si++)
	{
		alpha[si].resize(m_imgObj.gCnNum);
		for (int gi = 0; gi < m_imgObj.gCnNum; gi++)
			alpha[si][gi].create(m_imgObj.h, m_imgObj.w, CV_32FC1);
	}
	// 公式19，根据引导图像和原图像的协方差，以及之前计算出的引导图像协方差逆矩阵，计算alpha
	// 注意到对于3个波段的图像alpha是长度为3的向量
	runParBody(ComputeAlpha_ParBody(m_imgObj, alpha, covSrcGuide));
	covSrcGuide.clear();

	// 公式20，根据alpha、引导图像均值滤波结果和原图像均值滤波结果，得到beta，beta一定是标量
	// 注意到使用原图像均值滤波结果初始化beta，只需再减去alpha与引导图像均值滤波结果之内积即可
	vector<Mat>& beta = srcCnMean;
	runParBody(ComputeBeta_ParBody(m_imgObj, alpha, srcCnMean, beta));

	// 公式21的准备工作，对alpha和beta进行均值滤波
	parMeanFilter(beta, beta);
	parMeanFilter(alpha, alpha);

	// 公式21，得到滤波结果
	runParBody(ApplyTransform_ParBody(m_imgObj, alpha, beta));

	// 转为所需的目标图像类型
	if (m_imgObj.dstDepth != CV_32F)
	{
		for (int i = 0; i < srcCnNum; i++)
			beta[i].convertTo(beta[i], m_imgObj.dstDepth);
	}
	merge(beta, dst);
}
