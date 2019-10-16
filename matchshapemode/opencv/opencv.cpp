//**************//**************//**************//**************//**************//**************//
//**************//**************//**************//**************//**************//**************//
//**************//**************//**************//**************//**************//**************//
//**************//**************//**************//**************//**************//**************//
//**************																//**************//
//**************	以下代码由厦门大学陈默含提供，如需使用，记得说声谢谢，嗯	//**************//
//**************																//**************//
//**************//**************//**************//**************//**************//**************//
//**************//**************//**************//**************//**************//**************//
//**************//**************//**************//**************//**************//**************//
//**************//**************//**************//**************//**************//**************//
#include "StdAfx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include<ppl.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<stdio.h>
#include <mutex>

using namespace cv;
using namespace std;
using namespace concurrency;
#define WINDOW_NAME "[程序窗口]"

void imrotate(Mat& img, Mat& newIm, double angle);
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);

void CreateScaledShapeModel(Mat Template, int PyrLevel, int AngleStart, int AngleExtent, int AngleStep, float ScaleMin, float ScaleMax, float ScaleStep, \
	vector<Mat>* pModelImageSet, vector<int>* pModelPointSet, vector<float>* pScaleSet, vector<float>* pAngleSet);

void FindScaledShapeModel(Mat Image, vector<Mat> ModelImageSet, vector<int> ModelPointSet, vector<float> ScaleSet, vector<float> AngleSet, int PyrLevel, float MinScore, \
	vector<int>* pRow, vector<int> * pCol, vector<float>* pScale, vector<float>* pAngle, vector<float>* pScore);

int main()
{	

	

	//缩放参数
	float scaleMin = 0.9, scaleMax = 1.1, scaleStep = 0.1;
	//角度参数
	float angleStart = 0, angleExtent = 360, angleStep = 2;
	//金字塔层数，规定在多大的金字塔图中搜索，数字越大，图片越小，搜索越快
	int pyrLevel = 3;
	//最小得分限制
	float minScore = 0.6;
	
	//生成模版集合
	vector<float> scaleSet;
	vector<float> angleSet;
	vector<Mat> modelImageSet;
	vector<int> modelPointSet;
	vector<int> row, col;
	vector<float> scale, angle, score;

	//待检测图片
//	Mat srcImage = imread("E:\\image\\imgRgn.bmp",0);
	Mat srcImage =imread("img/t4.bmp",0);
//	resize(srcImage, srcImage, Size(0, 0), 0.4, 0.4);
	copyMakeBorder(srcImage, srcImage, 20, 20, 20, 20, BORDER_CONSTANT, Scalar(0));
	/*Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat tempElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(srcImage, srcImage, tempElement);*/
	Mat cannysrcImage, tempsrcImage;
//	cannysrcImage = srcImage.clone();
//	tempsrcImage = srcImage.clone();
	blur(srcImage, srcImage,Size(3,3));
	Canny(srcImage, cannysrcImage, 100, 200, 3, false);
	Canny(srcImage, tempsrcImage, 100, 200, 3, false);
	for (int i = 0; i < pyrLevel; i++)
	{
		pyrDown(tempsrcImage, tempsrcImage);
	}
	threshold(tempsrcImage, tempsrcImage,30, 255, THRESH_BINARY);
	
	
	imshow("原始图", tempsrcImage);
	imwrite("原始图.bmp", tempsrcImage);
	waitKey(10);
	//模版图片
	//Mat modelImage = imread("img/q.png", 0)(Rect(180, 180, 200, 200));
	Mat modelImage = imread("img/t0.bmp", 0);
//	dilate(modelImage, modelImage, element);
	copyMakeBorder(modelImage, modelImage, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(0));
//	int size = modelImage.u->size;
	Mat cannymodelImage,tempmodelImage;
//	cannymodelImage = modelImage.clone();
//	tempmodelImage = modelImage.clone();
	blur(modelImage, modelImage, Size(3, 3));
	Canny(modelImage, cannymodelImage, 100, 200, 3, false);

	Canny(modelImage, tempmodelImage, 100, 200, 3, false);
	for (int i = 0; i < pyrLevel; i++)
	{
		pyrDown(tempmodelImage, tempmodelImage);
	}
	threshold(tempmodelImage, tempmodelImage, 30, 255, THRESH_BINARY);
	imshow("原始模版", tempmodelImage);
	imwrite("原始模版.bmp", tempmodelImage);
	waitKey(10);
	//创建模版集
	/*Mat rotateImg;
	imrotate(tempmodelImage, rotateImg, 90);
	imwrite("旋转模板.bmp", rotateImg);*/
	CreateScaledShapeModel(cannymodelImage, pyrLevel,  angleStart, angleExtent,  angleStep, scaleMin,\
		scaleMax,  scaleStep, &modelImageSet,  &modelPointSet, &scaleSet,&angleSet);

	double start = static_cast<double>(getTickCount());
	//进行匹配
	FindScaledShapeModel(cannysrcImage,modelImageSet, modelPointSet, scaleSet,  angleSet, pyrLevel,minScore,&row, \
			&col,&scale,&angle,&score);
	//标注匹配位置
	for (int i = 0; i < row.size(); i++)
	{
		circle(srcImage, Point(row[i], col[i]), scale[i]*modelImage.rows / 2, 255, 2, 8, 0);
		putText(srcImage, format("Number:%d",i),Point(row[i], col[i]), cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
		cout << "第"<<i<<"号:" <<Point(col[i], row[i]) << " 角度:" << angle[i] << " 缩放:" << scale[i] <<" 成绩:"<<score[i]<< endl;
	}
	double time = ((double)getTickCount() - start) / getTickFrequency();
	cout << "所用时间为：" << time << "秒" << endl;
	imshow("原图位置", srcImage);
	imwrite("resImage.bmp", srcImage);
	waitKey(0);
	

}
void on_mouse(int event, int x, int y, int flags, void* userdata)
{
	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 2;
	int thickness = 2;

	if (event == CV_EVENT_MOUSEMOVE)
	{
		//int value= srcImage.at<uchar>(Point(x, y));
		Point pt = Point(x, y);
		char temp[16];
		//cout << pt.x << " " << pt.y << " 像素值" << value << endl;
	}
}


/*************************************************
Function:       //	CreateScaledShapeModel
Description:    //	创建模版集

Input:          //	Template:模版图片的边缘图
					PyrLevel:金字塔缩小层数
					AngleStart,AngleExtent,AngleStep:旋转角度候选参数
					ScaleMin,ScaleMax,ScaleStep:缩放比例候选参数
					

Output:         //	pModelImageSet,pModelPointSet,pScaleSet,pAngleSet:模版集指针
Return:         //  无
Others:         //	
*************************************************/
void CreateScaledShapeModel(Mat Template, int PyrLevel, int AngleStart, int AngleExtent, int AngleStep, float ScaleMin, float ScaleMax, float ScaleStep, \
	vector<Mat>* pModelImageSet, vector<int>* pModelPointSet, vector<float>* pScaleSet, vector<float>* pAngleSet)
{
	vector<Mat> ModelImageSet;
	vector<int> ModelPointSet;
	vector<float> AngleSet;
	vector<float> ScaleSet;
	while (ScaleMin <= ScaleMax)
	{
		cout << ScaleMax << endl;
		ScaleSet.push_back(ScaleMax);
		ScaleMax -= ScaleStep;
	}
	while (AngleStart <= AngleExtent)
	{
		cout << AngleExtent << endl;
		AngleSet.push_back(AngleExtent);
		AngleExtent -= AngleStep;
	}
	//模版生成	
	for (int level = 0; level <= PyrLevel; level++)
	{
		Mat pyrmodelImage = Template;
		for (int i = 0; i < level; i++)
		{
			pyrDown(pyrmodelImage, pyrmodelImage);
		}
		//缩放
		for (int i = 0; i < ScaleSet.size(); i++)
		{
			Mat scaleImage;
			resize(pyrmodelImage, scaleImage, Size(round(pyrmodelImage.cols*ScaleSet[i]), round(pyrmodelImage.rows*ScaleSet[i])), 0, 0, INTER_LINEAR);
			//旋转
			for (int j = 0; j < AngleSet.size(); j++)
			{
				Mat rotateImage;
				imrotate(scaleImage, rotateImage, AngleSet[j]);
				//threshold(rotateImage, rotateImage, 1, 255, 0);
				Canny(rotateImage, rotateImage, 50, 100, 3, false);
//				imwrite("旋转.jpg", rotateImage);
				/*imshow("旋转", rotateImage);
				imwrite("旋转.jpg", rotateImage);
				waitKey(0);*/
				rotateImage /= 255;
				ModelImageSet.push_back(rotateImage);
				int pointNum = 0;
				/*for (int i = 0; i < rotateImage.cols; i++)
				{
					for (int j = 0; j < rotateImage.rows; j++)
					{
						if (rotateImage.at<uchar>(Point(i, j)) != 0)
							pointNum++;
					}
				}*/
				pointNum = cv::sum(rotateImage)[0];
				ModelPointSet.push_back(pointNum);
				rotateImage.release();
			}
			scaleImage.release();
		}
	}
	*pModelImageSet = ModelImageSet;
	*pModelPointSet = ModelPointSet;
	*pAngleSet = AngleSet;
	*pScaleSet = ScaleSet;
}



/*************************************************
Function:       //	FindScaledShapeModel
Description:    //	在一张图片中搜索与模版相似的图形

Input:          //	Image:待检测图片
					ModelImageSet，ModelPointSet，ScaleSet，AngleSet:模版集
					PyrLevel:金字塔缩小层数
					MinScore:筛选相似度阈值		
Output:         //	pRow,pCol,pScale,pAngle,pScore:输出匹配到的元素参数集合的指针
Return:         //  无
Others:         //	使用该函数前需要先调用CreateScaledShapeModel
*************************************************/
void FindScaledShapeModel(Mat Image, vector<Mat> ModelImageSet, vector<int> ModelPointSet, vector<float> ScaleSet, vector<float> AngleSet, int PyrLevel, float MinScore,\
	vector<int>* pRow, vector<int> * pCol, vector<float>* pScale, vector<float>* pAngle, vector<float>* pScore)
{
	mutex mt;
	Mat modelImage = ModelImageSet[0];
	vector<int> Row;
	vector<int> Col;
	vector<float> Scale;
	vector<float> Angle;
	vector<float> Score;
	bool findFlag = false;
	Point center;
	center.x = Image.cols / 2;
	center.y = Image.rows / 2;
	
	//金字塔分层匹配
	for (int level = PyrLevel; !findFlag && level >= PyrLevel; level--)
	{		
		Mat pyrsrcImage = Image;
		for (int i = 0; i < level; i++)
		{
			pyrDown(pyrsrcImage, pyrsrcImage);
		}

		int kernSize = floor(sqrt(min(pyrsrcImage.rows / 100, pyrsrcImage.cols / 100)));		
		Mat kern = Mat::ones(2 * kernSize + 5, 2 * kernSize + 5, CV_8U);
			
		Mat blurImage;
		filter2D(pyrsrcImage, blurImage, pyrsrcImage.depth(), kern);		
		/*imshow("糊化原图", blurImage);
		moveWindow("糊化原图", 0, 0);
		waitKey(10);*/
		Mat tempblurImage;
		blurImage.convertTo(tempblurImage, CV_8U);
		tempblurImage /= 255;
		int parallelnum = ScaleSet.size() *AngleSet.size();

		parallel_for(0, parallelnum, [&](int k)	
		{
			Mat scoreImage(tempblurImage.size(), CV_16U);
			Mat tempmodelImage = ModelImageSet.at(ModelImageSet.size()- 1 - k);
			int temppointNum = ModelPointSet.at(ModelPointSet.size() - 1 - k);
			float max_score = 0;
			/*imshow("模版", tempmodelImage);
			resizeWindow("模版", tempmodelImage.rows, tempmodelImage.cols);		
			moveWindow("模版", blurImage.cols,0);
			waitKey(10);*/
			//double start = static_cast<double>(getTickCount());
			filter2D(tempblurImage, scoreImage, scoreImage.depth(), tempmodelImage);
			//double time = ((double)getTickCount() - start) / getTickFrequency();
			//cout << "所用时间为：" << time << "秒" << endl;
			mt.lock();
			while (1)
			{
				double v_min, v_max;
				int idx_min[2] = { 255,255 }, idx_max[2] = { 255, 255 };
				minMaxIdx(scoreImage, &v_min, &v_max, idx_min, idx_max);
				
				scoreImage.at<ushort>(idx_max[0], idx_max[1]) = 0;
				
				max_score = v_max / temppointNum;

				//cout << "第" << level << "层" << "第" << k + 1 << "个成绩：" << max_score << endl;
				if (max_score > MinScore)
				{
					float scale = ScaleSet[ScaleSet.size() - 1 - (k) / AngleSet.size()];
					float angle = AngleSet[AngleSet.size() - 1 - (k) % AngleSet.size()];
					//int selectx = (idx_max[1] - (tempmodelImage.cols - 1) / 2)*pow(2, level);
					//int selecty = (idx_max[0] - (tempmodelImage.rows - 1) / 2)*pow(2, level);
					//int pyrselectx = idx_max[1] - tempmodelImage.cols / 2;
					//int pyrselecty = idx_max[0] - tempmodelImage.rows / 2;
					int tempRow = idx_max[0] * pow(2, level);
					int tempCol = idx_max[1] * pow(2, level);
					if (abs(tempCol - center.x)<10 && abs(tempRow - center.y)<10)
					{
						Row.push_back(idx_max[1] * pow(2, level));
						Col.push_back(idx_max[0] * pow(2, level));
						Scale.push_back(scale);
						Angle.push_back(angle);
						Score.push_back(max_score);
						imwrite("scoreImage.bmp", scoreImage);
					}
					
					
					//cout << Point(selectx, selecty) << " " << Point(pyrselectx, pyrselecty) << endl;
					//rectangle(blurImage, Rect(pyrselectx, pyrselecty, tempmodelImage.cols, tempmodelImage.rows), 255, 2, 8);
					//rectangle(Image, Rect(selectx, selecty, modelImage.cols / scale, modelImage.rows / scale), 255, 2, 8);
					//imshow("缩放图位置", blurImage);
					//imshow("原图位置", Image);
					/*imshow("scoreImage", scoreImage);
					imwrite("scoreImage.bmp", scoreImage);
					waitKey(0);*/
					//findFlag = true;				
				}		
				else break;
			}
			tempmodelImage.release();
			scoreImage.release();
			mt.unlock();
		}
		);
		for (int m = 0; m < Row.size() ; m++)
		{
			for (int n = m+1; n < Row.size() ; n++)
			{
				if (abs(Col[n] - Col[m]) < modelImage.rows*0.8  && abs(Row[n] - Row[m]) < modelImage.cols*0.8)
				{
					if (Score[n] < Score[m])
					{
						Row.erase(Row.begin() + n);
						Col.erase(Col.begin() + n);
						Scale.erase(Scale.begin() + n);
						Angle.erase(Angle.begin() + n);
						Score.erase(Score.begin() + n);
						n--;
						continue;
					}
					else if (Score[n] >= Score[m])
					{
						swap(Row[m], Row[n]);
						swap(Col[m], Col[n]);
						swap(Scale[m], Scale[n]); 
						swap(Angle[m], Angle[n]); 
						swap(Score[m], Score[n]);
						Row.erase(Row.begin() + n);
						Col.erase(Col.begin() + n);
						Scale.erase(Scale.begin() + n);
						Angle.erase(Angle.begin() + n);
						Score.erase(Score.begin() + n);
						n = m ;
					}
				
				}
			}
		}
		*pRow = Row;
		*pCol = Col;
		*pScale = Scale;
		*pAngle = Angle;
		*pScore = Score;
	}
}
void imrotate(Mat& img, Mat& newIm, double angle) 
{
	int heightNew = int(img.cols*fabs(sin(angle*3.14 / 180)) + img.rows * fabs(cos(angle*3.14 / 180)));
	int widthNew = int(img.rows*fabs(sin(angle*3.14 / 180)) + img.cols * fabs(cos(angle*3.14 / 180)));
	int len = max(img.cols, img.rows);
	Point2f pt(img.cols / 2., img.rows / 2.);
//	Point2f pt(len / 2., len / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	r.at<double>(0, 2) += (widthNew - img.cols) / 2;
	r.at<double>(1, 2) += (heightNew - img.rows) / 2;	
	warpAffine(img, newIm, r, Size(widthNew, heightNew), INTER_LINEAR, BORDER_REPLICATE);
}
