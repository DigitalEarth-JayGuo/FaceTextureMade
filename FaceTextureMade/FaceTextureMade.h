#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string.h>
using namespace cv;
class FaceTextureMade
{
public:
	FaceTextureMade(void);
	~FaceTextureMade(void);
	void detectFace( cv::Mat& inImage,cv::Mat& outImage, double scale = 1,bool bIsCircled = false);
	void grabCutFace(cv::Mat& inImage,cv::Mat& outImage,int iterNum = 1,float scale = 1);
private:
	void calculatePix(cv::Mat& inImage,int type);
	void changePix(cv::Mat& inImage, int type, float inB=0, float inG=0, float inR=0);
	void removeErro(cv::Mat& inImage);
	void changeBrightness(cv::Mat& inImage, float inValue);
	void changeSaturature(cv::Mat& inImage, float inValue);
	void fillColor(cv::Mat& inImage, Vec3f inValue);
	void calculateAveragePix(cv::Mat& inImage, Vec3f& outValue);
	void mixPix(cv::Mat& inSrc, cv::Mat& inOutDes, float inValue = 1.0);
	void mixPixWithColor(cv::Mat& backImage,cv::Mat& frontImage, Vec3f& backColor, Vec3f& frontColor);
	Vec3f getROIColor(cv::Mat& inImage);
	void mixPixWithPointLight(cv::Mat& inSrc1, cv::Mat& inSrc2, cv::Mat& outPut);

	///**********************************************  
	//*by 垚  
	//*Windows7 +Visual studio 2010  
	//*功能 -- 双边滤波  
	//*input_img  -- 【输入】  
	//*output_img -- 【输出】  
	//*sigmaR -- 【输入】拉普拉斯方差  
	//*sigmaS -- 【输入】高斯方差  
	//*d      -- 【输入】半径  
	//***********************************************/  
	void bilateralBlur(cv::Mat &input_img, cv::Mat &output_img, float sigmaS, float sigmaR, int length);
};

