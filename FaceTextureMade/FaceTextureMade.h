#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string.h>
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
};

