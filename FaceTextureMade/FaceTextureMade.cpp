
#include "FaceTextureMade.h"
#include <vector>
#include <iostream>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace cv;
FaceTextureMade::FaceTextureMade(void)
{
}


FaceTextureMade::~FaceTextureMade(void)
{
}

void FaceTextureMade::detectFace( cv::Mat& inImage,cv::Mat& outImage, double scale,bool bIsCircled)
{
	
	string cascadeName = "Data/haarcascade_frontalface_alt.xml";
	//string nestedCascadeName = "F:\\MyProjections\\FaceTextureMade\\FaceTextureMade\\Data/haarcascade_eye_tree_eyeglasses.xml";
	CascadeClassifier cascade, nestedCascade;


// 	if (!nestedCascade.load( nestedCascadeName ))
// 	{
// 		cout<<"nestedCascadeName is not read!"<<endl;
// 	}
	
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	Mat gray, smallImg( cvRound (inImage.rows/scale), cvRound(inImage.cols/scale), CV_8UC1 );

	cvtColor( inImage, gray, CV_BGR2GRAY );
	resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
	equalizeHist( smallImg, smallImg );

	t = (double)cvGetTickCount();

	if (!cascade.load( cascadeName ))
	{
		cout<<"cascadeName is not read!"<<endl;
		outImage = inImage;
	}
	else
	{
		cascade.detectMultiScale( smallImg, faces,
			1.1, 2, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			|CV_HAAR_SCALE_IMAGE
			,
			Size(30, 30) );
		t = (double)cvGetTickCount() - t;
		printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
		for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
		{
			Mat smallImgROI;
			vector<Rect> nestedObjects;
			Point center;
			int radius;

			double aspect_ratio = (double)r->width/r->height;
			if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
			{
				center.x = cvRound((r->x + r->width*0.5)*scale);
				center.y = cvRound((r->y + r->height*0.5)*scale);
				radius = cvRound((r->width + r->height)*0.25*scale);

				if (bIsCircled)
				{
					const static Scalar colors[] =  { CV_RGB(0,0,255),
						CV_RGB(0,128,255),
						CV_RGB(0,255,255),
						CV_RGB(0,255,0),
						CV_RGB(255,128,0),
						CV_RGB(255,255,0),
						CV_RGB(255,0,0),
						CV_RGB(255,0,255)} ;
					Scalar color = colors[i%8];
					circle( inImage, center, radius, color, 3, 8, 0 );
				}



				CvSize size;
				size.height = radius *2 ;
				size.width = radius*2 ;
				CvRect rec;
				rec.x = center.x - radius;
				rec.y = center.y - radius;
				rec.height = size.height;
				rec.width = size.width;
				IplImage* pSrc;
				pSrc = &IplImage(inImage);
				cvSetImageROI(pSrc,rec);

				IplImage* pDest;
				pDest = cvCreateImage(size,pSrc->depth,pSrc->nChannels);
				cvCopy(pSrc,pDest);
				cvResetImageROI(pDest);

				outImage = cv::Mat(pDest,0);



				//cvSaveImage("C:\\Users\\JayGuo\\Desktop\\Roi.jpg",pDest);
			}



		}
	}
	
	
	
}

void FaceTextureMade::grabCutFace( cv::Mat& inImage,cv::Mat& outImage,int iterNum,float scale )
{
	//grab cut the image
	CvRect foreRec;
	cv::Mat bgModel, fgModel;
	

	cv::Mat mask;
	mask.create( inImage.size(), CV_8UC1);

	if( !mask.empty() )
		mask.setTo(Scalar::all(GC_BGD));

	mask.setTo( GC_BGD );
	foreRec.x = inImage.cols * 0.01 * scale;
	foreRec.y = inImage.rows * 0.01 * scale;
	foreRec.width = inImage.cols - foreRec.x * 2;
	foreRec.height = inImage.rows - foreRec.y * 2;
	(mask(foreRec)).setTo( Scalar(GC_PR_FGD));

	cv::grabCut(inImage,mask,foreRec,bgModel,fgModel,iterNum,cv::GC_INIT_WITH_MASK);
	mask = mask & 1;
	
	//cv::Mat foreground(inImage.size(),CV_8UC4,cv::Scalar(255,255,255,0));
	//threshold(1.0-mask,mask,0.2,0.6,cv::THRESH_BINARY_INV);

	cv::GaussianBlur(mask,mask,Size(21,21),11);

	//////////////////////////////////////////////////////////////////////////
	//skin color begin
	//////////////////////////////////////////////////////////////////////////
	cv::Mat test = cv::imread("C:\\Users\\\guozh\\Desktop\\LRHLJ14069_miners_face_S001_diff_NQ.jpg");
	
	
	cv::Mat skinColor;
	skinColor.create(test.size(), CV_8UC3);
	fillColor(skinColor, getROIColor(inImage));
	Vec3f frontColor = getROIColor(inImage);

	//back
// 	cv::Mat backColor;
// 	backColor.create(test.size(), CV_8UC3);
//	fillColor(backColor,Vec3f(157,180,236));
	Vec3f backColor = Vec3f(157, 180, 236);

	bilateralBlur(inImage, inImage, 1, 1, 10);
	//mixPixWithColor(test, inImage, backColor, frontColor);
	//addWeighted(test,1, skinColor, 1.5, 0, test);
	//mixPix(skinColor, test);
	
	//////////////////////////////////////////////////////////////////////////
	//skin color end
	//////////////////////////////////////////////////////////////////////////

	cv::resize(inImage,inImage,cv::Size(200,170));
	cv::resize(mask,mask,cv::Size(200,170));
	
	cv::Mat imageROI;
	imageROI = test(cv::Rect(256 - inImage.cols / 2,128 - inImage.rows / 2 - 30 ,inImage.cols,inImage.rows));
	
	
	//////////////////////////////////////////////////////////////////////////
	//modify color start
	//////////////////////////////////////////////////////////////////////////


// 	calculatePix(imageROI, CV_BGR2HSV);
// 	calculatePix(inImage, CV_BGR2HSV);

	//changePix(inImage, CV_BGR2HSV, 10, -10, 30);
	//removeErro(inImage);
	//////////////////////////////////////////////////////////////////////////
	//modify color end
	//////////////////////////////////////////////////////////////////////////
	
	//cvtColor(inImage, inImage, CV_GRAY2BGR);
	cv::Mat backImg;
	backImg.create(imageROI.size(), CV_8UC3);
	imageROI.copyTo(backImg);
	
	inImage.copyTo(imageROI,mask);
// 	cv::Mat outPut;
// 	outPut.create(imageROI.size(), CV_8UC3);
// 	mixPixWithPointLight(imageROI, backImg, outPut);
// 	imwrite("C:\\Users\\guozh\\Desktop\\front.jpg", outPut);
	
// 	cvtColor(imageROI, imageROI, CV_BGR2GRAY);
// 	cv::Mat contours;
	//cv::Canny(imageROI, contours, 125, 350);
	//cv::medianBlur(imageROI, imageROI, 5);
	
// 	CvAdaptiveSkinDetector skinDetector(1, CvAdaptiveSkinDetector::MORPHING_METHOD_NONE);
// 	IplImage* image = &IplImage(inImage);
// 	IplImage* mask1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
// 	cvZero(mask1);
// 	skinDetector.process(image, mask1);
// 	cvSaveImage("C:\\Users\\guozh\\Desktop\\test.jpg", mask1);
	//cv::imwrite("C:\\Users\\guozh\\Desktop\\test.jpg", test);
	outImage = test;

}

void FaceTextureMade::changePix(cv::Mat& inImage, int type, float inB, float inG, float inR)
{
	int reType;
	switch (type)
	{
	case CV_BGR2HSV:
		reType = CV_HSV2BGR;
		break;
	case CV_BGR2Lab:
		reType = CV_Lab2BGR;
		break;
	case CV_BGR2HLS:
		reType = CV_HLS2BGR;
		break;
	default:
		break;
	}

	cvtColor(inImage, inImage, type);

	cv::Mat_<cv::Vec3b>::iterator it = inImage.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itEnd = inImage.end<cv::Vec3b>();
	for (; it != itEnd; ++it)
	{
		(*it)[0] += inB;
		(*it)[1] += inG;
		(*it)[2] += inR;
		//(*it)[0] += 15;
	}

	cvtColor(inImage, inImage, reType);
	
	
}

void FaceTextureMade::removeErro(cv::Mat& inImage)
{
	cv::Mat_<cv::Vec3b>::iterator it = inImage.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itEnd = inImage.end<cv::Vec3b>();
	for (; it != itEnd; ++it)
	{
		(*it)[0] = (*it)[0] > 255 ? 255 : (*it)[0];
		(*it)[1] = (*it)[1] > 255 ? 255 : (*it)[1];
		(*it)[2] = (*it)[2] > 255 ? 255 : (*it)[2];
		//(*it)[0] += 15;
	}
}

void FaceTextureMade::changeBrightness(cv::Mat& inImage, float inValue)
{
// 	cv::Mat_<cv::Vec3b>::iterator it = inImage.begin < cv::Vec3b >();
// 	cv::Mat_<cv::Vec3b>::iterator itEnd = inImage.end<cv::Vec3b>();
// 	for (; it != itEnd; ++it)
// 	{
// 		countB += (*it)[0];
// 		countG += (*it)[1];
// 		countR += (*it)[2];
// 		//(*it)[0] += 15;
// 	}
}

void FaceTextureMade::changeSaturature(cv::Mat& inImage, float inValue)
{

}

void FaceTextureMade::fillColor(cv::Mat& inImage, Vec3f inValue)
{
	cv::Mat_<cv::Vec3b>::iterator it = inImage.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itEnd = inImage.end<cv::Vec3b>();
	for (; it != itEnd; ++it)
	{
		*it = inValue;
// 		(*it)[0];
// 		countG += (*it)[1];
// 		countR += (*it)[2];
		//(*it)[0] += 15;
	}
}

void FaceTextureMade::calculateAveragePix(cv::Mat& inImage, Vec3f& outValue)
{
	float countB = 0;
	float countG = 0;
	float countR = 0;
	cv::Mat_<cv::Vec3b>::iterator it = inImage.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itEnd = inImage.end<cv::Vec3b>();
	for (; it != itEnd; ++it)
	{
		countB += (*it)[0];
		countG += (*it)[1];
		countR += (*it)[2];
		//(*it)[0] += 15;
	}
	float pixNum = inImage.rows * inImage.cols;
	outValue[0] = countB / pixNum;
	outValue[1] = countG / pixNum;
	outValue[2] = countR / pixNum;
}

void FaceTextureMade::mixPix(cv::Mat& inSrc, cv::Mat& inOutDes, float inValue)
{
	cv::Mat_<cv::Vec3b>::iterator itSrc = inSrc.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itSrcEnd = inSrc.end<cv::Vec3b>();

	cv::Mat_<cv::Vec3b>::iterator itDes = inOutDes.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itDesEnd = inOutDes.end<cv::Vec3b>();

	for (; itSrc != itSrcEnd&& itDes!=itDesEnd; ++itSrc,++itDes)
	{
		//(*itDes) += (*itSrc) * inValue;
		for (int i =0; i<3;i++)
		{
			
			//(*itDes)[i] = ((*itSrc)[i] + (*itDes)[i]) / 2;
			float a = (*itSrc)[i];
			float b = (*itDes)[i];

			if(a <= 0.5)

			{

				b = 1 - (1 - b) / (2 * a);

			}

			else

			{

				b = b / (2 * (1 - a));

			}
			
		}
		
		
	}
}

void FaceTextureMade::mixPixWithColor(cv::Mat& backImage, cv::Mat& frontImage, Vec3f& backColor, Vec3f& frontColor)
{
	cv::Mat_<cv::Vec3b>::iterator itBack = backImage.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itBackEnd = backImage.end<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itFront = frontImage.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itFrontEnd = frontImage.end<cv::Vec3b>();
	for (; itBack != itBackEnd; ++itBack)
	{
		(*itBack) = ((Vec3f)(*itBack) + frontColor) / 2;
		
	}

	for (; itFront != itFrontEnd; ++itFront)
	{
		
		(*itFront) = ((Vec3f)(*itFront) * 0.8 + backColor * 0.2) ;
	}
}

cv::Vec3f FaceTextureMade::getROIColor(cv::Mat& inImage)
{
	cv::Mat colorROI;
	Vec2f center;
	center[0] = inImage.cols / 2;
	center[1] = inImage.rows / 2;
	float width = inImage.cols / 5;
	float length = inImage.rows / 5;

	colorROI = inImage(cv::Rect(center[0] - width, center[1] - length, width, length));

	//draw rec
	rectangle(inImage, cv::Rect(center[0] - width / 2, center[1] - length / 2, width, length), cv::Scalar(255, 0, 0), 3);

	//front
	Vec3f roiColor;
	calculateAveragePix(colorROI, roiColor);
	
	return roiColor;
}

void FaceTextureMade::mixPixWithPointLight(cv::Mat& inSrc1, cv::Mat& inSrc2, cv::Mat& outPut)
{
	float a = 0;

	float b = 0;

 	for (int index_row = 0; index_row < inSrc1.rows; index_row++)

	{

		for (int index_col = 0; index_col < inSrc1.cols; index_col++)

		{

			for (int index_c = 0; index_c < 3; index_c++)

			{

				a = inSrc1.at<Vec3b>(index_row, index_col)[index_c];

				b = inSrc2.at<Vec3b>(index_row, index_col)[index_c];

				if (b <= 2 * a - 1)

				{

					outPut.at<Vec3b>(index_row, index_col)[index_c] = 2 * a - 1;

				}

				else if (b <= 2 * a)

				{

					outPut.at<Vec3b>(index_row, index_col)[index_c] = b;

				}

				else

				{

					outPut.at<Vec3b>(index_row, index_col)[index_c] = 2 * a;

				}

			}

		}

	}
}

void FaceTextureMade::bilateralBlur(cv::Mat &input_img, cv::Mat &output_img, float sigmaS, float sigmaR, int length)
{
	// Create Gaussian/Bilateral filter --- mask ---       
	int i, j, x, y;
	int radius = (int)length / 2;//∞Îæ∂  
	int m_width = input_img.rows;
	int m_height = input_img.cols;
	std::vector<float> mask(length*length);

	//∂®“Â”Ú∫À  
	for (i = 0; i < length; i++)
	{
		for (j = 0; j < length; j++)
		{
			mask[i*length + j] = exp(-(i*i + j*j) / (2 * sigmaS*sigmaS));
		}
	}
	float sum = 0.0f, k = 0.0f;
	for (x = 0; x < m_width; x++)
	{
		unsigned char *pin = input_img.ptr<unsigned char>(x);
		unsigned char *pout = output_img.ptr<unsigned char>(x);
		for (y = 0; y < m_height; y++)
		{
			int centerPix = y;
			for (i = -radius; i <= radius; i++)
			{
				for (j = -radius; j <= radius; j++)
				{
					int m = x + i, n = y + j;
					if (x + i > -1 && y + j > -1 && x + i < m_width && y + j < m_height)
					{
						unsigned char value = input_img.at<unsigned char>(m, n);
						//spatial diff  
						float euklidDiff = mask[(i + radius)*length + (j + radius)];
						float intens = pin[centerPix] - value;//÷µ”Ú∫À  
						float factor = (float)exp(-0.5 * intens / (2 * sigmaR*sigmaR)) * euklidDiff;
						sum += factor * value;
						k += factor;
					}
				}
			}
			pout[y] = sum / k;
			sum = 0.0f;
			k = 0.0f;
		}
	}
}

void FaceTextureMade::calculatePix(cv::Mat& inImage, int type)
{
	int reType;
	switch (type)
	{
	case CV_BGR2HSV:
		reType = CV_HSV2BGR;
		break;
	case CV_BGR2Lab:
		reType = CV_Lab2BGR;
		break;
	case CV_BGR2HLS:
		reType = CV_HLS2BGR;
		break;
	default:
		break;
	}

	cvtColor(inImage, inImage, type);
	float countB = 0;
	float countG = 0;
	float countR = 0;
	cv::Mat_<cv::Vec3b>::iterator it = inImage.begin < cv::Vec3b >();
	cv::Mat_<cv::Vec3b>::iterator itEnd = inImage.end<cv::Vec3b>();
	for (; it != itEnd; ++it)
	{
		countB += (*it)[0];
		countG += (*it)[1];
		countR += (*it)[2];
		//(*it)[0] += 15;
	}


	cvtColor(inImage, inImage, reType);

	//analy
	
	int pixNum = inImage.rows * inImage.cols;
	cout <<"B: "<< countB/pixNum  << endl;
	cout << "G: " << countG/pixNum  << endl;
	cout << "R: " << countR/pixNum  << endl;



	
}
