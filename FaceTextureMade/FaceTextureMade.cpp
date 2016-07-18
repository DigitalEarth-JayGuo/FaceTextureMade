#include "FaceTextureMade.h"
#include <vector>
#include <iostream>


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

	cv::Mat test = cv::imread("C:\\Users\\\guozh\\Desktop\\LRHLJ14069_miners_face_S001_diff_NQ.jpg");
	
	
	
	cv::resize(inImage,inImage,cv::Size(200,170));
	cv::resize(mask,mask,cv::Size(200,170));
	cv::Mat imageROI;
	imageROI = test(cv::Rect(256 - inImage.cols / 2,128 - inImage.rows / 2 - 30 ,inImage.cols,inImage.rows));
	cvtColor(inImage,inImage,CV_RGB2HLS);
	
	
	
	IplImage *grayImg = cvCreateImage(imageROI.size(),IPL_DEPTH_8U,1);
	cv::Mat gray;
	cvtColor(imageROI,gray,CV_RGB2GRAY);  
	grayImg = &IplImage(gray); 
	CvScalar scalar = cvAvg(grayImg); 
	IplImage * out = cvCreateImage(imageROI.size(),IPL_DEPTH_8U,3);;
	//cv::convertScaleAbs(inImage,out,0.4/scalar.val[0]);
	IplImage* image = &IplImage(inImage);
	

	cvConvertScale(image,out,0.5);  
	cv::Mat res = cv::Mat(out,0);
// 	for (int i = 0; i < inImage.cols; i++)
// 	{
// 		for (int j = 0; j<inImage.rows; j++)
// 		{
// 			//if (inImage.at<cv::Vec3b>(j,i)[0] + inImage.at<cv::Vec3b>(j,i)[1] + inImage.at<cv::Vec3b>(j,i)[2] > 50 && inImage.at<cv::Vec3b>(j,i)[0] + inImage.at<cv::Vec3b>(j,i)[1] + inImage.at<cv::Vec3b>(j,i)[2] < 240 )
// 			{
// 				inImage.at<cv::Vec3b>(j,i)[1] = 1.6 ;
// // 				inImage.at<cv::Vec3b>(j,i)[1] = (imageROI.at<cv::Vec3b>(j,i)[1]* 0.75 + inImage.at<cv::Vec3b>(j,i)[1]* 0.25) ;
// // 				inImage.at<cv::Vec3b>(j,i)[2] = (imageROI.at<cv::Vec3b>(j,i)[2]* 0.75 + inImage.at<cv::Vec3b>(j,i)[2]* 0.25) ;
// 			}
// 			
// 		}
// 	}
	cvtColor(inImage,inImage,CV_HLS2RGB);
	inImage.copyTo(imageROI,mask);
	
	cv::imwrite("C:\\Users\\guozh\\Desktop\\test.jpg",res);
	outImage = test;

}
