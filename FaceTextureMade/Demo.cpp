#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include "FaceTextureMade.h"

using namespace std;
using namespace cv;

void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
	cv::Mat &output, cv::Point2i location)
{
	background.copyTo(output);


	// start at the row indicated by location, or at row 0 if location.y is negative.
	for(int y = std::max(location.y , 0); y < background.rows; ++y)
	{
		int fY = y - location.y; // because of the translation

		// we are done of we have processed all rows of the foreground image.
		if(fY >= foreground.rows)
			break;

		// start at the column indicated by location,

		// or at column 0 if location.x is negative.
		for(int x = std::max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; // because of the translation.

			// we are done with this row if the column is outside of the foreground image.
			if(fX >= foreground.cols)
				break;

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity =
				((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

				/ 255.;


			// and now combine the background and foreground pixel, using the opacity,

			// but only if opacity > 0.
			for(int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y*output.step + output.channels()*x + c] =
					backgroundPx * (1.-opacity) + foregroundPx * opacity;
			}
		}
	}
}


int main( int argc, const char** argv )
{
// 	cv::Mat foreground = imread("C:\\Users\\JayGuo\\Desktop\\logo.png", -1);
// 	cv::Mat background = imread("C:\\Users\\JayGuo\\Desktop\\meitu_00004.jpg");
// 	cv::Mat result;
// 
// 	overlayImage(background, foreground, result, cv::Point(110,110)	//cv::imshow( "result", result );


	cv::Mat image = cv::imread("C:\\Users\\guozh\\Desktop\\meitu_00001.jpg");
	cv::Mat detectFaceImg;
	cv::Mat grabFaceImg;
	FaceTextureMade* faceTextureMade = new FaceTextureMade;
	faceTextureMade->detectFace(image,detectFaceImg,1,false);
	//resize the image before cut
	cv::Mat resizedImg;
	cv::Size size;
	size.height = 512;
	size.width = size.height / detectFaceImg.rows * detectFaceImg.cols;
	cv::resize(detectFaceImg,resizedImg,size);
 	faceTextureMade->grabCutFace(resizedImg,grabFaceImg,3,5);
	

	
	cv::imshow( "result", grabFaceImg );
	waitKey(0);

    cvDestroyWindow("result");

	delete faceTextureMade;

    return 0;
}

