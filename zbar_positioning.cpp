#include <opencv/cv.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <zbar.h>
#include <iostream>

#define PI 3.1415926

using namespace std;
using namespace cv;
using namespace zbar;

int main()
{
    //打开0号摄像头，从摄像头中获取视频
    VideoCapture capture(0);
    //摄像头不存在
    if(!capture.isOpened())
        return 1;
    //创建窗口，名称为“debug”，自动调整大小
    cvNamedWindow("debug",CV_WINDOW_AUTOSIZE);
    //灰度图
    IplImage* grayFrame=0;
    //创建zbar图像扫描器
    ImageScanner scanner;
    //配置zbar图片扫描器
    scanner.set_config(ZBAR_NONE,ZBAR_CFG_ENABLE,1);
    Mat img;
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoints;
    Point3f p0(0,0,0);
    Point3f p1(100,0,0);
    Point3f p2(0,100,0);
    Mat cameraMatrix = (Mat_<float>(3,3)<<795.3635424012639, 0, 296.5970906573888, 0, 795.7472390480578, 245.5318558470171, 0, 0, 1);
    Mat distCoeffs = (Mat_<float>(1,5)<<0.05787003020570314, 1.290330734507909, -0.007109846921777598, -0.002686357312794918, -7.856551180361896);
    while(1)
    {
	objectPoints.clear();
	imagePoints.clear();
	capture >> img;
	IplImage frame0 = IplImage(img);
        //从摄像头中抓取一帧
        IplImage* frame = &frame0;
        //图像不为空
        if(frame)
        {
            //如果灰度图没有创建，就创建一个和原图一样大小的灰度图（8位色深，单通道）
            if(!grayFrame)
                grayFrame=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
            //原图转灰度图
            cvCvtColor(frame,grayFrame,CV_BGR2GRAY);
            //显示灰度图
            cvShowImage("debug",grayFrame);
            //创建zbar图像
            Image image(frame->width,frame->height,"Y800",grayFrame->imageData,frame->width*frame->height);
            //扫描图像，识别二维码，获取个数
            int symbolCount=scanner.scan(image);
            //获取第一个二维码
            Image::SymbolIterator symbol=image.symbol_begin();
            //遍历所有识别出来的二维码
            while(symbolCount--)
            {
		string numbers;
		
                //输出二维码内容
                numbers = symbol->get_data();
                char number = numbers[0];
                //获得四个点的坐标
    		double x0=symbol->get_location_x(0);
   	 	double y0=symbol->get_location_y(0);
    		double x1=symbol->get_location_x(1);
    		double y1=symbol->get_location_y(1);
    		double x2=symbol->get_location_x(2);
    		double y2=symbol->get_location_y(2);
    		double x3=symbol->get_location_x(3);
    		double y3=symbol->get_location_y(3);
   
    		//两条对角线的系数和偏移
   	 	double k1=(y2-y0)/(x2-x0);
    		double b1=(x2*y0-x0*y2)/(x2-x0);
    		double k2=(y3-y1)/(x3-x1);
    		double b2=(x3*y1-x1*y3)/(x3-x1);
    		//两条对角线交点的X坐标
    		double crossX=-(b1-b2)/(k1-k2);
    		double crossY=k1*crossX+b1;
		Point2f p2D(crossX, crossY);
		if (number == '0')
		{
			objectPoints.push_back(p0);
			imagePoints.push_back(p2D);
		}
		else if (number == '1')
		{
			objectPoints.push_back(p1);
			imagePoints.push_back(p2D);
		}
		else if (number == '2')
		{
			objectPoints.push_back(p2);
			imagePoints.push_back(p2D);
		}
                //下一个二维码
                ++symbol;
            }
        if (objectPoints.size() >= 3 && imagePoints.size() >= 3) {
		Mat rvec;
    		Mat tvec;
		Mat rotM, rotT;
		solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    		Rodrigues(rvec, rotM);  //将旋转向量变换成旋转矩阵
    		Rodrigues(tvec, rotT);
		double theta_x, theta_y, theta_z;
		//根据旋转矩阵求出坐标旋转角
    		theta_x = atan2(rotM.at<double>(2, 1), rotM.at<double>(2, 2));
    		theta_y = atan2(-rotM.at<double>(2, 0),
    		sqrt(rotM.at<double>(2, 1)*rotM.at<double>(2, 1) + rotM.at<double>(2, 2)*rotM.at<double>(2, 2)));
    		theta_z = atan2(rotM.at<double>(1, 0), rotM.at<double>(0, 0));

    		//将弧度转化为角度
    		theta_x = theta_x * (180 / PI);
    		theta_y = theta_y * (180 / PI);
    		theta_z = theta_z * (180 / PI);
		cout << "x:" << tvec.at<double>(0, 0) << " y:" << tvec.at<double>(1, 0) << " z:" << tvec.at<double>(2, 0) << endl;
		cout << "x轴:" << theta_x << " y轴:" << theta_y<< " z轴:" << theta_z << endl;
	}
	
        }
        //延时50ms，如果按了ESC就退出
        if(cvWaitKey(50)== ' ')
            break;
    }
    //释放灰度图
    cvReleaseImage(&grayFrame);
    //销毁窗口
    cvDestroyWindow("debug");
    //释放内存
    //cvReleaseCapture(&capture);
    return 0;
}
