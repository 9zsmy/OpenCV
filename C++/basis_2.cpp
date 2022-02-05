//
//  Chapter2.cpp
//  
//
//  Created by NJR on 2021/11/27.
//


//
//  Chapter1.cpp
//  OpenCV_Study
//
//  Created by NJR on 2021/11/27.
//

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int hmin = 0 , smin = 0 , vmin = 0;
int hmax = 179 , smax = 255 , vmax = 255;
Mat imgHSV,mask;

class Demo{
public:
    void image_test(){
        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/test.png";
        Mat img = imread(path);
        imshow("Image", img);
        waitKey(0);
    }
    void video_test(){
        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/test_video.mp4";
        VideoCapture cap(path);
        Mat img;
        while(true){
            cap.read(img);
            imshow("Image", img);
            waitKey(20);
        }
    }
    void webcam_test(){
        VideoCapture cap(0);
        Mat img;
        while(true){
            cap.read(img);
            imshow("Image", img);
            waitKey(1);
        }
    }
    void BasicFunction(){
        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/test.png";
        Mat img = imread(path);
        Mat imgGrey, imgBlur, imgCanny, imgDil, imgErode;
        
        //灰度
        cvtColor(img, imgGrey, COLOR_BGR2GRAY);
        //高斯模糊
        GaussianBlur(img, imgBlur, Size(7,7), 3,0);
        //边缘检测
        Canny(imgBlur, imgCanny, 50, 150);
        //图像膨胀
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(imgCanny, imgDil, kernel);
        //图像侵蚀
        erode(imgDil, imgErode, kernel);
        
        imshow("Image", img);
        imshow("Image Grey", imgGrey);
        imshow("Image Blur", imgBlur);
        imshow("Image Canny", imgCanny);
        imshow("Image Dilation", imgDil);
        imshow("Image Erode", imgErode);
        
        waitKey(0);
    }
    void ResizeAndCrop(){
        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/test.png";
        Mat img = imread(path);
        Mat imgResize, imgCrop;
        
        //调整大小，0.5，0.5是将长宽缩小原来的二分之一，如果想修改为具体的尺寸则在Size()中填写长和宽。
        resize(img, imgResize, Size(),0.5,0.5);
        //裁剪,200是x，100是y，即从距原图横向200，纵向100处裁剪出长为300，宽为250的图像。
        Rect roi(200,100,300,250);
        imgCrop = img(roi);
        
        imshow("Image", img);
        imshow("Image Resize", imgResize);
        imshow("Image Crop", imgCrop);

        waitKey(0);
    }
    void DrawShapeAndText(){
        //Blank Image
        Mat img(512,512,CV_8UC3,Scalar(255,255,255));
        
        //圆形
        circle(img, Point(256,256), 155, Scalar(0,169,255),FILLED);
        //矩形
        rectangle(img, Point(130,226), Point(380,286), Scalar(255,255,255),FILLED);
        //线
        line(img, Point(130,296), Point(380,296), Scalar(255,255,255),2);
        //文字
        putText(img, "Hello OpenCV!", Point(137,262), FONT_HERSHEY_TRIPLEX, 1, Scalar(40,169,255),5);
        
        imshow("Image", img);
        
        waitKey(0);
    }
    
    void WarpImage(){
        float w = 250, h = 350;
        Mat matrix,imgWarp;
        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/cards.jpg";
        Mat img = imread(path);
        
        Point2f src[4] = {{776,109},{1015,83},{841,358},{1117,331}};
        Point2f dst[4] = {{0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h}};
        
        //透视变换，变形
        matrix = getPerspectiveTransform(src, dst);
        warpPerspective(img, imgWarp, matrix, Point(w,h));
        
        for (int i = 0 ; i < 4 ; i++){
            circle(img, src[i], 10, Scalar(0,169,255),FILLED);
        }
        
        imshow("Image", img);
        imshow("Image Warp",imgWarp);
        waitKey(0);
    }
    
    //ColorPicker
    void colorDetection(){
//        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/shapes.png";
        VideoCapture cap(0);
        //Mat img = imread(path);
        Mat img;
        namedWindow("TrackBars",(640,200));
        
        createTrackbar("HueMin", "TrackBars", &hmin,179);
        createTrackbar("HueMax", "TrackBars", &hmax,179);
        createTrackbar("SatMin", "TrackBars", &smin,255);
        createTrackbar("SatMax", "TrackBars", &smax,255);
        createTrackbar("ValMin", "TrackBars", &vmin,255);
        createTrackbar("ValMax", "TrackBars", &vmax,255);
        
        while(true){
            cap.read(img);
            flip(img,img,1);
            cvtColor(img, imgHSV, COLOR_BGR2HSV);
            Scalar lower(hmin,smin,vmin);
            Scalar upper(hmax,smax,vmax);
            inRange(imgHSV, lower, upper, mask);
            imshow("MASK", mask);
            cout << hmin << " " << hmax << " " << smin << " " << smax << " " << vmin << " " << vmax << " " << endl;
            //imshow("IMAGE HSV",imgHSV);
            imshow("IMAGE",img);
            waitKey(1);
        }
    }
    
    void getContours(Mat imgDil, Mat img){
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
//        drawContours(img, contours, -1, Scalar(255,0,255),2,LINE_AA);
        
        for (int i = 0 ; i < contours.size() ; i++){
            int area = contourArea(contours[i]);
            vector<vector<Point>> conPoly(contours.size());
            vector<Rect> boundRect(contours.size());
            
            string objectType;
            
            if (area > 1000){
                float peri = arcLength(contours[i], true);
                approxPolyDP(contours[i], conPoly[i], 0.02*peri, true);
                
                //绘制轮廓
                drawContours(img, conPoly, i, Scalar(255,0,255),2);
                cout << conPoly[i].size() << endl;
                
                //绘制边界矩形
                boundRect[i] = boundingRect(conPoly[i]);
                rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0));
                
                int objCor = (int)conPoly[i].size();
                //cout << "conPoly.size() = " << conPoly.size() << endl;
                
                if (objCor == 3) objectType = "Tri";
                else if (objCor == 4) {
                    float aspRatio= (float)boundRect[i].width / (float)boundRect[i].height;
                    if (aspRatio > 0.95 && aspRatio < 1.05) objectType = "Square";
                    else objectType = "Rect";
                    
                }
                else if (objCor > 4) objectType = "Circle";
                
                
                
                putText(img, objectType, {boundRect[i].x, boundRect[i].y}, FONT_HERSHEY_TRIPLEX, 1, Scalar(40,169,255),2);
            }
        }
    }
    
    void ShapesContourDetection(){
        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/shapes.png";
        Mat img = imread(path);
        Mat imgGrey, imgBlur, imgCanny, imgDil, imgErode;
        
        cvtColor(img, imgGrey, COLOR_BGR2GRAY);
        GaussianBlur(img, imgBlur, Size(7,7), 3,0);
        Canny(imgBlur, imgCanny, 50, 150);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(imgCanny, imgDil, kernel);
        
        getContours(imgDil, img);
        
        imshow("Image", img);;
    }
    
    void FaceDetection(){
        
        VideoCapture cap(0);
        Mat img;
        
        while (1){
            cap.read(img);
            CascadeClassifier faceCascade;
            faceCascade.load("/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/haarcascade_frontalface_default.xml");
            if (faceCascade.empty()) cout << "XML File not loaded" << endl;
            
            vector<Rect> faces;
            faceCascade.detectMultiScale(img, faces, 1.1, 10);
            
            for (int i = 0 ; i < faces.size() ; i++){
                rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255,0,255),2);
            }
            
            imshow("Image", img);
            waitKey(1);
        }
//        string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/test.png";
//        Mat img = imread(path);
        
        
    }
    
    
};


// ---------------------vedio---------------------
int main(int argc, const char * argv[]) {
    // insert code here...
    Chapter1 cp1;
    //cp1.image_test();
    //cp1.video_test();
    //cp1.webcam_test();
    //cp1.BasicFunction();
    //cp1.ResizeAndCrop();
    //cp1.DrawShapeAndText();
    //cp1.WarpImage();
    //cp1.colorDetection();
    //cp1.ShapesContourDetection();
    //cp1.FaceDetection();
    waitKey(0);
    return 0;
    
 
    //return 0;
}
