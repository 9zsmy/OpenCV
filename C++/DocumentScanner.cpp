//
//  main.cpp
//  Document Scanner
//
//  Created by NJR on 2022/1/18.
//

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat imgGrey, imgCanny, imgBlur, imgDil, imgOrigin, imgThre, imgWarp, imgCrop;
vector<Point> initialPoints, docPoints;

float w = 420, h = 596;

Mat preProcessing(Mat img){
    
    cvtColor(img, imgGrey, COLOR_BGR2GRAY);
    GaussianBlur(img, imgBlur, Size(3,3), 3,0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
    dilate(imgCanny, imgDil, kernel);
    
    return imgDil;
}


vector<Point> getContours(Mat imgDil){
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    
    vector<Point> biggest;
    int maxArea = 0;
    for (int i = 0 ; i < contours.size() ; i++){
        int area = contourArea(contours[i]);
        if (area > 1000){
            //封闭轮廓的周长或曲线的弧长
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02*peri, true);
            
            drawContours(imgOrigin, conPoly, i, Scalar(255,0,255),5);
            rectangle(imgOrigin, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0),4);
            if (area > maxArea && conPoly[i].size() == 4){
                drawContours(imgDil, conPoly, i, Scalar(255,0,255),5);
//                rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0),4);
                biggest = {conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3]};
                maxArea = area;
                
            }
            
//            drawContours(imgDil, conPoly, i, Scalar(255,0,255),2);
        }
    }
    return biggest;
}

void drawPoints(vector<Point> points, Scalar color){
    for (int i = 0 ; i < points.size() ; i++){
        circle(imgOrigin, points[i], 5, color, FILLED);
        putText(imgOrigin, to_string(i), points[i], FONT_HERSHEY_PLAIN, 4, color, 4);
    }
}

vector<Point> reorder(vector<Point> points){
    vector<Point> newPoints;
    vector<int> sumPoints, subPoints;
    
    for (int i = 0 ; i < 4 ; i++){
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }
    
    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
    
    return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h){
    Point2f src[4] = {points[0], points[1], points[2], points[3]};
    Point2f dst[4] = {{0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h}};
    
    //透视变换，变形
    Mat matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(w,h));
    
    return imgWarp;
}

void Capture(){
    //视频
    VideoCapture cap(0);
    while (1){
        Mat imgWarp, imgCrop;
        cap.read(imgOrigin);
        //flip(imgOrigin, imgOrigin, 1);

        imshow("imgOrigin",imgOrigin);

        imgThre = preProcessing(imgOrigin);

        vector<Point> initialPoints = getContours(imgThre);
        if (initialPoints.size() != 4) continue;
        vector<Point> docPoints = reorder(initialPoints);
        imgWarp = getWarp(imgOrigin, docPoints, w, h);

        int cropVal = 5;
        Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
        imgCrop = imgWarp(roi);
        imshow("imgCrop", imgCrop);


        waitKey(1);
    }

}

int main(int argc, const char * argv[]) {
    // insert code here...
    //string path = "/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/paper.jpg";
    string path = "/Users/admin/Desktop/WechatIMG13.jpeg";
    imgOrigin = imread(path);
    imshow("imgOrigin", imgOrigin);
    //resize(imgOrigin, imgOrigin, Size(), 0.5,0.5);
    // 1:preprocessing
    imgThre = preProcessing(imgOrigin);

    // 2:GetConTours - Biggest
    initialPoints = getContours(imgDil);
    //drawPoints(initialPoints, Scalar(0,0,255));
    //重新排列四个点的位置
    docPoints = reorder(initialPoints);
    //drawPoints(docPoints, Scalar(0,255,0));
    imgWarp = getWarp(imgOrigin, docPoints, w, h);
    //Crop
    int cropVal = 5;
    Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
    imgCrop = imgWarp(roi);

    imshow("imgOrigin", imgOrigin);
    imshow("imgDil", imgDil);
    imshow("imgCrop", imgCrop);
    waitKey(0);
    
    return 0;
    
}
