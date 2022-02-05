//
//  main.cpp
//  License Plate Detector
//
//  Created by NJR on 2022/1/18.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    
    VideoCapture cap(0);
    Mat img;
    
    CascadeClassifier plateCascade;
    plateCascade.load("/Users/admin/Desktop/openCV /OpenCV_Study/OpenCV_Study/resources/haarcascade_russian_plate_number.xml");
    if (plateCascade.empty()) cout << "XML File not loaded" << endl;
    
    vector<Rect> plates;
    
    while (1){
        cap.read(img);
        plateCascade.detectMultiScale(img, plates, 1.1, 10);
        
        for (int i = 0 ; i < plates.size() ; i++){
            rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255,0,255),2);
        }
        
        imshow("Image", img);
        waitKey(1);
    }
    
    
    
    
    return 0;
}
