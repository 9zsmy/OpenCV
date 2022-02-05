//
//  main.cpp
//  OpenCV4
//
//  Created by NJR on 2022/1/9.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <opencv2/dnn.hpp>


using namespace std;
using namespace cv;

Point sp(-1,-1);
Point ep(-1,-1);
Mat temp;

double coefficient = 1;


class Demo{
public:
    
    //图像转换颜色模型为HSV和GREY
    void demo1(Mat &img){
        Mat HSV,GREY;
        cvtColor(img, HSV, COLOR_BGR2HSV);
        cvtColor(img, GREY, COLOR_BGR2GRAY);
        imshow("HSV", HSV);
        imshow("GREY", GREY);
        imwrite("/Users/admin/Desktop/opencv_temp/HSV.jpeg", HSV);
        imwrite("/Users/admin/Desktop/opencv_temp/GREY.jpeg", GREY);
    }
    
    //创建新的Mat对象 以及clone和copyTo两种拷贝方法
    void demo2(){
        Mat m1 = Mat::zeros(100, 100, CV_8UC3);//10 * 10大小、8位、unsigned char类型、三通道
        m1 = Scalar(206,45,128);//R,G,B
        imshow("m1", m1);
        cout << "hight = " << m1.rows << " width = " << m1.cols << " channels = " << m1.channels() << endl;
        //cout << m1 << endl;
        
        Mat m2;
        m1.copyTo(m2);
        m2 = Scalar(128,206,45);
        imshow("m2", m2);
        
        Mat m3;
        m3 = m1.clone();
        m3 = Scalar(45,128,206);
        imshow("m3", m3);
    }
    
    //图像像素读写(通过数组方式)
    void demo3(Mat &img){
        int h = img.rows;
        int w = img.cols;
        int c = img.channels();
        for (int row = 0 ; row < h; row++){
            for (int col = 0 ; col < w ; col++){
                if (c == 1){//灰度
                    int pv = img.at<uchar>(row,col);
                    img.at<uchar>(row,col) = 255 - pv;
                }
                else if (c == 3){//彩色
                    Vec3b bgr = img.at<Vec3b>(row,col);
                    img.at<Vec3b>(row,col)[0] = 255 - bgr[0];
                    img.at<Vec3b>(row,col)[1] = 255 - bgr[1];
                    img.at<Vec3b>(row,col)[2] = 255 - bgr[2];
                }
            }
        }
        imshow("after...", img);
    }
    
    //图像像素读写(通过指针方式)
    void demo4(Mat &img){
        int h = img.rows;
        int w = img.cols;
        int c = img.channels();
        for (int row = 0 ; row < h; row++){
            uchar* current = img.ptr<uchar>(row);
            for (int col = 0 ; col < w ; col++){
                if (c == 1){
                    *current++ = 255 - (*current);
                }
                else if (c == 3){
                    *current++ = 255 - (*current);
                    *current++ = 255 - (*current);
                    *current++ = 255 - (*current);
                }
            }
        }
        imshow("after...", img);
    }
    
    //图像像素算术操作,加减乘除
    void demo5(Mat &img){
        Mat dst;
        Mat temp = Mat::zeros(img.size(), img.type());
        temp = Scalar(2,2,2);
        add(img, temp, dst);
        imshow("add", dst);
        
        subtract(img, temp, dst);
        imshow("subtract", dst);
        
        multiply(img, temp, dst);
        imshow("multiply", dst);
        
        divide(img, temp, dst);
        imshow("divide", dst);
    }
    
    //demo6的回调函数
    static void on_lightness(int parameter , void* data){
        
        Mat img = *((Mat*)data);
        Mat dst = Mat::zeros(img.size(), img.type());
        Mat m = Mat::zeros(img.size(), img.type());
        m = Scalar(0,0,0);
        addWeighted(img, 1.0, m, 0, parameter, dst);
        imshow("亮度和对比度调整", dst);
    }
    
    static void on_contrast(int parameter , void* data){
        
        Mat img = *((Mat*)data);
        Mat dst = Mat::zeros(img.size(), img.type());
        Mat m = Mat::zeros(img.size(), img.type());
        double contrast = parameter / 100.0;
        addWeighted(img, contrast, m, 0.0, 0, dst);
        imshow("亮度和对比度调整", dst);
    }
    //滚动条操作改变图像亮度和对比度
    void demo6(Mat &img){
        namedWindow("亮度和对比度调整",WINDOW_AUTOSIZE);
        
        int max_value = 100;
        int lightness = 50;
        int contrast_value = 100;
        
        createTrackbar("lightnessBar", "亮度和对比度调整", &lightness, max_value,on_lightness,(void*)(&img));
        createTrackbar("contrastBar", "亮度和对比度调整", &contrast_value, 200,on_contrast,(void*)(&img));
        on_lightness(50, &img);
    }
    
    //键盘响应 1转换为HSV , 2转换为灰度 ， 3将RGB分别增加50 , 但2后再3会出错 , 因为灰度只有一个通道。
    void demo7(Mat &img){
        Mat m = Mat::zeros(img.size(), img.type());
        while (1){
            int c = waitKey(100);
            switch (c) {
                case 27:
                    return;
                case 49:
                    cvtColor(img, m, COLOR_BGR2HSV);
                    break;
                case 50:
                    cvtColor(img, m, COLOR_BGR2GRAY);
                    break;
                case 51:
                    cout << "channels = " << m.channels() << endl;
                    if (m.channels() == 3){
                        m = Scalar(50,50,50);
                        add(img,m,m);
                    }
                     
                    else if (m.channels() == 1){
                        break;
                    }
                    
                    break;
                default:
                    break;
            }
            imshow("After...", m);
        }
    }
    
    //opencv自带颜色表操作
    void demo8(Mat &img){
        int COLORS[] = {
            COLORMAP_HOT,
            COLORMAP_HSV,
            COLORMAP_JET,
            COLORMAP_BONE,
            COLORMAP_COOL,
            COLORMAP_PINK,
            COLORMAP_MAGMA,
            COLORMAP_OCEAN,
            COLORMAP_TURBO,
            COLORMAP_AUTUMN,
            COLORMAP_PARULA,
            COLORMAP_PLASMA,
            COLORMAP_SPRING,
            COLORMAP_SUMMER,
            COLORMAP_WINTER,
            COLORMAP_CIVIDIS,
            COLORMAP_INFERNO,
            COLORMAP_RAINBOW,
            COLORMAP_VIRIDIS,
            COLORMAP_TWILIGHT,
            COLORMAP_DEEPGREEN,
            COLORMAP_TWILIGHT_SHIFTED
        };
        
        Mat dst;
        int index = 0;
        while(1){
            int c = waitKey(100);
            if (c == 27) break;
            applyColorMap(img,dst,COLORS[index % 22]);
            index++;
            imshow("Changing...", dst);
        }
    }
    
    //图像像素逻辑操作
    void demo9(){
        
        Mat m1 = Mat::zeros(Size(256,256),CV_8UC3);
        Mat m2 = Mat::zeros(Size(256,256),CV_8UC3);
        //第二个参数Rect的前两个参数为起始点的XY坐标,三四个参数为长和宽
        //第四个参数小于0为填充,大于0为绘制
        rectangle(m1, Rect(100,100,80,80), Scalar(255,255,0), 1 , LINE_8 , 0);
        rectangle(m2, Rect(150,150,80,80), Scalar(0,255,255), 1 , LINE_8 , 0);
        imshow("m1", m1);
        imshow("m2", m2);
        
        Mat dst;
        bitwise_or(m1, m2, dst);
        imshow("bitwisw_or", dst);
        
        bitwise_and(m1, m2, dst);
        imshow("bitwise_and", dst);
        
        bitwise_not(m1,dst);
        imshow("bitwise_not", dst);
        
        bitwise_xor(m1, m2, dst);
        imshow("bitwise_xor", dst);
    }
    
    //通道分离,合并与混合
    void demo10(Mat &img){
        //分离
        vector<Mat> mv;
        split(img, mv);
        imshow("B",mv[0]);
        imshow("G",mv[1]);
        imshow("R",mv[2]);
        
        //将B和G设置为0,合并
        Mat dst;
        mv[0] = 0;
        mv[1] = 0;
        merge(mv,dst);
        imshow("after...", dst);
        
        //混合
        dst = Mat::zeros(img.size(), img.type());
        
        //参数5，第0个通道到第1个通道 第1个通道到第2个通道 第2个通道到第0个通过
        int from_to[] ={0,1,1,2,2,0};
        
        //参数
        //1:输入矩阵，可以为一个也可以为多个，但是矩阵必须有相同的大小和深度.
        //2:输入矩阵的个数。
        //3:输出矩阵，可以为一个也可以为多个，但是所有的矩阵必须事先分配空间（如用create），大小和深度须与输入矩阵等同.
        //4:输出矩阵的个数。
        //5:设置输入矩阵的通道对应输出矩阵的通道
        //6:即参数fromTo中的有几组输入输出通道关系，其实就是参数fromTo的数组元素个数除以2.
        mixChannels(&img , 1 , &dst , 1 , from_to, 3);
        imshow("afterMix...", dst);
        
    }
    
    
    //图像色彩空间转换
    void demo11(Mat &img){
        Mat hsv,mask;
        cvtColor(img, hsv, COLOR_BGR2HSV);
        imshow("HSV",hsv);
        //绿色HSV最低分别为:35,43,46,最高为:77,255,255
        inRange(hsv, Scalar(35,43,46), Scalar(77,255,255), mask);
        
        Mat back = Mat::zeros(img.size(), img.type());
        back = Scalar(40,40,255);
        bitwise_not(mask, mask);
        
        imshow("mask",mask);
        
        img.copyTo(back,mask);
        imshow("back",back);
        
    }
    
    //图像像素值统计,获取像素最大/小值,平均值,方差等...
    void demo12(Mat &img){
        double minV,maxV;
        Point minLoc,maxLoc;
        
        //转换为单通道
        vector<Mat> mv;
        split(img, mv);
        
        for (int i = 0 ; i < mv.size() ; i++){
//            在一个数组中找到全局最小值和全局最大值
//            参数解释
//            参数1：InputArray类型的src，输入单通道数组（图像）。
//            参数2：double*类型的minVal，返回最小值的指针。若无须返回，此值置为NULL。
//            参数3：double*类型的maxVal，返回最大值的指针。若无须返回，此值置为NULL。
//            参数4：Point*类型的minLoc，返回最小位置的指针（二维情况下）。若无须返回，此值置为NULL。
//            参数5：Point*类型的maxLoc，返回最大位置的指针（二维情况下）。若无须返回，此值置为NULL。
//            参数6：InputArray类型的mask，用于选择子阵列的可选掩膜。

            minMaxLoc(mv[i], &minV, &maxV, &minLoc, &maxLoc,Mat());
            cout << "No." << i << ": min value = " << minV << " max value = " << maxV << " min Loc = " << minLoc << " max Loc = " << maxLoc << endl;
        }
        
        Mat mean , stddev;
        Mat back = Mat::zeros(img.size(), img.type());
        back = Scalar(40,40,255);
        
//        计算矩阵的均值和标准偏差。
//        参数1：输入矩阵，这个矩阵应该是1-4通道的，这可以将计算结果存在Scalar_ ‘s中
//        参数2：输出参数，计算均值
//        参数3：输出参数，计算标准差
//        参数4：可选参数
        meanStdDev(back, mean, stddev);
        imshow("back", back);
        
        cout << "means = " << mean << endl;
        cout << "stddev = " << stddev << endl;
        
    }
    
    //图像几何形状绘制
    void demo13(Mat &img){
        Rect rect;
        rect.x = 100;
        rect.y = 100;
        rect.width = 200;
        rect.height = 200;
        
        Mat bg;
        bg = Mat::zeros(img.size(), img.type());
        //矩形
        rectangle(bg, rect, Scalar(255,0,0),-1,8,0);
        //圆形
        circle(bg, Point(200,200), 15, Scalar(0,255,0),-1,8,0);
        //直线
        line(bg, Point(100,100), Point(300,300), Scalar(0,0,255),2,LINE_AA,0);
        //椭圆形
        RotatedRect rtr;
        rtr.center = Point(200,200);
        rtr.angle = 90;
        rtr.size = Size(100,100);
        ellipse(bg, rtr, Scalar(255,255,0), 2, 8);
        
        Mat dst;
        addWeighted(img, 0.7, bg, 0.3, 0.0, dst);
        
        imshow("bg", bg);
        imshow("dst", dst);
    }
    
    //随机数与随机颜色
    void demo14(){
        Mat canvas = Mat::zeros(Size(512,512), CV_8UC3);
        RNG rng(123456);
        
        int w = canvas.cols;
        int h = canvas.rows;
        
        while (1){
            int c = waitKey(100);
            if (c == 27) break;
            
            
            int x1 = rng.uniform(0, w);
            int x2 = rng.uniform(0, w);
            int y1 = rng.uniform(0, h);
            int y2 = rng.uniform(0, h);
            
            int b = rng.uniform(0, 255);
            int g = rng.uniform(0, 255);
            int r = rng.uniform(0, 255);
            
            
            line(canvas,Point(x1,y1),Point(x2,y2),Scalar(b,g,r),2,LINE_AA,0);
            imshow("canvas", canvas);
        }
        
    }
    
    //多边形绘制与填充
    void demo15(){
        Mat canvas = Mat::zeros(Size(512,512), CV_8UC3);
        Point p1(200,300);
        Point p2(100,390);
        Point p3(280,320);
        Point p4(400,300);
        Point p5(340,90);
        
        vector<Point> vp;
        vp.push_back(p1);
        vp.push_back(p2);
        vp.push_back(p3);
        vp.push_back(p4);
        vp.push_back(p5);
        
//        //填充方法1: polylines(绘制) + fillPoly(填充)
//
//        //polylines
//        //1、多边形将被画到img上
//        //2、多边形的顶点集为vp
//        //3、多边形的颜色定义为Scarlar(255,255,255)
//        //4、线段渲染方式
//        //5、相对位移
//        fillPoly(canvas, vp, Scalar(255,0,255), 8, 0);
//        polylines(canvas, vp, true, Scalar(0,0,255), 2, LINE_AA,0);
//        imshow("method 1", canvas);
        
        //填充方法2: drawContours 支持多个多边形的绘制
        vector<vector<Point>> contours;
        //参数3 绘制个数,-1为全部
        //参数5 填充或绘制
        contours.push_back(vp);
        drawContours(canvas, contours, -1, Scalar(255,0,255),-1);
        imshow("method 2", canvas);
    }
    
    //鼠标操作与响应
    //参数1 鼠标事件
    static void on_draw(int event , int x , int y , int flags , void* userdata){
        Mat img = *((Mat*)userdata);
        //鼠标左键按下
        if (event == EVENT_LBUTTONDOWN){
            sp.x = x;
            sp.y = y;
            cout << "start: (" << sp.x << "," << sp.y << ")" << endl;
        }
        //鼠标右键按下
        else if (event == EVENT_LBUTTONUP){
            ep.x = x;
            ep.y = y;
            
            int dx = ep.x - sp.x;
            int dy = ep.y - sp.y;
            if (dx > 0 && dy > 0){
                Rect box(sp.x,sp.y,dx,dy);
                
                //去除ROI外边的矩形框
                temp.copyTo(img);
                
                imshow("ROI",img(box));
                rectangle(img, box, Scalar(0,0,255), 2, 8, 0);
                imshow("鼠标绘制", img);
                
                //一次绘制结束后重新设置起始位置
                sp.x = -1;
                sp.y = -1;
            }
            
        }
        //鼠标移动时
        else if (event == EVENT_MOUSEMOVE){
            //判断是否按下了左键
            if (sp.x > 0 && sp.y > 0){
                ep.x = x;
                ep.y = y;
                
                int dx = ep.x - sp.x;
                int dy = ep.y - sp.y;
                if (dx > 0 && dy > 0){
                    temp.copyTo(img);
                    Rect box(sp.x,sp.y,dx,dy);
                    rectangle(img, box, Scalar(0,0,255), 2, 8, 0);
                    imshow("鼠标绘制", img);
                }
            }
            
        }
    }
    
    void demo16(Mat &img){
        namedWindow("鼠标绘制",WINDOW_AUTOSIZE);
        setMouseCallback("鼠标绘制", on_draw, (void*)(&img));
        imshow("鼠标绘制", img);
        //保存原图
        temp = img.clone();
    }
    
    //图像像素类型转换、归一化
    void demo17(Mat &img){
        Mat dst;
        cout << img.type() << endl; // 16 CV_8UC3
        
        // CV_32F float CV_32S int
        img.convertTo(img, CV_32F);
        cout << img.type() << endl; // 21 CV_32F
        
        normalize(img, dst, 1.0, 0, NORM_MINMAX);
        cout << dst.type() << endl; // 21 CV_32F
        
        imshow("dst",dst);
    }
    
    //图像放缩与插值
    static void m_mouse(int event , int x , int y , int flags , void* userdata){
        Mat img = *((Mat*)userdata);
        Mat dst;
        int w = img.cols;
        int h = img.rows;

        //mac EVENT_MOUSEHWHEEL无法使用,所以改成 EVENT_LBUTTONUP 和 EVENT_RBUTTONDOWN , 不知道是鼠标问题还是系统问题
        if (event == EVENT_LBUTTONUP){
            
            coefficient += 0.2;
        }
        else if (event == EVENT_RBUTTONDOWN){
            if (coefficient <= 0) coefficient += 0.2;
            coefficient -= 0.2;
        }
        
        resize(img, dst, Size(w * coefficient,h * coefficient), 0, 0 ,INTER_LINEAR);
        imshow("dts",dst);
    }
    
    void demo18(Mat &img){
//        Mat zoomout , zoomin;
//        int h = img.rows;
//        int w = img.cols;
//
//        resize(img, zoomin, Size(w * 2,h * 2), 0, 0 ,INTER_LINEAR);
//        imshow("zoomin",zoomin);
//
//        resize(img, zoomout, Size(w / 2,h / 2), 0, 0 ,INTER_LINEAR);
//        imshow("zoomout",zoomout);
        
        namedWindow("Resize",WINDOW_AUTOSIZE);
        setMouseCallback("Resize", m_mouse, (void*)(&img));
        imshow("Resize", img);
    }
    
    //图像翻转
    void demo19(Mat &img){
        Mat dst;
        
        // 第三个参数：
        // 0: 上下翻转
        // 1: 左右翻转
        // -1:对角线翻转
        flip(img, dst, 0);
        imshow("filp", dst);
    }
    
    //图像旋转
    void demo20(Mat &img){
        Mat dst, M;
        int w = img.cols;
        int h = img.rows;
        
        //1:中心点 2:角度 3:倍数
        M = getRotationMatrix2D(Point2f(w / 2 , h / 2), 45, 1.0);
        //旋转后大小会发生变化
        double cos = abs(M.at<double>(0,0));
        double sin = abs(M.at<double>(0,1));
        int nw = cos * w + sin * h;
        int nh = sin * w + cos * h;
        M.at<double>(0,2) += (nw / 2 - w / 2);
        M.at<double>(1,2) += (nh / 2 - h / 2);
        
        warpAffine(img, dst, M, Size(nw,nh),INTER_LINEAR, 0, Scalar(255,0,0));
        
        imshow("旋转后",dst);
    }
    
    //视频读取与摄像头
    void demo21(){
        VideoCapture capture(0);
        Mat frame;
        while (1){
            capture.read(frame);
            flip(frame, frame, 1);
            if (frame.empty()) break;
            imshow("frame", frame);
            int flag = waitKey(1);
            if (flag == 27) break;
        }
    }
    
    //视频处理与保存
    
    void demo22(){
        VideoCapture capture("/Users/admin/Downloads/opencv_tutorial_data-master/images/sample.mp4");
        Mat frame;
        //视频宽高
        int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
        int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
        //总帧数
        int count = capture.get(CAP_PROP_FRAME_COUNT);
        //Frame Per Second
        double fps = capture.get(CAP_PROP_FPS);
        
        cout << "frame_width = " << frame_width << " frame_height = " << frame_height << endl;
        cout << "count = " << count << " FPS = " << fps << endl;
        
        //保存 参数2:哪一种编码方式
        VideoWriter write("/Users/admin/Desktop/sample.mp4",capture.get(CAP_PROP_FOURCC),fps,Size(frame_width,frame_height));

        Mat HSV;
        
        
        while (1){
            capture.read(frame);
            flip(frame, frame, 1);
            if (frame.empty()) break;
            imshow("frame", frame);
            cvtColor(frame, HSV, COLOR_BGR2HSV);
            write.write(HSV);
            int flag = waitKey(1);
            if (flag == 27) break;
        }
        //释放
        capture.release();
        write.release();
    }
    
    //一维图像直方图
    void demo23(Mat &img){
        //分离三通道
        vector<Mat> bgr;
        split(img, bgr);
        
        //参数变量
        const int channels[1] = {0};
        const int bins[1] = {256};
        float hranges[2] = {0,255};
        const float* ranges[1] = {hranges};
        
        Mat b_hist;
        Mat g_hist;
        Mat r_hist;
        
        //计算三通道的直方图
        //第几个通道，几张图 . . 直方图输出 维数 . 取值范围
        calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
        calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
        calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
        
        //显示直方图
        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound(double(hist_w)/bins[0]);
        Mat histImage= Mat::zeros(hist_h, hist_w, CV_8UC3);
        
        //归一化直方图数据
        
        normalize(b_hist,b_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
        normalize(g_hist,g_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
        normalize(r_hist,r_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
        
        //绘制直方图曲线
        for (int i = 1 ; i < bins[0] ; i++){
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                 Point(bin_w * (i),hist_h - cvRound(b_hist.at<float>(i))),Scalar(255,0,0),2,8,0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                 Point(bin_w * (i),hist_h - cvRound(g_hist.at<float>(i))),Scalar(0,255,0),2,8,0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                 Point(bin_w * (i),hist_h - cvRound(r_hist.at<float>(i))),Scalar(0,0,255),2,8,0);
        }
        
        //显示直方图
        namedWindow("histogram",WINDOW_AUTOSIZE);
        imshow("histogram",histImage);
    }
    
    //二维直方图
    
    void demo24(Mat &img){
            Mat hsvImage;                                           //hsv图像
            cvtColor(img, hsvImage, cv::COLOR_RGB2HSV);    //将R、G、B图像转换为HSV格式
         
            int hsvImgNum = 1;                                          //图像数
            int hsvChannels[2] = { 0 , 1 };                             //需要计算的通道号 hsv的 0通道和1通道
            Mat hsvHist;                                            //hsv图像的二维直方图
            const int hsvHistDim = 2;                                   //直方图维数
            
            int hBins = 30, sBins = 30;
            const int hsvHistSize[2] = { hBins,sBins };                 //存放每个维度直方图尺寸（bin数量）的数组histSize
            
            float hRanges[2] = { 0,256 };                               //hue的值得统计范围      0-179
            float sRanges[2] = { 0,256 };                               //saturation的统计范围   0-255
            const float *hsvHistRanges[2] = { hRanges , sRanges };      //hsv统计范围的指针
            
            bool hsvUniform = true;                                     //是否均匀
            bool hsvAccumulate = false;                                 //是否累积
         
            //计算HSV图像的hst通道的二维直方图
            calcHist( &hsvImage,
                      hsvImgNum,
                      hsvChannels,
                      cv::Mat(),
                      hsvHist,
                      hsvHistDim,
                      hsvHistSize,
                      hsvHistRanges,
                      hsvUniform,
                      hsvAccumulate);
         
            double hsvMaxVal = 0;
            double hsvMinVal = 0;
            minMaxLoc(hsvHist, &hsvMinVal, &hsvMaxVal, 0, 0); //找到直方图矩阵hist中的最大值
            
            int hsvScale = 15;
         
            //创建直方图画布，画布矩阵中同行的saturation值相同，同列的hue值相同
            Mat hsvHistImage = Mat::zeros(sBins*hsvScale, hBins*hsvScale, CV_8UC3);
         
         
            //扫描直方图，填充画布
            for (int i = 0; i<sBins; i++)
            {
                for (int j = 0; j < hBins; j++)
                {
                    float binValue = hsvHist.at<float>(i, j);
                    //将直方图中的值归一化到0到255
                    int intensity = cvRound(binValue * 255 / hsvMaxVal);
                    //画矩形柱状图,Point的坐标中x值对应着hue维度，y值对应值saturation维度，这与画布矩阵必须一致
                    rectangle( hsvHistImage,
                           Point(i*hsvScale, j*hsvScale),
                           Point((i + 1)*hsvScale - 1, (j + 1)*hsvScale - 1),
                           Scalar::all(intensity),
                           FILLED);
                }
            }
         
            namedWindow("H-S Histogram");
            imshow("H-S Histogram", hsvHistImage);
    }
    
    //直方图均衡化
    void demo25(Mat &img){
        Mat grey;
        cvtColor(img, grey, COLOR_BGR2GRAY);
        imshow("grey",grey);
        
        Mat dst;
        equalizeHist(grey, dst);
        imshow("equalizeHist",dst);
    }
    
    //图像卷积
    void demo26(Mat &img){
        Mat dst;
        //Point(-1,-1)代表卷积核中心位置
        blur(img, dst, Size(30,30),Point(-1,-1));
        imshow("blur",dst);
    }
    
    //高斯模糊
    void demo27(Mat &img){
        Mat dst;
        //size()卷积核的大小，一定是整的奇数。最后两个参数越大 模糊越厉害 一般是改动最后一个参数的大小
        GaussianBlur(img, dst, Size(0,0), 15);
        imshow("GaussianBlur",dst);
    }
    
    //高斯双边模糊
    void demo28(Mat &img){
        Mat dst;
        //size()卷积核的大小，一定是整的奇数。最后两个参数越大 模糊越厉害 一般是改动最后一个参数的大小
        bilateralFilter(img, dst, 0, 100, 10);
        imshow("bilateralFilter",dst);
    }
};


void verificationCode(){
    Mat bg = imread("/Users/admin/Desktop/bg.png");
    Mat dst = Mat::zeros(bg.size(), bg.type());
    
    cvtColor(bg, dst, COLOR_BGR2GRAY);
    GaussianBlur(dst, dst, Size(3,7), 0);
    imshow("GaussianBlur",dst);
    Canny(dst, dst, 180, 300);
    
    imshow("Canny",dst);
    
    waitKey(0);
}

void test(){
    Mat img = imread("/Users/admin/Desktop/test.jpeg");
    //Mat img = imread("/Users/admin/Desktop/1.JPG");
    
    if (img.empty()){
        cout << "Could not open the iamge...";
        return;
    }
    else {
        imshow("Test", img);
        Demo D;
        //D.demo1(img);
        //D.demo2();
        //D.demo3(img);
        //D.demo4(img);
        //D.demo5(img);
        //D.demo6(img);
        //D.demo7(img);
        //D.demo8(img);
        //D.demo9();
        //D.demo10(img);
        //D.demo11(img);
        //D.demo12(img);
        //D.demo13(img);
        //D.demo14();
        //D.demo15();
        //D.demo16(img);
        //D.demo17(img);
        //D.demo18(img);
        //D.demo19(img);
        //D.demo20(img);
        //D.demo21();
        //D.demo22();
        //D.demo23(img);
        //D.demo24(img);
        //D.demo25(img);
        //D.demo26(img);
        //D.demo27(img);
        //D.demo28(img);
        waitKey(0);
        return;
    }
}

int main(int argc, const char * argv[]) {
    // insert code here...
    test();
    // verificationCode();
    
    
    return 0;
}
