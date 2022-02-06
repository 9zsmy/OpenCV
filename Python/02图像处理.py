# -*- coding: utf-8 -*-
# 开发人员   ：NJR
# 开发时间   ：2022/2/5  20:31 
# 文件名称   ：02图像处理.PY
# 开发工具   ：PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('material/test.jpeg', cv2.IMREAD_GRAYSCALE)
if img1 is not None:
    cv2.imshow("img1", img1)

# 图像阈值
# 参数1：src 输入图(单通道)
# 参数2：dst 输出图
# 参数3：thresh 阈值
# 参数4：maxval 当像素值超过或小于阈值时所赋予的值(根据参数4确定)
# 参数5：type 二值化操作的类型，包括5种
ret, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)  # 超过阈值部分取maxval, 否则取0
ret, thresh2 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)  # THRESH_BINARY的反转
ret, thresh3 = cv2.threshold(img1, 127, 255, cv2.THRESH_TRUNC)  # 大于阈值部分设置为阈值， 否则不变
ret, thresh4 = cv2.threshold(img1, 127, 255, cv2.THRESH_TOZERO)  # 大于阈值部分不改变， 否则设为0
ret, thresh5 = cv2.threshold(img1, 127, 255, cv2.THRESH_TOZERO_INV)  # THRESH_TOZERO的反转

title = ['Original', 'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
images = [img1, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 图像平滑
# 均值滤波，简单的平均卷积操作
blur = cv2.blur(img1, (3,3))

# 方框滤波, 如果归一化为True， 则与均值滤波完全一样
box = cv2.boxFilter(img1, -1, (3,3), normalize=True)

# 方框滤波归一化为False， 容易越界， 一旦越界全部取255
box = cv2.boxFilter(img1, -1, (3,3), normalize=False)

# 高斯滤波
gaussianblue = cv2.GaussianBlur(img1, (5, 5), 1)

# 中值滤波
medium = cv2.medianBlur(img1, 5)

# hstack水平，vstack垂直，将[]中的图像合并在一起
res = np.hstack([blur, gaussianblue, medium])
cv2.imshow('medium vs average', res)

# 腐蚀操作
kernel = np.ones((30, 30), np.uint8)
erosion1 = cv2.erode(img1, kernel, iterations=1)
erosion2 = cv2.erode(img1, kernel, iterations=2)
erosion3 = cv2.erode(img1, kernel, iterations=3)
res = np.hstack([erosion1, erosion2, erosion3])
cv2.imshow('erosion', res)

# 膨胀操作
kernel = np.ones((30, 30), np.uint8)
dilate1 = cv2.dilate(img1, kernel, iterations=1)
dilate2 = cv2.dilate(img1, kernel, iterations=2)
dilate3 = cv2.dilate(img1, kernel, iterations=3)
res = np.hstack([dilate1, dilate2, dilate3])
cv2.imshow('dilate', res)

# 开运算与闭运算
# 开：先腐蚀再膨胀
img = img1
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
# 闭：先膨胀再腐蚀
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)

# 梯度运算 梯度 = 膨胀 - 腐蚀
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(img, kernel, iterations=5)
erode = cv2.erode(img, kernel, iterations=5)
res = np.hstack([dilate, erode])
cv2.imshow("gradient 1", res)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient 2', gradient)

# 礼帽与黑帽
# 礼帽 = 原始输入 - 开运算
# 黑帽 = 闭运算 - 原始输入
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("tophat", tophat)

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("blackhat", blackhat )

# sobel算子
# 参数1：输入图像
# 参数2：ddepth 图像深度，一般是-1，与原图一致
# 参数3：dx 水平方向
# 参数4：dy 竖直方向
# 参数5：ksize Sobel算子的大小

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
# 白到黑是整数，黑到白是负数，会被截断成0，所以要取绝对值
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
sobely = cv2.convertScaleAbs(sobely)

# 先计算x和y，再求和
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

# 不建议直接同时计算, f分开计算效果更好
# sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize = 3)
# sobelxy = cv2.convertScaleAbs(sobelxy)
# cv2.imshow("sobelxy_2", sobelxy )

# scharr算子,更为敏感，描绘相对于sobel更细致
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

# laplacian算子，一般与其他方法一起使用
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack([sobelxy, scharrxy, laplacian])
cv2.imshow("res1", res)

# canny边缘检测
# 参数1：img
# 参数2：minvalue 小于该参数舍弃
# 参数3：maxvalue 大于该参数舍弃
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)
res = np.hstack([v1, v2])
cv2.imshow("v1 and v2", res)

# 图像金字塔
img = cv2.imread("material/test.jpeg")
up = cv2.pyrUp(img)
print(up.shape)
down = cv2.pyrDown(up)
print(down.shape)
cv2.imshow('up',up)
cv2.imshow('down',down)

# 金字塔制作方式
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
l_1 = img - cv2.resize(down_up, (img.shape[1], img.shape[0]))
cv2.imshow("l_1", l_1)

# 图像轮廓
# 参数1：img
# 参数2：mode 图像检索模式
# cv2.RETR_TREE 检测所有的轮廓，并重构嵌套轮廓层次
# cv2.RETR_CCOMP 检测所有轮廓，并将它们组织成两层，顶层是各部分外部边界，第二层是空洞的边界
# cv2.RETR_LIST 检测所有轮廓，并保存到一个链表中
# cv2.RETR_EXTERNAL 只检测最外边轮廓
# 参数3：method 轮廓逼近方法
# cv2.CHAIN_APPROX_NONE 以freeman链码的形式输出轮廓，所有其他方法输出多边形(顶点的序列)
# cv2.CHAIN_APPROX_SIMPLE 压缩水平的，垂直的和斜的部分，也就是只保留终点部分

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
# cv2.findContours的opencv旧版本返回3个值，新版本返回两个值
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 会改变原图像，所以要先复制
drawImg = img.copy()
# 绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
res = cv2.drawContours(drawImg, contours, -1, (0, 0, 255), 2)
cv2.imshow("res", res)

# 轮廓特征
cnt = contours[-1]
# 面积
print(cv2.contourArea(cnt))
# 周长， True表示闭合
cv2.arcLength(cnt, True)

# 轮廓近似
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
draw_img = img.copy()
res_ = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv2.imshow("res_", res_)

img_ = img.copy()
# 外接矩形
x, y, w, h = cv2.boundingRect(cnt)
img_ = cv2.rectangle(img_, (x, y), (w + x, h + y), (0, 255, 0), 2)
cv2.imshow("rectangle", img_)
# 外接圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
img_ = cv2.circle(img_, center, radius, (255, 0, 0), 2)
cv2.imshow('circle', img_)

cv2.waitKey(0)
cv2.destroyAllWindows()
