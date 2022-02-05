# -*- coding: utf-8 -*-
# 开发人员   ：NJR
# 开发时间   ：2022/2/5  16:01 
# 文件名称   ：图像基本操作.PY
# 开发工具   ：PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像与显示
img1 = cv2.imread('/Users/admin/Downloads/python/opencv/material/test.jpeg')
if img1 is not None:
    cv2.imshow("test1", img1)

    # 获取长 宽 通道数
    # Height Width Channels
    HWC = img1.shape

# 以灰度方式读取
img2 = cv2.imread('/Users/admin/Downloads/python/opencv/material/test.jpeg', cv2.IMREAD_GRAYSCALE)
if img2 is not None:
    cv2.imshow("test2", img2)

    # 保存
    cv2.imwrite("Gray.jpeg", img2)

    # 类型
    Type = type(img2)

    # 像素点个数
    count = img2.size

    # 数据类型
    Dtype = img2.dtype

# 读取视频或访问摄像头
vc = cv2.VideoCapture("/Users/admin/Downloads/python/opencv/material/sample.mp4")
if vc.isOpened():
    # 参数1:是否已经打开
    # 参数2:当前帧的图像
    isOpen, frame = vc.read()
else:
    isOpen = False

while isOpen:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("video", gray)
        if cv2.waitKey(100) & 0xFF == 27:
            break

vc.release()
cv2.destroyAllWindows()

# 裁剪图像
if img1 is not None:
    ROI = img1[0:100, 0:200]
    cv2.imshow("ROI", ROI)

# 颜色通道提取
b, g, r = cv2.split(img1)
# 只保留b, g, r中的一种
cur_img = img1.copy()
cur_img[:, :, 0] = 0  # b设置为0
cur_img[:, :, 1] = 0  # g设置为0
cv2.imshow("R", cur_img)
# 合并
cv2.merge((b, g, r))

# 边界填充
top, bottom, left, right = (100, 100, 100, 100)
# 复制最边缘像素
replicate = cv2.copyMakeBorder(img1, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
# 对感兴趣图像中的像素在两边进行复制,如:gfedcba abcdefgh hgfedcb
reflect = cv2.copyMakeBorder(img1, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)
# 以最边缘像素为轴,如:gfedcb abcdefgh gfedcba
reflect101 = cv2.copyMakeBorder(img1, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
# 外包装,如:cdefgh abcdefgh abcdefg
wrap = cv2.copyMakeBorder(img1, top, bottom, left, right, borderType=cv2.BORDER_WRAP)
# 常量数值填充
constant = cv2.copyMakeBorder(img1, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('Original')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('reflect101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('wrap')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant')

plt.show()

# 数值计算 直接+ - * / 或
# cv2.add()
# cv2.subtract()
# cv2.multiply()
# cv2.divide()

# 图像融合
img3 = cv2.imread('/Users/admin/Downloads/python/opencv/material/1.JPG')
cv2.imshow("img3",img3)
w, h, c = img3.shape
img1 = cv2.resize(img1,(h, w))

# img1 * 0.4 + img3 * 0.6 + 0
res = cv2.addWeighted(img1, 0.4, img3, 0.6, 0)
cv2.imshow("res", res)

cv2.waitKey(0)
