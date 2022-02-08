# -*- coding: utf-8 -*-
# 开发人员   ：NJR
# 开发时间   ：2022/2/7  16:56 
# 文件名称   ：03图像处理2.PY
# 开发工具   ：PyCharm

import matplotlib.pyplot as plt
import cv2
import numpy as np

# 模版匹配
img = cv2.imread("material/sb-2.jpg", 0)
template = cv2.imread("material/sb-1.jpg", 0)
cv2.imshow('img', img)
cv2.imshow('ROI', template)

h, w = template.shape[:2]

# 六种方法，建议使用归一化后的，即normed
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
res = cv2.matchTemplate(img, template, 1)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

i = 0
for meth in methods:

    img2 = img.copy()
    method = eval(meth)
    res = cv2.matchTemplate(img, template, method)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + w, topLeft[1] + h)

    cv2.rectangle(img2, topLeft, bottomRight, 255, 2)
    cv2.imshow('img2_%d' % i, img2)

    i += 1


# 直方图
# 参数1：img， 格式为uint8或float32，传入函数时应用中括号，如[image]
# 参数2：channels 灰度[0] 彩色[0][1][2]分别代表bgr
# 参数3：mask 掩膜图像 统计整幅图为None，统计某一部分就制作掩膜使用
# 参数4：histSize bin的数目，也需要用中括号
# 参数5：像素值范围[0-256]
img = cv2.imread('material/test.jpeg', 0)  # 0代表灰度
hist = cv2.calcHist([img], [0], None, [256], [0,256])

plt.hist(img.ravel(), 256)
plt.show()

# mask操作
mask = np.zeros(img.shape[:2], np.uint8)
mask[1000:2000, 1000:2000] = 255
cv2.imshow("mask", mask)

maskedImg = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow('maskedImg', maskedImg)

# 均衡化
img = cv2.imread('material/test.jpeg', 0)
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(), 256)
plt.show()

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
resClahe = clahe.apply(img)

res = np.hstack([img, equ, resClahe])
cv2.imshow("res", res)


cv2.waitKey(0)
cv2.destroyAllWindows()

