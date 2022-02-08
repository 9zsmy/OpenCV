# -*- coding: utf-8 -*-
# 开发人员   ：NJR
# 开发时间   ：2022/2/7  19:16 
# 文件名称   ：03图像处理3.PY
# 开发工具   ：PyCharm

import matplotlib.pyplot as plt
import cv2
import numpy as np

# 频域变换
img = cv2.imread('material/test.jpeg', 0)
imgFloat32 = np.float32(img)

dft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
# 得到灰度图能表现的形式
magnitudeSpectrum = 20 * np.log(cv2.magnitude(dftShift[:,:,0], dftShift[:,:,1]))

plt.subplot(121), plt.imshow(img, 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitudeSpectrum, 'gray')
plt.title('magnitudeSpectrum'), plt.xticks([]), plt.yticks([])

plt.show()

# 低频 使图像磨合
img = cv2.imread('material/test.jpeg', 0)
imgFloat32 = np.float32(img)
dft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
rows, cols = img.shape
# 中心位置
crow, ccol = int(rows / 2), int(cols / 2)
# 高通滤波
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# IDFT
fshift = mask * dftShift
f_ifshift = np.fft.ifftshift(fshift)
imgBack = cv2.idft(f_ifshift)
imgBack = cv2.magnitude(imgBack[:,:,0], imgBack[:,:,1])

plt.subplot(121), plt.imshow(img, 'gray')
plt.title('Input Image2'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imgBack, 'gray')
plt.title('imgBack2'), plt.xticks([]), plt.yticks([])

# 高频 使图像细节增强
img = cv2.imread('material/test.jpeg', 0)
imgFloat32 = np.float32(img)
dft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
rows, cols = img.shape
# 中心位置
crow, ccol = int(rows / 2), int(cols / 2)
# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0
# IDFT
fshift = mask * dftShift
f_ifshift = np.fft.ifftshift(fshift)
imgBack = cv2.idft(f_ifshift)
imgBack = cv2.magnitude(imgBack[:,:,0], imgBack[:,:,1])

plt.subplot(121), plt.imshow(img, 'gray')
plt.title('Input Image3'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imgBack, 'gray')
plt.title('imgBack3'), plt.xticks([]), plt.yticks([])
