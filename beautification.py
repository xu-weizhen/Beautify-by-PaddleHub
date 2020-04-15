#!/usr/bin/env python
# coding: utf-8


# 参数
videos_path = 'girl.mp4'            # 原始视频路径
frames_save_path = 'frame'          # 图片帧保存路径
time_interval = 1                   # 图片帧保存间隔
target_frames_save_path = 'beauty'  # 美化后图片帧保存路径
target_videos_path = 'beauty.mp4'   # 生成的视频保存路径
fps = 13                            # 生成的视频的帧率

do_thin_face = False             # 是否瘦脸
do_enlarge_eyes = True           # 是否放大双眼
enlarge_eyes_radius = 15         # 眼睛放大范围
enlarge_eyes_strength = 15       # 眼睛放大程度
do_rouge = False                 # 是否画红唇
rouge_ruby = True                # 是否采用深色口红
do_whitening = True              # 是否美白



import cv2
import paddlehub as hub
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import math
import os
from PIL import Image




# 将视频转换为图片帧
def video2frame(videos_path,frames_save_path,time_interval):
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    try:
        while success:
            success, image = vidcap.read()
            count += 1
            
            if count % time_interval == 0:
                if count < 10:
                    num = '00' + str(count)
                elif count < 100:
                    num = '0' + str(count)
                else:
                    num = str(count)
                cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame{}.jpg".format(num))
    except Exception as e:
        print(str(e))
    print('视频已转换为图片')

video2frame(videos_path, frames_save_path, time_interval + 1)


# 获取文件名
photos_name = []
for a,b,c in os.walk(frames_save_path):
    photos_name.append(c)
photos_name = photos_name[0]
photos_name.sort()



# 读取图片
img = []
for name in photos_name:
    img.append(cv2.imread(frames_save_path + '/' + name))

module = hub.Module(name="face_landmark_localization")
result = module.keypoint_detection(images = img)




# 局部平移算法
def local_traslation_warp(image, start_point, end_point, radius):
    radius_square = math.pow(radius, 2)
    image_cp = image.copy()

    dist_se = math.pow(np.linalg.norm(end_point - start_point), 2)
    height, width, channel = image.shape
    for i in range(width):
        for j in range(height):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（start_point[0], start_point[1])的矩阵框中
            if math.fabs(i - start_point[0]) > radius and math.fabs(j - start_point[1]) > radius:
                continue

            distance = (i - start_point[0]) * (i - start_point[0]) + (j - start_point[1]) * (j - start_point[1])

            if (distance < radius_square):
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (radius_square - distance) / (radius_square - distance + dist_se)
                ratio = ratio * ratio

                # 映射原位置
                new_x = i - ratio * (end_point[0] - start_point[0])
                new_y = j - ratio * (end_point[1] - start_point[1])

                new_x = new_x if new_x >=0 else 0
                new_x = new_x if new_x < height-1 else height-2
                new_y = new_y if new_y >= 0 else 0
                new_y = new_y if new_y < width-1 else width-2

                # 根据双线性插值法得到new_x, new_y的值
                image_cp[j, i] = bilinear_insert(image, new_x, new_y)
                
    return image_cp

# 双线性插值法
def bilinear_insert(image, new_x, new_y):
    w, h, c = image.shape
    if c == 3:
        x1 = int(new_x)
        x2 = x1 + 1
        y1 = int(new_y)
        y2 = y1 + 1

        part1 = image[y1, x1].astype(np.float) * (float(x2) - new_x) * (float(y2) - new_y)
        part2 = image[y1, x2].astype(np.float) * (new_x - float(x1)) * (float(y2) - new_y)
        part3 = image[y2, x1].astype(np.float) * (float(x2) - new_x) * (new_y - float(y1))
        part4 = image[y2, x2].astype(np.float) * (new_x - float(x1)) * (new_y - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


# 瘦脸
def thin_face(image, face_landmark):
    end_point = face_landmark[30]

    # 瘦左脸，3号点到5号点的距离作为瘦脸距离
    dist_left = np.linalg.norm(face_landmark[3] - face_landmark[5])
    local_traslation_warp(image, face_landmark[3], end_point, dist_left)

    # 瘦右脸，13号点到15号点的距离作为瘦脸距离
    dist_right = np.linalg.norm(face_landmark[13] - face_landmark[15])
    image = local_traslation_warp(image, face_landmark[13], end_point, dist_right)
    return image


# 大眼
def enlarge_eyes(image, face_landmark, radius=15, strength=10):
    """
    image： 人像图片
    face_landmark: 人脸关键点
    radius: 眼睛放大范围半径
    strength：眼睛放大程度
    """
    # 以左眼最低点和最高点之间的中点为圆心
    left_eye_top = face_landmark[37]
    left_eye_bottom = face_landmark[41]
    left_eye_center = (left_eye_top + left_eye_bottom)/2
    # 以右眼最低点和最高点之间的中点为圆心
    right_eye_top = face_landmark[43]
    right_eye_bottom = face_landmark[47]
    right_eye_center = (right_eye_top + right_eye_bottom)/2

    # 放大双眼
    local_zoom_warp(image, left_eye_center, radius=radius, strength=strength)
    local_zoom_warp(image, right_eye_center, radius=radius, strength=strength)


# 图像局部缩放算法
def local_zoom_warp(image, point, radius, strength):
    height = image.shape[0]
    width = image.shape[1]      
    left =int(point[0] - radius) if point[0] - radius >= 0 else 0
    top = int(point[1] - radius) if point[1] - radius >= 0 else 0
    right = int(point[0] + radius) if point[0] + radius < width else width-1
    bottom = int(point[1] + radius) if point[1] + radius < height  else height-1

    radius_square = math.pow(radius, 2)
    for y in range(top, bottom):
        offset_y = y - point[1]
        for x in range(left, right):
            offset_x = x - point[0]
            dist_xy = offset_x * offset_x + offset_y * offset_y

            if dist_xy <= radius_square:
                scale = 1 - dist_xy / radius_square
                scale = 1 - strength / 100 * scale
                new_x = offset_x * scale + point[0]
                new_y = offset_y * scale + point[1]
                new_x = new_x if new_x >=0 else 0
                new_x = new_x if new_x < height-1 else height-2
                new_y = new_y if new_y >= 0 else 0
                new_y = new_y if new_y < width-1 else width-2

                image[y, x] = bilinear_insert(image, new_x, new_y)


# 涂口红
def rouge(image, face_landmark, ruby=True):
    """
    image： 人像图片
    face_landmark: 人脸关键点
    ruby：是否需要深色口红
    """
    image_cp = image.copy()

    if ruby:
        rouge_color = (0,0,255)
    else:
    	rouge_color = (0,0,200)

    points=face_landmark[48:68]
    
    hull = cv2.convexHull(points)
    cv2.drawContours(image, [hull], -1, rouge_color, -1)
    cv2.addWeighted(image, 0.2, image_cp, 1-0.1, 0, image_cp)
    return image_cp


# 美白
def whitening(img, face_landmark):
	# 简单估计额头所在区域
	# 根据0号、16号点画出额头(以0号、16号点所在线段为直径的半圆)
	radius=(np.linalg.norm(face_landmark[0] - face_landmark[16]) / 2).astype('int32')
	center_abs=tuple(((face_landmark[0] + face_landmark[16]) / 2).astype('int32'))
	angle=np.degrees(np.arctan((lambda l:l[1]/l[0])(face_landmark[16]-face_landmark[0]))).astype('int32')
	face = np.zeros_like(img)
	cv2.ellipse(face,center_abs,(radius,radius),angle,180,360,(255,255,255),2)

	points=face_landmark[0:17]
	hull = cv2.convexHull(points)
	cv2.polylines(face, [hull], True, (255,255,255), 2)

	index = face >0
	face[index] = img[index]
	dst = np.zeros_like(face)
	# v1:磨皮程度
	v1 = 3
	# v2: 细节程度
	v2 = 2

	tmp1 = cv2.bilateralFilter(face, v1 * 5, v1 * 12.5, v1 * 12.5)
	tmp1 = cv2.subtract(tmp1,face)
	tmp1 = cv2.add(tmp1,(10,10,10,128))
	tmp1 = cv2.GaussianBlur(tmp1,(2*v2 - 1,2*v2-1),0)
	tmp1 = cv2.add(img,tmp1)
	dst = cv2.addWeighted(img, 0.1, tmp1, 0.9, 0.0)
	dst = cv2.add(dst,(10, 10, 10,255))

	index = dst>0
	img[index] = dst[index]

	return img


# 瘦脸
if do_thin_face:
    for i in range(len(img)):
        img[i] = thin_face(img[i], np.array(result[i]['data'][0], dtype='int'))

# 放大双眼
if do_enlarge_eyes:
    for i in range(len(img)):
        enlarge_eyes(img[i], np.array(result[i]['data'][0], dtype='int'), enlarge_eyes_radius, enlarge_eyes_strength)

# 画红唇
if do_rouge:
    for i in range(len(img)):
        rouge(img[i], np.array(result[i]['data'][0], dtype='int'), rouge_ruby)

# 美白
if do_whitening:
    for i in range(len(img)):
        whitening(img[i], np.array(result[i]['data'][0], dtype='int'))

# 保存图片
for i in range(len(img)):
    if i < 10:
        count = '00' + str(i)
    elif i < 100:
        count = '0' + str(i)
    else:
        count = str(i)

    name = target_frames_save_path + '/' + count + '.jpg'
    cv2.imwrite(name, img[i])

print('图片帧美化完成')



# 图片帧转视频
def frame2video(im_dir,video_dir,fps):
    im_list = os.listdir(im_dir)
    im_list.sort()
    img = Image.open(im_dir + '/' + im_list[0])
    img_size = img.size

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in im_list:
        im_name = im_dir + '/' + i
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
    videoWriter.release()
    print('图片帧已合成为视频')

frame2video(target_frames_save_path, target_videos_path, fps)



# 美化前后效果展示
img1 = mpimg.imread('frame/frame002.jpg') 
img2 = mpimg.imread('beauty/000.jpg') 
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(img1)
plt.axis('off') 
plt.subplot(1,2,2)
plt.imshow(img2)
plt.axis('off') 
plt.show()


# 上图演示了对视频进行美白和放大眼睛之后的效果。
# 通过调整参数还可以对人物进行瘦脸和画红唇，感兴趣的朋友可以自己试试。

