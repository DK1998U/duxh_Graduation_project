import os
import sys
import cv2
from PIL import Image
import numpy as np
from PIL import ImageEnhance
import random
from PIL import ImageChops
import time
def sp_noise(image,prob=0.1):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
def test_radom():
    return np.random.randint(0,32)
    
def d_img_enhance(img,width = 224,height = 224,angle1 = 10,angle2 = -10,angle3 = 5,angle4 = -5,xoff = 15,yoff = 15):
    '''
    img  OPENCV打开的图像
    '''
    cvimage = cv2.resize(img, (width, height))
    image = Image.fromarray(cv2.cvtColor(cvimage,cv2.COLOR_BGR2RGB))
    random_choose = np.random.randint(0,25)
    if random_choose ==0:
        return image.rotate(angle1, Image.BICUBIC)#旋转angle角度1
    elif random_choose ==1:
        return image.rotate(angle2, Image.BICUBIC)#旋转angle角度2
    elif random_choose ==2:
        return image.rotate(angle3, Image.BICUBIC)#旋转angle角度3
    elif random_choose ==3:
        return image.rotate(angle4, Image.BICUBIC)#旋转angle角度4
    elif random_choose ==4:
        return ImageChops.offset(image,xoff,0)#水平平移 off个像素
    elif random_choose ==5:
        return ImageChops.offset(image,0,yoff)#水平平移 off个像素
    elif random_choose ==6:
        return image.transpose(Image.FLIP_LEFT_RIGHT)#左右反转
    elif random_choose ==7:
        return image.transpose(Image.FLIP_TOP_BOTTOM)#上下反转
    elif random_choose ==8:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Color(image).enhance(random_factor)    # 调整图像的饱和度_____T
    elif random_choose ==9:
        random_factor = np.random.randint(low=10, high=21) / 10.0
        return ImageEnhance.Brightness(image).enhance(random_factor)   # 调整图像的亮度_____T
    elif random_choose ==10:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=10, high=21) / 10.0
        return ImageEnhance.Brightness(color_image).enhance(random_factor)   # 调整图像的饱和度、亮度_____T
    elif random_choose ==11:
        random_factor = np.random.randint(low=0, high=21) / 10.0
        return ImageEnhance.Contrast(image).enhance(random_factor)      # 调整图像的对比度_____T
    elif random_choose ==12:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=21) / 10.0
        return ImageEnhance.Contrast(color_image).enhance(random_factor)    # 调整图像的饱和度、对比度_____T
    elif random_choose ==13:
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image = ImageEnhance.Brightness(image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=21) / 10.0
        return ImageEnhance.Contrast(color_image).enhance(random_factor)    # 调整图像的亮度、对比度_____T
    elif random_choose ==14:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image =ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=21) / 10.0 
        return ImageEnhance.Contrast(color_image).enhance(random_factor)     # 调整图像的饱和度、亮度、对比度_____T
    elif random_choose ==15:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像的锐度
    elif random_choose ==16:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的饱和度、锐度
    elif random_choose ==17:
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image =ImageEnhance.Brightness(image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的亮度、锐度
    elif random_choose ==18:
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image =ImageEnhance.Contrast(image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的对比度、锐度
    elif random_choose ==19:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image =ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的饱和度、亮度、锐度
    elif random_choose ==20:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=21) / 10.0 
        color_image = ImageEnhance.Contrast(color_image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的饱和度、对比度、锐度
    elif random_choose ==21:
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image =ImageEnhance.Brightness(image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=21) / 10.0 
        color_image = ImageEnhance.Contrast(color_image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的亮度、对比度、锐度
    elif random_choose ==22:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image =ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的饱和度、亮度、锐度
    elif random_choose ==23:
        random_factor = np.random.randint(low=0, high=31) / 10.0
        color_image = ImageEnhance.Color(image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=21) / 10.0
        color_image =ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(low=0, high=21) / 10.0 
        color_image = ImageEnhance.Contrast(color_image).enhance(random_factor) 
        random_factor = np.random.randint(low=0, high=31) / 10.0
        return ImageEnhance.Sharpness(color_image).enhance(random_factor)  # 调整图像的饱和度、亮度、对比度、锐度
    elif random_choose ==24:
        return sp_noise(cvimage)#椒盐噪声
    else:
        return gasuss_noise(cvimage)#高斯噪声

'''

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    filenames.sort()
    for filename in filenames:
        if filename.endswith('.png'):
            print('正在处理第 %s 张图片' % index)
            img_path = path + '/' + filename
            img = cv2.imread(img_path)
            print('照片文件名：%s' % img_path)

            image = Image.open(img_path)
            cvimg = cv2.imread(img_path)
            cvimage = cv2.resize(cvimg, (width, height))
            image = image.resize((width, height), Image.BILINEAR)  # 双线性法，改变尺寸

            image_rotated_1 = image.rotate(angle1, Image.BICUBIC)#旋转angle角度1
            image_rotated_2 = image.rotate(angle2, Image.BICUBIC)#旋转angle角度2
            xoffset = ImageChops.offset(image,xoff,0)#水平平移 off个像素
            yoffset = ImageChops.offset(image,0,yoff)#水平平移 off个像素
            image_transpose1 = image.transpose(Image.FLIP_LEFT_RIGHT)#左右反转
            image_transpose2 = image.transpose(Image.FLIP_TOP_BOTTOM)#上下反转
            #img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
            random_factor = np.random.randint(low=0, high=31) / 10.0        # 随机的扰动因子
            color_image = ImageEnhance.Color(image).enhance(random_factor)    # 调整图像的饱和度
            random_factor = np.random.randint(low=10, high=21) / 10.0
            temp_1 = ImageEnhance.Brightness(color_image).enhance(random_factor)   # 调整图像的亮度
            bright_image = ImageEnhance.Brightness(image).enhance(random_factor)   # 调整图像的亮度
            random_factor = np.random.randint(low=10, high=21) / 10.0
            temp_2 = ImageEnhance.Contrast(temp_1).enhance(random_factor)  # 调整图像的对比度
            contrast_image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像的对比度
            random_factor = np.random.randint(low=0, high=31) / 10.0
            sharp_image = ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像的锐度
            temp_3 = ImageEnhance.Sharpness(temp_2).enhance(random_factor)  # 调整图像的锐度
            sp_noiseimg=sp_noise(cvimage)#椒盐噪声
            gasuss_noiseimg=gasuss_noise(cvimage)#高斯噪声
            image_rotated_1.save(os.path.join(output_dir, 'a' + str(index) + '.png'))
            image_rotated_2.save(os.path.join(output_dir, 'b' + str(index) + '.png'))
            xoffset.save(os.path.join(output_dir, 'c' + str(index) + '.png'))
            image_transpose1.save(os.path.join(output_dir, 'd' + str(index) + '.png'))
            image_transpose2.save(os.path.join(output_dir, 'e' + str(index) + '.png'))
            color_image.save(os.path.join(output_dir, 'f' + str(index) + '.png'))
            temp_1.save(os.path.join(output_dir, 'g' + str(index) + '.png'))
            bright_image.save(os.path.join(output_dir, 'h' + str(index) + '.png'))
            temp_2.save(os.path.join(output_dir, 'i' + str(index) + '.png'))
            contrast_image.save(os.path.join(output_dir, 'j' + str(index) + '.png'))
            sharp_image.save(os.path.join(output_dir, 'k' + str(index) + '.png'))
            temp_3.save(os.path.join(output_dir, 'm' + str(index) + '.png'))           
            cv2.imwrite(output_dir + '/n' + str(index) + '.png', sp_noiseimg)
            cv2.imwrite(output_dir + '/p' + str(index) + '.png', gasuss_noiseimg)
            index += 1
'''





