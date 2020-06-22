import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from d_edge_crf_s import *
BATCH_SIZE = 8
model = tf.keras.models.load_model("./result/best_models/FCRN_sobel_s_new.h5",custom_objects={'d_sobel_loss_s': d_sobel_loss_s})
#model = tf.keras.models.load_model("./result/best_models/FCRN_s.h5")     #这个是加载基础主干网络用的

########获取测试集数据##########
src_dir = './dataset/larger_testset/test_img/'
dst_dir = './dataset/larger_testset/test_label/'
src_imgs=[]
dst_imgs=[]
src_img_path=[]
for (path, dirnames, filenames) in sorted(os.walk(os.path.abspath(src_dir))):
    filenames.sort()
    for filename in filenames:
        if filename.endswith('.png'):
            src_img_path.append(os.path.abspath(os.path.join(path ,filename)))
            src_imgs.append(cv2.imread(os.path.abspath(os.path.join(path ,filename))))
            dst_imgs.append(cv2.imread(os.path.abspath(os.path.join(dst_dir,filename))))
print(len(src_imgs))
print(len(dst_imgs))

########计算平均交并比MIoU######################
def compute_IoU(p, l, num=0.5):
    p = tf.where(p>num, tf.ones_like(p), tf.zeros_like(p))
    l = tf.where(l>num, tf.ones_like(l), tf.zeros_like(l))
    temp1 = tf.where(tf.equal(p, 0), tf.ones_like(p), tf.zeros_like(p))
    temp2 = tf.where(tf.equal(l, 1), tf.ones_like(p), tf.zeros_like(p))
    temp3 = tf.multiply(temp1, temp2)
    
    temp4 = tf.where(tf.equal(p, 1), tf.ones_like(p), tf.zeros_like(p))
    temp5 = tf.where(tf.equal(l, 0), tf.ones_like(p), tf.zeros_like(p))
    temp6 = tf.multiply(temp4, temp5)
    
    #temp7 = tf.where(tf.equal(p, l), tf.ones_like(p), tf.zeros_like(p))
    temp7 = tf.multiply(p, l)
    
    temp8 = tf.reduce_sum(temp3)
    temp9 = tf.reduce_sum(temp6)
    temp10 = tf.reduce_sum(temp7)
    output = temp10/(temp8+temp9+temp10)
    
    return output

##########计算像素准确度############
def compute_PA_s_TF(label,pred,rate = None):
    '''
    输入label为bs,256,256,3 opencv读取的图像batch
    pred为网络输出
    rate 0-1
    '''
    nlabel = tf.math.reduce_sum(label,axis=-1)/255
    #nlabel = label
    pred = tf.math.reduce_sum(pred,axis=-1)
    m = tf.keras.metrics.BinaryAccuracy()
    if rate != None:
        #print("123")
        pred = tf.where(tf.math.greater(pred,rate),tf.ones_like(pred),tf.zeros_like(pred))
    m.update_state(nlabel, pred)
    return m.result().numpy() # Final result

#为了展示效果，转为彩色图像
def to_green(pred,rate = 0.75):
    #input is the output of network
    npre = tf.where(tf.math.greater(pred,rate),tf.ones_like(pred),tf.zeros_like(pred))
    a = tf.concat([tf.zeros_like(pred),pred],axis = -1)
    nnpre = tf.concat([a, tf.zeros_like(pred)],axis = -1)
    return nnpre


#网络测试输出保存地址
tempdir = "./result/best_models/FCRN_sobel_s_lambda/"
if not os.path.exists(os.path.abspath(tempdir)):
    os.makedirs(os.path.abspath(tempdir))
aimg = np.zeros((BATCH_SIZE,256,256,3),dtype = np.float32)
alabel = np.zeros((BATCH_SIZE,256,256,3),dtype = np.uint8)
PA_log = []
start_time = time.time()
sum_iou = 0
for i in range(int(len(src_img_path)/BATCH_SIZE)):
    for j in range(BATCH_SIZE):
        img = src_imgs[i*BATCH_SIZE+j]/255.0
        aimg[j] = img
        label = dst_imgs[i*BATCH_SIZE+j]
        alabel[j] = label
    pred = model.predict(aimg)
    
    newpred = to_green(pred)
    PA= compute_PA_s_TF(alabel,pred)
    PA_log.append(PA)
    for j in range(BATCH_SIZE):
        filename = src_img_path[i*BATCH_SIZE+j].split("/")
        temp = os.path.join(tempdir,filename[-1])
        cv2.imwrite(temp,np.array(newpred[j]*255.0,dtype=np.uint8))
        
        sum_iou += compute_IoU(pred[j],tf.cast(tf.math.reduce_sum(alabel[j]/255.0,axis=-1,keepdims=True),dtype=tf.float32),0.75)
        
print("MIoU:")
print((sum_iou/int(len(src_img_path)/BATCH_SIZE))/BATCH_SIZE)
a = np.array(PA_log,dtype = np.float32)
print("PA:")
print(np.mean(a))
print(time.time() - start_time)
