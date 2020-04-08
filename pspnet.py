#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:04:49 2019

@author: thanatos
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 使用第一块GPU
tf.reset_default_graph()

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Dataset/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "models/", "Path to save model mat")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224
def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    #stamp = ("".join(time_stamp.split()[0].split("-"))+"".join(time_stamp.split()[1].split(":"))).replace('.', '')
    return time_stamp

def avg_pool(input, k_h, k_w, s_h, s_w, name, padding='VALID'):
    output = tf.nn.avg_pool(input,ksize=[1, k_h, k_w, 1],strides=[1, s_h, s_w, 1],padding=padding,name=name,data_format=DEFAULT_DATAFORMAT)
    return output

def batch_normalization(input, name, scale_offset=True, relu=False):
    output = tf.layers.batch_normalization(input,momentum=0.95,epsilon=1e-5,training=self.is_training,name=name)
    if relu:
        output = tf.nn.relu(output)
    return output

def resize_bilinear(input, size, name):
    return tf.image.resize_bilinear(input, size=size, align_corners=True, name=name)


def D_PSP(x,kernel_num):
    
    #shape = x.get_shape().as_list()
    #print(x.get_shape().as_list())
    shape = tf.shape(x)[1:3]
    psp_half = slim.avg_pool2d(x, [2, 2], scope='psp_half')
    psp_small = slim.max_pool2d(psp_half, [2, 2], scope='psp_small')
    psp_min = slim.max_pool2d(psp_small, [2, 2], scope='psp_min')
    #print(psp_min.get_shape().as_list())
    psp_whole = slim.conv2d(x, kernel_num, kernel_size=[1,1], stride=1, activation_fn = tf.nn.relu)
    psp_min = slim.conv2d(psp_min, kernel_num, kernel_size=[1,1], stride=1, activation_fn = tf.nn.relu)
    psp_half = slim.conv2d(psp_half, kernel_num, kernel_size=[1,1], stride=1, activation_fn = tf.nn.relu)
    psp_small = slim.conv2d(psp_small, kernel_num, kernel_size=[1,1], stride=1, activation_fn = tf.nn.relu)
    '''

    psp_half_u = slim.conv2d_transpose(psp_half, kernel_num, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
    psp_small_u = slim.conv2d_transpose(psp_small, kernel_num, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
    psp_small_u = slim.conv2d_transpose(psp_small_u, kernel_num, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)



    psp_min_u = slim.conv2d_transpose(psp_min, kernel_num, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
    print(psp_min_u.get_shape().as_list())
    psp_min_u = slim.conv2d_transpose(psp_min_u, kernel_num, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
    psp_min_u = tf.pad(psp_min_u,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
    print(psp_min_u.get_shape().as_list())
    psp_min_u = slim.conv2d_transpose(psp_min_u, kernel_num, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
    print(psp_min_u.get_shape().as_list())
    '''
    psp_half_u = resize_bilinear(psp_half,shape,'psp_half_u')
    psp_min_u = resize_bilinear(psp_min,shape,'psp_min_u')
    psp_small_u = resize_bilinear(psp_small,shape,'psp_small_u')
    psp_whole_u = resize_bilinear(psp_whole,shape,'psp_whole_u')

    psp_out = tf.concat([psp_min_u,psp_half_u,psp_small_u,psp_whole,x],-1)
    return psp_out

def FCN(x,keep_prob):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('FCN')]) > 0
    #print (x.get_shape())
    with tf.variable_scope('FCN', reuse = reuse):
        shape = tf.shape(x)[1:3]
        x = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        x = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        x = slim.max_pool2d(x, [2, 2], scope='pool1')#2x
        y_p1=x
        x = slim.conv2d(x, 128, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        x = slim.conv2d(x, 128, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        x = slim.max_pool2d(x, [2, 2], scope='pool2')#4x
        y_p2=x
        for i in range(3):
            x = slim.conv2d(x, 256, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        x = slim.max_pool2d(x, [2, 2], scope='pool3')#8x
        y_p3=x
        for i in range(3):
            x = slim.conv2d(x, 512, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob=keep_prob)
        x = slim.max_pool2d(x, [2, 2], scope='pool4')#16x
        
        y_p4=x
        for i in range(3):
            x = slim.conv2d(x, 512, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        '''
        x = slim.max_pool2d(x, [2, 2], scope='pool5')#32x
        y_p5=x
        
        #JPU
        t_conv4 = slim.conv2d_transpose(y_p4, 256, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        x = tf.add(t_conv4,y_p3,name=None)
        t_conv5 = slim.conv2d_transpose(y_p5, 512, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        t_conv5 = slim.conv2d_transpose(t_conv5, 256, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        x = tf.add(t_conv5,x,name=None)

        s_conv1 = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu,rate=1)
        s_conv2 = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu,rate=2)
        s_conv3 = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu,rate=4)
        s_conv4 = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu,rate=8)
        x = tf.concat([s_conv1,s_conv2,s_conv3,s_conv4],3)
        x = slim.conv2d(x, 64, kernel_size=[3,3], stride=1, activation_fn = tf.nn.relu)
        
        print(x.get_shape().as_list())
        x = slim.conv2d_transpose(x, 512, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        print(x.get_shape().as_list())
        '''
        x = tf.add(x,y_p4,name=None)
        #PSP
        x = D_PSP(x,256)
        print(x.get_shape().as_list())
        x = slim.conv2d_transpose(x, 256, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        x = tf.add(x,y_p3,name=None)
        x = slim.conv2d_transpose(x, 128, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        #
        x = slim.conv2d_transpose(x, 64, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        
        x = slim.conv2d_transpose(x, 64, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.tanh)

        x = tf.nn.dropout(x, keep_prob=keep_prob)
        z = slim.conv2d(x, 3, kernel_size=[3,3], stride=1, activation_fn = tf.nn.sigmoid)
       
    return z, y_p1




def focal_loss(pred, y, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
    Multi-labels Focal loss formula:
    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
    ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
    pred: A float tensor of shape [batch_size, num_anchors,
    num_classes] representing the predicted logits for each class
    y: A float tensor of shape [batch_size, num_anchors,
    num_classes] representing one-hot encoded classification targets
    alpha: A scalar tensor for focal loss alpha hyper-parameter
    gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
    loss: A (scalar) tensor representing the value of the loss function
    """
    zeros = tf.zeros_like(pred, dtype=pred.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = tf.where(y > zeros, y - pred, zeros) # positive sample 寻找正样本，并进行填充

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(y > zeros, zeros, pred) # negative sample 寻找负样本，并进行填充
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)

batch_size=2

#gradient_img=np.zeros([batch_size,IMAGE_SIZE,IMAGE_SIZE,1],dtype=np.float32)
out_img2=np.zeros([batch_size,IMAGE_SIZE,IMAGE_SIZE,3],dtype=np.float32)
lab_img2=np.zeros([batch_size,IMAGE_SIZE,IMAGE_SIZE,3],dtype=np.float32)

out_img = tf.random_uniform([batch_size,IMAGE_SIZE,IMAGE_SIZE,3],minval=0, maxval=1.0, dtype=tf.float32)
lab_img = tf.random_uniform([batch_size,IMAGE_SIZE,IMAGE_SIZE,3],minval=0, maxval=1.0, dtype=tf.float32)
keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
pre, tmp = FCN(out_img,keep_probability)

# loss
err_g=tf.subtract(lab_img,pre)
loss_focal=tf.reduce_mean(err_g**2)
tf.summary.scalar('loss', loss_focal)


#cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=tf.nn.softmax(pre), labels=tf.nn.softmax(lab_img))

# Create a tensor named cross_entropy for logging purposes.
#tf.identity(cross_entropy, name='cross_entropy')
#loss_summary = tf.summary.scalar('cross_entropy', cross_entropy)

#loss_focal=focal_loss(pre,lab_img)




gen_global_step = tf.Variable(0, trainable=False)
train_epoch = tf.Variable(0, trainable=False)
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'FCN' in var.name]

train_gen = tf.train.AdamOptimizer(0.00001).minimize(loss_focal, var_list = g_vars, global_step = gen_global_step)


saver = tf.train.Saver()

with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./summary/train',sess.graph)
        test_writer = tf.summary.FileWriter('./summary/test')
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("./Chect_point_pspnet/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        #数据预处理
        data_dir = "../dataset/d_data/train_224/"
        test_dir = "../dataset/d_data/test_224/"
        if not os.path.exists(data_dir):
            print("ERROR: No Data Found!!!")
        else:
            for (path, dirnames, img_filenames) in os.walk(os.path.abspath(os.path.join(data_dir,"img"))):
                img_filenames.sort()
            for (path, dirnames, label_filenames) in os.walk(os.path.abspath(os.path.join(data_dir,"label"))):
                label_filenames.sort()
            label_arr = np.array(label_filenames)
            img_arr = np.array(img_filenames)
            data_size = len(img_arr)
            print("data_size:",data_size)
            img_arr = img_arr.reshape(data_size,1)
            label_arr = label_arr.reshape(data_size,1)
            d_sample = np.hstack((img_arr,label_arr))#[0]img,[1]label
        if not os.path.exists(test_dir):
            print("ERROR: No test Data Found!!!")
        else:
            for (path, dirnames, test_img_filenames) in os.walk(os.path.abspath(os.path.join(test_dir,"img"))):
                test_img_filenames.sort()
            for (path, dirnames, test_label_filenames) in os.walk(os.path.abspath(os.path.join(test_dir,"label"))):
                test_label_filenames.sort()
            test_label_arr = np.array(test_label_filenames)
            test_img_arr = np.array(test_img_filenames)
            test_data_size = len(test_img_arr)
            print("test_data_size:",test_data_size)
            test_img_arr = test_img_arr.reshape(test_data_size,1)
            test_label_arr = test_label_arr.reshape(test_data_size,1)
            d_test = np.hstack((test_img_arr,test_label_arr))#[0]img,[1]label
        index=0
        t_index=0
        train_epoch = (gen_global_step * batch_size) // data_size
        for iter in range(500000):
            #start_t = time.time()
            for j in range(batch_size):
                if(index+j)>=data_size:
                    np.random.shuffle(d_sample)
                    index = 0
                out_img2[j] = cv2.imread(os.path.join(data_dir,"img",d_sample[index+j][0])) / 255.0
                lab_img2[j] = cv2.imread(os.path.join(data_dir,"label",d_sample[index+j][1])) / 255.0
            index+=batch_size
            
            feeds = {out_img: out_img2, lab_img: lab_img2,keep_probability: 0.5}
            t_epoch,summary,l_fo, _, l_g_step= sess.run([train_epoch,merged,loss_focal, train_gen, gen_global_step],feeds)

            if  iter % 1000 == 0:
                feeds = {out_img: out_img2, lab_img: lab_img2,keep_probability: 1.0}
                #summary,l_fo, _,t_epoch= sess.run([merged,loss_focal, train_gen,train_epoch],feeds)
                train_writer.add_summary(summary, iter)
                print("%s  loss=%f, epoch: %d ,step=%d:"%(get_time_stamp(),l_fo,t_epoch,l_g_step))
                print("iter:", '%04d' % (iter + 1))
                #t_img=sess.run(gen,feeds)
                t_img=lab_img2
                t_img0=255.0*t_img[0,:,:,:]
                str="./picpsp20/t%d.jpg"%(iter)
                cv2.imwrite(str,  t_img0)
                t_img=out_img2
                t_img0=255.0*t_img[0,:,:,:]
                str="./picpsp20/t%d_a.jpg"%(iter)
                cv2.imwrite(str,  t_img0)
                t_img=sess.run(pre,feeds)
                t_img0=255.0*t_img[0,:,:,:]
                str="./picpsp20/t%d_b.jpg"%(iter)
                cv2.imwrite(str,  t_img0)

            if  iter % 5000 == 0:
                print("testing")
                start_t = cv2.getTickCount()
                np.random.shuffle(d_test)
                for j in range(batch_size):
                    out_img2[j] = cv2.imread(os.path.join(test_dir,"img",d_test[j][0])) / 255.0
                    lab_img2[j] = cv2.imread(os.path.join(test_dir,"label",d_test[j][1])) / 255.0
                feeds = {out_img: out_img2, lab_img: lab_img2,keep_probability: 1.0}
                #summary,l_fo, _, l_g_step,t_epoch= sess.run([merged,loss_focal, train_gen, gen_global_step,train_epoch],feeds)
                #test_writer.add_summary(summary, iter)
                #t_img=sess.run(gen,feeds)
                t_img=lab_img2
                t_img0=255.0*t_img[0,:,:,:]
                str="./picpsp2/t%d.jpg"%(iter)
                cv2.imwrite(str,  t_img0)
                t_img=out_img2
                t_img0=255.0*t_img[0,:,:,:]
                str="./picpsp2/t%d_a.jpg"%(iter)
                cv2.imwrite(str,  t_img0)
                t_img=sess.run(pre,feeds)
                t_img0=255.0*t_img[0,:,:,:]
                str="./picpsp2/t%d_b.jpg"%(iter)
                cv2.imwrite(str,  t_img0)
                print("testing_time_cost:",(cv2.getTickCount()-start_t)/cv2.getTickFrequency())
                saver.save(sess,'./Chect_point_pspnet/psp.ckpt',iter)
            #print("cost_time:",time.time()-start_t)
        print("完成!")


