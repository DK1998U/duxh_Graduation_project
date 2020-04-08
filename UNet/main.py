import model
import tensorflow as tf
import numpy as np
import pdb

tf.reset_default_graph()
#net parameters
channel=4
num_classes = 2
max_value = 2000.0 #用来将数据归一化到0-1
use_cpu=False #True

#training parameters
batch_size = 96
learning_rate = 1e-3
iter_num=2000

#training data dir
train_img_floder = "/home/thanatos/bussiness/Unet/data/train/imgs"  # orignal image path
train_label_floder = "/home/thanatos/bussiness/Unet/data/train/segs" # orignal segmentation path 
eval_img_dir = "/home/thanatos/bussiness/Unet/data/eval_img"
eval_label_dir = "/home/thanatos/bussiness/Unet/data/eval_label" 
model_save_dir = '/home/thanatos/bussiness/Unet/model' #存放训练模型

#test data dir
test_img_floder = '/home/thanatos/bussiness/Unet/data/train/imgs_org' #测试图像path
pred_img_save_floder = '/home/thanatos/bussiness/Unet/data/test/pred' #预测结果保存位置

#model building
model= model.SegNet(channel=channel, num_classes=num_classes, use_cpu=use_cpu, model_dir=model_save_dir, max_value=max_value) #建立网络

#training
model.train(batch_size=batch_size, learning_rate=learning_rate, iter_num=iter_num,train_img_dir=train_img_floder,train_label_dir=train_label_floder,eval_img_dir=eval_img_dir,eval_label_dir=eval_label_dir)


#testing
#model.test(test_img_floder, pred_img_save_floder)


