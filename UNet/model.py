
import random
import tensorflow as tf
import time
import numpy as np
from skimage import io
from scipy import misc
import glob
import pdb       #pdb，交互的源代码调试功能
import os
import tensorflow.contrib.slim as slim  
import rgb_label

class SegNet:
    def __init__(self, channel, num_classes, use_cpu, model_dir, max_value):   #定义初始化
        self.channel=channel
        self.num_classes = num_classes
        self.use_cpu = use_cpu
        self.model_dir = model_dir
        self.max_value = max_value
        self.broader = 7

    def train(self, learning_rate, batch_size, iter_num, train_img_dir, train_label_dir, eval_img_dir, eval_label_dir):#定义训练模型
        self.is_training = True
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iter_num = iter_num
    
        self.build(use_cpu=self.use_cpu)
        
        saver = tf.train.Saver()
        
        config = tf.ConfigProto(allow_soft_placement=True)     #保存模型
        with tf.Session(config=config) as session:
            tf.set_random_seed(123)
            session.run(tf.global_variables_initializer())
            if os.path.exists(self.model_dir):
                name = tf.train.latest_checkpoint(self.model_dir)
                #pdb.set_trace()
                #name = 'model\\model.ckpt-0'
                #saver.restore(session, name)
	 
            start_time = time.time()      #计时器
            trainset = glob.glob(train_img_dir+'/*.tif')
            for step in range(self.iter_num):
                batch_x, batch_y = self.traindataload(self.batch_size, train_img_dir, train_label_dir, trainset)
                #print(batch_y.shape)
                #batch_y -= 1 #label from 1 to 6, 6 class         
                #batch_y = batch_y[:,self.broader:-self.broader,self.broader:-self.broader]
                #d_note 注释掉了上面一行
                batch_spw = np.ones_like(batch_y, dtype=np.float32)
                #batch_spw[np.where(batch_y==1)] = 1.2   
                #pdb.set_trace()
                #pdb.set_trace()
                session.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.spw:batch_spw})
                #pdb.set_trace()
                if step%2==0:
                    logits, loss, acc = session.run([self.logits, self.loss, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.spw:batch_spw})
                    elapsed=time.time()-start_time
                    #
                    print("step: " + str(step) + ', loss: ' + str(loss) 
                          + ', acc: ' + str(acc) + ". elapsed time:" + str(elapsed) + 's')
                    #pdb.set_trace()
                if step%1000==0:
                    #self.eval(session, eval_img_dir, eval_label_dir)
                    saver.save(session, self.model_dir + '/model.ckpt', step)
            saver.save(session, self.model_dir + '/model.ckpt', self.iter_num)

    def eval(self, session, eval_img_dir, eval_label_dir):	
        print('\nevalution ...')
        evalset = glob.glob(eval_img_dir+'/*.tif')
        turn_num = len(evalset)//self.batch_size +1
        correct = []
        for i in range(turn_num):
            batch_x, batch_y = self.traindataload(self.batch_size, eval_img_dir, eval_label_dir)
            batch_y -= 1 #label from 1 to 6, 6 class         
            batch_y = batch_y[:,self.broader:-self.broader,self.broader:-self.broader]

            correct_current = session.run(self.correct_pred, feed_dict={self.x: batch_x, self.y: batch_y})
            correct.append(correct_current)      
        print('acc: ' + str(np.mean(correct)) + '\n')


    def test(self, test_img_dir, pred_img_save_floder):	
        self.is_training = False
        self.build(use_cpu=self.use_cpu)
        saver = tf.train.Saver()
        name = tf.train.latest_checkpoint(self.model_dir)
        #pdb.set_trace()      
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as session:
            session.run(tf.global_variables_initializer())
            saver.restore(session, name)

            testset = glob.glob(test_img_dir+'/*.tif')
            num_test = len(testset)  
            #pdb.set_trace()  
            for i in range(num_test):
                batch_x, imagename = self.testdataload(testset, i)
                batch_x = batch_x[:,0:3000,2800:2900,:] #太大了测不动
                prediction = session.run(self.pred, feed_dict={self.x: batch_x})
                #np.save('./testdata/prediction/label/'+imagename+'_label.npy', prediction)
                #predicte_color_im = Image.fromarray(prediction)
                #pdb.set_trace()  
                io.imsave(pred_img_save_floder+ '/' +imagename+'.jpg', rgb_label.label2rgb(prediction[0,:,:]))

    def traindataload(self, batch_size, train_img_dir, train_label_dir, trainset):	   
        images = []
        labels = []
        for i in range(batch_size):
            image, label = self.traindataload_one(train_img_dir, train_label_dir, trainset)
            images.append(image)
            labels.append(label)
        return np.concatenate(images, 0), np.concatenate(labels, 0)

    def traindataload_one(self, train_img_dir, train_label_dir, trainset):	           
        #
        image_file = random.choice(trainset).split('/')[-1]        
        im = io.imread(train_img_dir+'/'+image_file)
        image = np.float32(im/self.max_value)
        label = io.imread(train_label_dir+'/'+image_file)
        
        label = np.int32(label)
        m,n,c = image.shape
        image.shape = 1,m,n,c
        label.shape = 1,m,n
        return image, label

    def testdataload(self, testset, ith):	       
        image_file = testset[ith]
        image = np.float32(io.imread(image_file)/self.max_value)
        m,n,c = image.shape
        image.shape = 1,m,n,c
        return image, image_file.split('/')[1].split('.')[-1]

    def build(self, use_cpu=False):
        '''
        use_cpu allows you to test or train the network even with low GPU memory
        anyway: currently there is no tensorflow CPU support for unpooling respectively
        for the tf.nn.max_pool_with_argmax metod so that GPU support is needed for training
        and prediction
        '''
        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'
        if self.is_training:
            padding = 'SAME'   
            keep_dropout = 0.5        
        else:
            padding = 'VALID'
            keep_dropout = 1.0
        
        with tf.device(device):
            self.x = tf.placeholder(tf.float32, [None, 32, 32, self.channel])
            self.y = tf.placeholder(tf.int32, [None, 32, 32])
            self.spw = tf.placeholder(tf.float32, [None, None, None])
            #self.mask = tf.placeholder(tf.int32, [None, None, None, self.num_classes])
            
            conv_1_1 = self.conv2d(self.x, 32, [3, 3], is_training=self.is_training, name='conv_1_1', padding=padding)          
            
            conv_1_2 = self.conv2d(conv_1_1, 32, [3, 3], is_training=self.is_training, name='conv_1_2', padding=padding)
            pool1 = self.max_pool(conv_1_2)
            print(pool1.shape.as_list())
            conv_2_1 = self.conv2d(pool1, 64, [3, 3], is_training=self.is_training, name='conv_2_1', padding=padding)          
            conv_2_2 = self.conv2d(conv_2_1, 64, [3, 3], is_training=self.is_training, name='conv_2_2', padding=padding)
            pool2 = self.max_pool(conv_2_2)
            for_output2_20=self.conv2d_transpose2(pool2,64,[3,3],is_training=self.is_training,strides=2,name='for_output2_20_transpose')
            for_output2_2=self.conv2d_transpose2(for_output2_20,64,[3,3],is_training=self.is_training,strides=2,name='for_output2_2_transpose')
            print(for_output2_2.shape.as_list())
            conv_3_1 = self.conv2d(pool2, 128, [3, 3], is_training=self.is_training, name='conv_3_1', padding=padding)          
            conv_3_2 = self.conv2d(conv_3_1, 128, [3, 3], is_training=self.is_training, name='conv_3_2', padding=padding)
            pool3 = self.max_pool(conv_3_2)
            for_output2_30=self.conv2d_transpose2(pool3,128,[3,3],is_training=self.is_training,strides=2,name='for_output2_30_transpose')
            for_output2_31=self.conv2d_transpose2(for_output2_30,64,[3,3],is_training=self.is_training,strides=2,name='for_output2_3_21_transpose')
            for_output2_32=self.conv2d_transpose2(for_output2_31,32,[3,3],is_training=self.is_training,strides=2,name='for_output2_3_22_transpose')
            print(for_output2_32.shape.as_list())
            conv_4_1 = self.conv2d(pool3, 256, [3, 3], is_training=self.is_training, name='conv_4_1', padding=padding)          
            conv_4_2 = self.conv2d(conv_4_1, 256, [3, 3], is_training=self.is_training, name='conv_4_2', padding=padding)
            pool4 = self.max_pool(conv_4_2)
            

            unpool4 = self.bicubic(pool4,tf.shape(conv_4_2))
            concat4 = tf.concat([conv_4_2, unpool4], axis=3, name='concat4')   
            deconv_4_2 = self.conv2d_transpose(concat4, 256, [3, 3], is_training=self.is_training, name='deconv_4_2', padding=padding)
            deconv_4_1 = self.conv2d_transpose(deconv_4_2, 128, [3, 3], is_training=self.is_training, name='deconv_4_1', padding=padding)

            unpool3 = self.bicubic(deconv_4_1,tf.shape(conv_3_2))
            concat3 = tf.concat([conv_3_2, unpool3], axis=3, name='concat3')   
            deconv_3_2 = self.conv2d_transpose(concat3, 128, [3, 3], is_training=self.is_training, name='deconv_3_2', padding=padding)
            deconv_3_1 = self.conv2d_transpose(deconv_3_2, 64, [3, 3], is_training=self.is_training, name='deconv_3_1', padding=padding)
            
            unpool2 = self.bicubic(deconv_3_1,tf.shape(conv_2_2))
            concat2 = tf.concat([conv_2_2, unpool2], axis=3, name='concat2')   
            print(concat2.shape.as_list())
            deconv_2_2 = self.conv2d_transpose(concat2, 64, [3, 3], is_training=self.is_training, name='deconv_2_2', padding=padding)
            print(deconv_2_2.shape.as_list())
            deconv_2_1 = self.conv2d_transpose(deconv_2_2, 32, [3, 3], is_training=self.is_training, name='deconv_2_1', padding=padding)
              
            unpool1 = self.bicubic(deconv_2_1,tf.shape(conv_1_2))
            #net =  self.conv2d_transpose(unpool1, 32, [1, 1], is_training=self.is_training, name='fc1', padding=padding)
            #net = tf.nn.dropout(net, keep_dropout)
            net =  self.conv2d(unpool1, self.num_classes, [1, 1], is_training=self.is_training, name='fc', padding=padding)
            print(net.shape.as_list())
            self.logits = tf.reshape(net, [-1, self.num_classes])
            
            
            for_output2_concat1 = tf.concat([conv_1_2, for_output2_2], axis=3, name='for_output2_concat1') 
            for_output2_concat2 = tf.concat([for_output2_concat1, for_output2_32], axis=3, name='for_output2_concat2')
            for_output2_conv1=self.conv2d(for_output2_concat2,128,[3,3], is_training=self.is_training, name='for_output2_conv1', padding='SAME')
            for_output2_conv2=self.conv2d(for_output2_conv1,64,[3,3], is_training=self.is_training, name='for_output2_conv2', padding='SAME')
            for_output2_conv3=self.conv2d(for_output2_conv1,32,[3,3], is_training=self.is_training, name='for_output2_conv3', padding='SAME')
            output2 =  self.conv2d(for_output2_conv3, self.num_classes, [1, 1], is_training=self.is_training, name='output2', padding=padding)
            print(output2.shape.as_list())
            if self.is_training:
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(self.y, [-1]) , logits=self.logits)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optimizer  = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            self.prob = tf.reshape( tf.nn.softmax(self.logits), tf.shape(net))
            self.pred = tf.argmax(self.prob, 3)

            self.correct_pred = tf.equal(self.pred, tf.to_int64(self.y))             
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))



    def max_pool(self, inp, k=2):  
        return tf.nn.max_pool(inp, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

    def bicubic(self, inp, out_shape):
        new_height =  out_shape[1]
        new_width  = out_shape[2]
        return tf.image.resize_bilinear(inp, [new_height, new_width])

    def batch_normalization(self, data, is_training, name, reuse=None):
        return tf.layers.batch_normalization(data, momentum=0.9, training=is_training,
                                             beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                             gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                             reuse=reuse, name=name)

    def conv2d(self, input, output, kernel_size, is_training, name, padding="SAME",
               with_bn=True, activation=tf.nn.elu):
        conv2d = tf.layers.conv2d(input, output, kernel_size=kernel_size, strides=(1, 1), padding=padding,
                                  activation=activation,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                  name=name)
        if with_bn:
            conv2d = self.batch_normalization(conv2d, is_training, name + '_bn')

        return activation(conv2d) if activation is not None else conv2d

    def conv2d_transpose(self, input, output, kernel_size, is_training, name, padding="SAME",
                         reuse=None, with_bn=True, activation=tf.nn.elu):
        conv2d_transpose = tf.layers.conv2d_transpose(input, output, kernel_size=kernel_size, strides=(1, 1),
                                                      padding=padding,
                                                      activation=activation,
                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                                      reuse=reuse, name=name, use_bias=not with_bn)
        return self.batch_normalization(conv2d_transpose, is_training, name + '_bn',
                                        reuse) if with_bn else conv2d_transpose
                                        
    def conv2d_transpose2(self, input, output, kernel_size, is_training,strides, name, padding="SAME",
                         reuse=None, with_bn=True, activation=tf.nn.elu):
        conv2d_transpose = tf.layers.conv2d_transpose(input, output, kernel_size=kernel_size, strides=strides,
                                                      padding=padding,
                                                      activation=activation,
                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                                      reuse=reuse, name=name, use_bias=not with_bn)
        return self.batch_normalization(conv2d_transpose, is_training, name + '_bn',
                                        reuse) if with_bn else conv2d_transpose
    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)


        

        

        
        
      


