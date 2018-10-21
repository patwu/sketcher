
import tensorflow as tf
import numpy as np
import shutil
import os

class CNN(object):

    def __init__(self,input_w,input_h, n_channel=32, n_output=2):
        self.w=input_w
        self.h=input_h
        self.n_output=n_output
        self.n_channel=n_channel

        self.build()

    def loss(self,logits,labels):
        ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        loss=tf.reduce_mean(ce)
        return loss

    def forward(self,x):
        h=tf.contrib.layers.conv2d(x,self.n_channel,(3,3),scope='cnn1')
        h=tf.contrib.layers.max_pool2d(h,(2,2),scope='maxpool1')
        h=tf.contrib.layers.conv2d(h,self.n_channel*2,(3,3),scope='cnn2')
        h=tf.contrib.layers.max_pool2d(h,(2,2),scope='maxpool2')
        h=tf.contrib.layers.flatten(h)
        h=tf.contrib.layers.fully_connected(h,self.n_channel*4)
        logit=tf.contrib.layers.fully_connected(h,self.n_output,activation_fn=None)
        pred=tf.nn.softmax(logit)

        return logit,pred

    def build(self):
        global_step = self.global_step=tf.Variable(0, name='global_step', trainable=False)

        x=self.x=tf.placeholder(tf.float32,[None,self.h,self.w,1])
        y=self.y=tf.placeholder(tf.int64,[None])

        logit,pred=self.forward(x)
        self.loss_step=loss=self.loss(logit,y)

        self.prob_step=pred
        self.pred_step=tf.argmax(pred,axis=1)

        opt = tf.train.GradientDescentOptimizer(0.1)
        self.train_step=opt.minimize(loss, global_step=global_step)
       
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver=tf.train.Saver()

    def train(self,x,y):
        x=np.expand_dims(np.asarray(x,dtype=np.float32),3)
        y=np.asarray(y)

        feed = {self.x:x,self.y:y}
        _,loss,pred=self.sess.run([self.train_step,self.loss_step,self.pred_step], feed_dict=feed)
        return pred,loss

    def predict(self,x):
        x=np.expand_dims(np.asarray(x,dtype=np.float32),3)
        feed = {self.x:x}
        pred,prob=self.sess.run([self.pred_step,self.prob_step], feed_dict=feed)
        return pred,prob
       
    def save(self,path):
        if os.path.exists(path):
            shutil.rmtree(path)  
        tf.saved_model.simple_save(self.sess,path,inputs={"x":self.x},outputs={"pred_y":self.prob_step})
        
