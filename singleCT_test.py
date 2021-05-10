# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np
import cv2
import argparse
import tensorflow as tf
 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D

import json
import dicom_read

def parse_args():
    parser = argparse.ArgumentParser(description='demo,you can specify GPU to use')    
    parser.add_argument('--gpu', dest='gpu', help='gpu id ', default='0', type=str)
    args = parser.parse_args()
    return args
    
class U_Net():    
    def __init__(self, test_data_path):
        self.dicom_path = './' + test_data_path
        # 設置圖片基本參數
        self.height = 512
        self.width = 512
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)

        # 優化器
        self.lr = 0.01
        self.optimizer = Adam(self.lr, 0.5)

        # u_net
        self.liver_unet = self.build_unet()  # 創建肝臟分割網絡變量        
        #self.liver_unet.summary()
        self.tumor_unet = self.build_unet()  # 創建肝臟腫瘤分割網絡變量        
        #self.tumor_unet.summary()

    def build_unet(self, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):

        # 定義一個多次使用的捲積塊
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            # the first layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # the second layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            X = Activation('relu')(x)
            return X

        # 構建一個輸入
        img = Input(shape=self.shape)

        # contracting path
        c1 = conv2d_block(img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # extending path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)

        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return Model(img, output)
        
    def load_data(self, path_):
        image = []  # 定義一個空列表，用於保存數據集
        # read image from dicom file
        img = dicom_read.get_dcm_img(path_)/255.
        img = img.astype(np.float32)
        image.append(img) 
        image = np.expand_dims(np.array(image), axis=3)  # 擴展維度，增加第4維(num, 512, 512, 1)
        return image 
        
    def metric_fun(self, y_true, y_pred):
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        return fz / fm    
        
    def tumor_test(self, batch_size=1):
        # 存放測試結果資料夾
        if not osp.exists('./test_result/'):
            os.mkdir('./test_result/')
            
        # 加載已經訓練的肝臟分割模型
        self.liver_unet.load_weights('./weights/liver_lr=0.01-36-0.957.h5')
        self.liver_unet.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[self.metric_fun])
             
        # 加載已經訓練的腫瘤分割模型
        self.tumor_unet.load_weights('./weights/tumor_lr=0.01-54-0.878.h5')    
        self.tumor_unet.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[self.metric_fun])               
            
        # 獲得數據        
        test_img = self.load_data(self.dicom_path)       
        test_num = test_img.shape[0]
        index = 0 
        while index < test_num:
            liver_mask = self.liver_unet.predict(test_img[index:index + batch_size]) > 0.1
            remain = liver_mask==0
            test_img[index:index + batch_size][remain] = 0
                      
            tumor_mask = self.tumor_unet.predict(test_img[index:index + batch_size]) > 0.1       
            
            liver_mask = np.squeeze(np.uint8(liver_mask), axis=0)
            liver_mask = np.squeeze(np.uint8(liver_mask), axis=2)
            tumor_mask = np.squeeze(np.uint8(tumor_mask), axis=0)
            tumor_mask = np.squeeze(np.uint8(tumor_mask), axis=2)
            
            liver_mask = np.array(liver_mask, np.uint8)
            liver_contours, hierarchy = cv2.findContours(liver_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            tumor_mask = np.array(tumor_mask, np.uint8)
            tumor_contours, hierarchy = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            index += batch_size
            
        #print(liver_contours)
        #print(tumor_contours)
        return liver_contours, tumor_contours
                  
def main(test_data_path):
    #args = parse_args()
    #print(args.gpu)
    #print(tf.test.is_gpu_available())
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    unet = U_Net(test_data_path)
    liver_contours, tumor_contours = unet.tumor_test() # 測試腫瘤分割結果
    return liver_contours, tumor_contours

#if __name__ == '__main__':
    #main('Test data/00A08001/00A08001_3_17.dcm')