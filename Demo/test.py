# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import tensorflow as tf
import cv2
import Dicom_read

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D

def parse_args():
    parser = argparse.ArgumentParser(description='demo,you can specify GPU to use')    
    parser.add_argument('--gpu', dest='gpu', help='gpu id ', default='0', type=str)
    parser.add_argument('-p', '--plot', help='plot the result ', action="store_true")
    parser.add_argument('-lw', '--liver_weight', help='liver weight path ', default='liver_weight.h5', type=str)
    parser.add_argument('-tw', '--tumor_weight', help='tumor weight path ', default='tumor_weight.h5', type=str)
    args = parser.parse_args()
    return args
    
class U_Net():    
    def __init__(self, test_data_path, plot, liver_weight, tumor_weight):
        self.dicom_path = './' + test_data_path + '/'
        self.plot = plot
        self.liver_weight = liver_weight
        self.tumor_weight = tumor_weight
        # 設置圖片基本參數
        self.height = 512
        self.width = 512
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)

        # 優化器
        self.lr = 0.01
        self.optimizer = Adam(self.lr, 0.5)

        # u_net
        self.liver_unet = self.build_unet(input_shape=self.shape)  # 創建肝臟分割網絡變量        
        #self.liver_unet.summary()
        self.tumor_unet = self.build_unet(input_shape=self.shape)  # 創建肝臟腫瘤分割網絡變量        
        #self.tumor_unet.summary()

    def build_unet(self, input_shape, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):
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
        img = Input(shape=input_shape)
        
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

    def metric_fun(self, y_true, y_pred):
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.5), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.5), tf.float32)) + 1e-8
        return fz / fm
    
    def load_data(self, path_):
        counts_most_hu_list = []
        image = []  # 定義一個空列表，用於保存數據集
        dcm_name = [] #用於保存dicom數據集
        # 獲取文件夾名稱
        dcm_path = self.dicom_path + path_ + '/'
        name = os.listdir(dcm_path)
        #name.sort(key=lambda x: int(str(''.join(list(filter(str.isdigit, x)))).lstrip('0')))
        for n_ in name:
            # read image from dicom file
            img, counts_most_hu = Dicom_read.get_dcm_img(dcm_path + n_)
            counts_most_hu_list.append(counts_most_hu)
            img = img/255.
            img = img.astype(np.float32)
            image.append(img)
            dcm_name.append(n_) 
        image = np.expand_dims(np.array(image), axis=3) 
        return image, dcm_name, counts_most_hu_list 
 
    def tumor_test(self):
        # 加載已訓練的肝臟分割模型
        self.liver_unet.load_weights('./weights/' + self.liver_weight)
        self.liver_unet.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[self.metric_fun])
        # 加載已訓練的腫瘤分割模型
        self.tumor_unet.load_weights('./weights/' + self.tumor_weight)      
        self.tumor_unet.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[self.metric_fun])    
        
        dcm_name_list = []
        tumor_contours_list = []        
        
        # 獲取文件夾名稱
        path = os.listdir(self.dicom_path)
        #path.sort(key=lambda x: int(str(''.join(list(filter(str.isdigit, x)))).lstrip('0'))) 
        for path_ in path: # different patient
            # 獲得數據
            test_img, dcm_name, counts_most_hu_list = self.load_data(path_)  # one patient's CT series
            liver_center = np.argmax(counts_most_hu_list)
            
            temp = self.liver_unet.predict(test_img[liver_center:liver_center+1]) > 0.5
            ### use contours    
            temp = np.squeeze(temp, axis=0)
            temp = np.squeeze(temp, axis=2)
            contours, hierarchy = cv2.findContours(np.uint8(temp), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                temp = cv2.rectangle(np.uint8(temp), (x-30,y-30), (x+w+30, y+30+h), 255, -1)
            largest_liver_mask = temp==0 
            largest_liver_mask = np.expand_dims(np.array(largest_liver_mask), axis=0)
            largest_liver_mask = np.expand_dims(largest_liver_mask, axis=3)
            
            test_num = test_img.shape[0]
            for index in range(test_num):
                liver_mask = self.liver_unet.predict(test_img[index:index+1]) > 0.5
                liver_mask[largest_liver_mask] = 0
                
                test_img[index:index+1][liver_mask==0] = 0
                
                tumor_mask = self.tumor_unet.predict(test_img[index:index+1]) > 0.5  
                tumor_mask = np.squeeze(tumor_mask, axis=0)
                tumor_mask = np.squeeze(tumor_mask, axis=2)           
                tumor_mask = np.array(tumor_mask, np.uint8)
                
                tumor_contours, hierarchy = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                
                if len(tumor_contours) > 0:
                    dcm_name_list.append(dcm_name[index])
                    
                    temp = np.concatenate(([tumor_contours[i] for i in range(len(tumor_contours))]), axis=0)
                    temp = np.squeeze(temp, axis=1)
                    tumor_contours_list.append(temp)    
                    
                    # 存放預測結果
                    if self.plot:
                        if not os.path.exists('./Prediction/'+ path_ + '/'):
                            os.mkdir('./Prediction/'+ path_ + '/')                        
                        img, counts_most_hu = Dicom_read.get_dcm_img(self.dicom_path + path_ + '/' + dcm_name[index])
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
                        cv2.drawContours(img, tumor_contours, -1, (0, 0, 255), 1)  
                        cv2.imwrite('./Prediction/'+ path_ + '/' + str(index) + '.png', img)
                        
        return dcm_name_list, tumor_contours_list

def main(test_data_path):
    args = parse_args()
    print(args.gpu)
    print(args.plot)
    print(args.liver_weight)
    print(args.tumor_weight)
    print(tf.test.is_gpu_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    unet = U_Net(test_data_path, args.plot, args.liver_weight, args.tumor_weight)
    dcm_name_list, tumor_contours_list = unet.tumor_test() # 測試腫瘤分割結果
    print('DICOM name with liver tumor : ', dcm_name_list)
    print('Liver tumor coordinate : ', tumor_contours_list)
    return dcm_name_list, tumor_contours_list    
        
if __name__ == '__main__':
    main('Test data')