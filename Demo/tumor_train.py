# -*- coding: utf-8 -*-
import os
import os.path as osp
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser(description='demo,you can specify GPU to use')    
    parser.add_argument('--gpu', dest='gpu', help='gpu id ', default='0', type=str)
    parser.add_argument('-w', '--weight', help='choose the liver segmentation model weight ', type=str)
    args = parser.parse_args()
    return args
    
class U_Net():    
    def __init__(self, weight):
        self.dicom_path = './Train data/Dicom file/'
        self.label_path = './Train data/Label file/'
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
        self.weight = weight
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

    def metric_fun(self, y_true, y_pred):
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.5), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.5), tf.float32)) + 1e-8
        return fz / fm
    
    def load_data(self):
        print('Loading data......')
        image = []  # 定義一個空列表，用於保存數據集
        label = []
        # 獲取文件夾名稱
        path = os.listdir(self.label_path)
        for path_ in path: 
            dcm_path = self.dicom_path + path_ + '/'
            lbl_path = self.label_path + path_ + '/'
            name = os.listdir(lbl_path)
            for n_ in name:
                # read image from dicom file
                img, _ = Dicom_read.get_dcm_img(dcm_path + n_[:-3] + 'dcm')
                img = img/255.
                img = img.astype(np.float32)
                image.append(img)
                # read label
                img = np.array(cv2.imread(lbl_path + n_[:-3] + 'png', 0))
                img[img == 127] = 0
                img[img == 255] = 1
                label.append(img)
        
        image = np.expand_dims(np.array(image), axis=3)  # 擴展維度，增加第4維
        label = np.expand_dims(np.array(label), axis=3)  # 變為網絡需要的輸入維度(num, 512, 512, 1)
        print('image.shape',label.shape)
        return image, label
    
    def liver_seg(self, batch_size=1):
        # 加載已經訓練的肝臟分割模型
        self.liver_unet.load_weights('./weights/' + self.weight)            
        # 獲得數據
        img, lbl = self.load_data()
        num = img.shape[0]
        index = 0
        self.liver_unet.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[self.metric_fun])
        while index < num:
            mask = self.liver_unet.predict(img[index:index + batch_size]) > 0.1
            remain = mask==0
            
            img[index:index + batch_size][remain] = 0
            lbl[index:index + batch_size][remain] = 0                     
            index += batch_size            
        return img, lbl
        
    def tumor_train(self, epochs=200, batch_size=8):
        # 存放權重資料夾
        if not osp.exists('./weights/'):
            os.mkdir('./weights/')
        # 獲得數據
        train_img, train_lbl = self.liver_seg()
        
        np.random.seed(115)  # 設置相同的隨機種子，確保數據匹配
        np.random.shuffle(train_img)  # 對第一維度進行亂序
        np.random.seed(115)
        np.random.shuffle(train_lbl)
        
        print('Start training......')       
        # 設置訓練的checkpoint
        filepath = './weights/tumor' + '-{epoch:02d}-{val_metric_fun:.3f}.h5'
        callbacks = [EarlyStopping(patience=100, verbose=2),
            ReduceLROnPlateau(factor=0.5, patience=15, min_lr=0.00005, verbose=2),
            ModelCheckpoint(filepath, monitor='val_metric_fun', verbose=1, save_best_only=True, mode='max')]
        w = np.sum(train_lbl)/(train_lbl.shape[0]*512*512)
        self.tumor_unet.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[self.metric_fun])
        # 進行訓練
        results = self.tumor_unet.fit(train_img, train_lbl, batch_size=batch_size, epochs=epochs, verbose=1,
                    callbacks=callbacks, validation_split=0.1, shuffle=True, class_weight=[w, 1-w])

if __name__ == '__main__':
    args = parse_args()
    print(args.gpu)
    print(tf.test.is_gpu_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    unet = U_Net(args.weight)
    unet.tumor_train()