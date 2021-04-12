from keras.models import Model
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D,ZeroPadding2D,GlobalAveragePooling2D
from keras.layers import *
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras import layers, optimizers, models
from keras.optimizers import Adam
from keras.layers import Activation
from keras.layers import Input
import tensorflow as tf
import keras.backend as K
import numpy as np
import math,os

#测试tensorflow-gpu安装成功
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
print(tf.__version__)
a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)
print('GPU:', tf.test.is_gpu_available())

#定义参数
BatchSize = 16

TRAIN_DIR='./nums/train'
VALID_DIR='./nums/test'

#训练集、验证集数量
num_train_samples = sum([len(files) for r1, d1, files in os.walk(TRAIN_DIR)])
num_valid_samples = sum([len(files) for r2, d2, files in os.walk(VALID_DIR)])

# 样本数 / batch_size
num_train_steps = math.floor(num_train_samples / BatchSize)
num_valid_steps = math.floor(num_valid_samples / BatchSize)

data = ImageDataGenerator(
    # 数据增强参数
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

#以文件夹路径为参数，生成经过数据增强/归一化后的数据，在一个无限循环中无限产生batch数据
train_generator = data.flow_from_directory(
    './nums/train',
    target_size=(224, 224),
    batch_size=BatchSize,
    shuffle=True,
    class_mode='categorical',)

validation_generator = data.flow_from_directory(
    './nums/test',
    target_size=(224, 224),
    batch_size=BatchSize,
    class_mode='categorical',)

print(validation_generator.class_indices)  # 输出对应的标签文件夹

print("训练集图片数量", BatchSize * len(train_generator))
print("验证集图片数量", BatchSize * len(validation_generator))

# ResNet50模型,
# def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', dilation_rate=1, name=None):
#     if name is not None:
#         bn_name = name + '_bn'
#         conv_name = name + '_conv'
#     else:
#         bn_name = None
#         conv_name = None
#
#     x = Conv2D(filters=nb_filter, kernel_size=kernel_size, padding=padding, strides=strides, name=conv_name)(x)
#     x = BatchNormalization(axis=3, name=bn_name)(x)
#     x = Activation('relu')(x)
#     return x
# # 残差层
# def Res_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
#     x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
#     # 原始resnet，中间是1个3*3卷积
#     x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
#     x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
#
#     # with_conv_shortcut 是否需要下采样，需要下采样要多经过一个卷积两个特征图才能统一大小
#     if with_conv_shortcut:
#         shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
#         x = add([x, shortcut])
#         return x
#     else:
#         x = add([x, inpt])
#         return x

# resnet主网络
# def creatcnn():
#     inpt = Input(shape=(224, 224, 3))
#     x = ZeroPadding2D((3, 3))(inpt)
#     x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
#
#     #
#     x = Res_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
#     x = Res_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
#     map56 = x
#
#     x = Res_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
#     x = Res_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
#     map28 = x
#
#     x = Res_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
#     x = Res_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
#     map14 = x
#
#     x = Res_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
#     x = Res_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
#     x = Res_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
#     map7 = x
#     '''
#     ######################多特征图融合##################################
#     x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
#
#     f56=Flatten()((map56))
#     f28=Flatten()((map28))
#     f14=Flatten()((map14))
#     f7=Flatten()((map7))
#     x = Concatenate()([f56,f28,f14,f7])
#     print(x.shape)
#     x = Dense(5, activation='softmax')(x)
#
#     '''
#     x = AveragePooling2D(pool_size=(7, 7))(x)
#     x = Flatten()(x)
#     # print(x.shape)
#     x = Dense(12, activation='softmax')(x)
#
#     model = Model(inputs=inpt, outputs=x)
#     return model
# # 优化器，动量0.9，学习率0.0001
# sgd = SGD(decay=0.0001, momentum=0.9)
# model = creatcnn()
# model.build((224,224))

'''迁移学习resnet50，建议直接用该模型，在进一步测试效果提升'''
#   调用ResNet50模型
Models = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#   建立分类识别模型
model = models.Sequential()     #模型序列化
model.add(Models)
model.add(layers.Flatten())     #将多维数据一维化，实现从卷积层到全连接层的过渡
model.add(layers.Dense(10,activation='softmax'))    #全连接层，10分类

Models.trainable = False          #冻结层，因为已经训练完毕，无需再次训练

# #自建模型
# model = models.Sequential()
#
# model.add(Convolution2D(25, (5, 5), input_shape=(28, 28, 3)))
# model.add(MaxPooling2D(2, 2))
# model.add(Activation('relu'))
# model.add(Convolution2D(50, (5, 5)))
# model.add(MaxPooling2D(2, 2))
# model.add(Activation('relu'))
# model.add(Flatten())
#
# model.add(Dense(50))
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(Activation('softmax'))
#输出模型各层参数
model.summary()

# 打印学习率值
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
optimizer = Adam(lr=1e-4)
lr_metric = get_lr_metric(optimizer)

'''
学习率调整
'''
# def scheduler(epoch):
#     # 每隔10个epoch，学习率减小为原来的1/10
#     if epoch % 10 == 0 and epoch != 0:
#         lr = K.get_value(model.optimizer.lr)
#         K.set_value(model.optimizer.lr, lr * 0.1)
#         print("lr changed to {}".format(lr * 0.1))
#     return K.get_value(model.optimizer.lr)
# lr_scheduler = LearningRateScheduler(scheduler)
#
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)
# callbacks = [lr_reducer, lr_scheduler]

#优化器、损失函数、准确率评测标准
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', lr_metric])

#训练
model.fit_generator(train_generator,
                    steps_per_epoch=num_train_steps,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=num_valid_steps,
                    # callbacks=callbacks
                    )
#保存模型
model.save('model.h5')