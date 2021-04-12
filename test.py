#导入包
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.resnet50 import  preprocess_input
import tensorflow.keras.backend as K

#加载模型
model = load_model(r'F:\AIresnet\model.h5')

# 载入图片
img = load_img(path=r'F:\AIresnet\nums\test\6\6_1.bmp', target_size=(224, 224))

#转换为numpy数组
x = img_to_array(img)   #（224,224,3）

#训练数组为4维（图片数量，224,224,3），扩充维度
x = K.expand_dims(x, axis=0)

#预处理
x = preprocess_input(x)

#数据预测
result = model.predict(x, steps=1)
print("result:", K.eval(K.argmax(result)))