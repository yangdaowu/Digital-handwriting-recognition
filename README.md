1.Installation

首先通过命令行创建一个anaconda虚拟环境
conda create -n your_name python=3.8

打开pycharm，添加虚拟环境。随后进入Terminal（Alt+F12）

cd requirements.txt所在文件夹内
输入pip install -r requirements.txt安装相应包

2.Dataset
数据集设置，nums文件夹下分为train和test两个文件夹，
每个文件夹下按照类别分子文件夹。
train和test数据划分比为7:3，手动划分

3.Run
安装相应环境和包后运行main.py，过程中会加载ResNet模型预权重，等待即可
test.py为测试单张图片的预测效果