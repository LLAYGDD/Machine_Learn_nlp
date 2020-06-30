from __future__ import  print_function
from __future__ import division

"""
基本定义
"""
import os

dir_data='D:/DeepLearning/Machine_Learn/DataSets/'
datasets='input_LDA'
if not os.listdir(dir_data):
    try:
        print("文件不存在！")
    except:
        print("文件存在，请继续执行！")
        pass




