from __future__ import print_function
from __future__ import division

"""
定义
"""
import os
import cv2
print(cv2.__version__)

dir_path='D:/DeepLearning/Machine_Learn/SVM/datasets/'
datasets='image'
datasets_2='image128'

walk=os.walk(dir_path+datasets)

print(walk)
for root,dir,file in walk:
    for name in file:
        f=open(os.path.join(root,name),'r',encoding='utf-8',errors='ignore')
    raw=f.read()
    print(raw)



image_width=100
image_height=128