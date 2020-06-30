from ulits.__init__ import dir_path
from ulits.__init__ import image_height,image_width
from ulits.__init__ import datasets
from ulits.__init__ import datasets_2


import os
from PIL import Image

def fixed_size(filePath,savePath):
    im = Image.open(filePath)
    out = im.resize((image_width, image_height), Image.ANTIALIAS)
    out.save(savePath)


def changeSize():
    filePath = r'D:\DeepLearning\Machine_Learn\SVM\datasets\image'
    destPath = r'D:\DeepLearning\Machine_Learn\SVM\datasets\image128_1'
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file[-1]=='g':
                fixed_size(os.path.join(filePath, file), os.path.join(destPath, file))
    print('Done')

if __name__ == '__main__':
    changeSize()
