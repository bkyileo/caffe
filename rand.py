import os
import numpy as np
from matplotlib import pyplot as plt
import shutil
import random
from PIL import Image
# for cnt in range(0,34):
#     dirpath = 'newdata/original/' + str(cnt)
#     valname = 'newdata/train/' + str(cnt)
#     os.mkdir(valname)
#     for fname in os.listdir(dirpath):
#         print(cnt , '->' , dirpath+'/'+fname,'->',valname)
#         img= Image.open(dirpath+'/'+fname)
#         img_gray = img.convert("L")
#         print(img.mode,'->',img_gray.mode)
#         img_gray.save(valname+'/'+fname)
# def ResizeImage(filein, fileout, width, height,type):
#     img = Image.open(filein)
#     out = img.resize((width, height), Image.ANTIALIAS)
#     img_gray = out.convert("L")
#     if(img_gray.mode=='RGB'):
#         print("Have RGB")
#     img_gray.save(fileout)
# for cnt in range(0,34):
#     dirpath = 'newdata/train_gray/' + str(cnt)
#     width = 28
#     height = 28
#     valname = 'newdata/train/' + str(cnt)
#     os.mkdir(valname)
#     for fname in os.listdir(dirpath):
#         print(cnt , '->' , dirpath+'/'+fname,'->',valname)
#         ResizeImage(dirpath+'/'+fname,valname+'/'+fname,width,height,'jpg')

# for cnt in range(0,34):
#     dirpath = 'newdata/train/' + str(cnt)
#     numfile=0
#     for fname in os.listdir(dirpath):
#         newfilename = str(numfile) + '.jpg'
#         numfile = numfile + 1
#         print(cnt , '->' , numfile)
#         os.rename(os.path.join(dirpath,fname), os.path.join(dirpath, newfilename))

for cnt in range(0,34):
    valname = 'newDigitWord/' + str(cnt)
    os.mkdir(valname)

# txt = open('newdata/train.txt', 'w')
# for cnt in range(0,34):
#     filename = 'newdata/train/' + str(cnt)
#     imgfile = GetFileList(filename)
#     for img in imgfile:
#         st = img + '\t' + str(cnt) + '\n'
#         txt.writelines(st)
#txt = open('newdata/val.txt', 'w')
#for cnt in range(0,34):
#    filename = 'newdata/val/' + str(cnt)
#    imgfile = GetFileList(filename)
#    for img in imgfile:
#        st = img + '\t' + str(cnt) + '\n'
#        txt.writelines(st)


