import tensorflow as tf

# tf.enable_eager_execution()
import loader
import config
import matplotlib.pyplot as plt
import numpy as np
from model import auto
import math
import random
import model_use

# img=watermark.read_visibleWatermark(64,64)
# # img = np.resize(img, [1, 64, 64, 3])
# # img=tf.convert_to_tensor(img)
# img=1-img
#     # print(target)
# img=np.array(img[0])
# plt.imshow(img)
# plt.show()
# print(img)
# np.savetxt("1.txt",img[:,:,0])


# conf=config.ConfigLoader(".\\app\\a\\config.txt")

# load=loader.DataLoader(".\\app\\a\\data\\test",True)
# data=load.load(config_loader=conf)

# batch=tf.zeros(shape=)
# for input_image, ground_truth in data:
#     print(input_image)

#
#
# count=0
# for input_image, target in data:
#     print(target)
#     mark=target[:,0:128,0:128,:]
#     # print(target)
#     img=np.array(mark[0]*0.5+0.5)
#     img[img>1]=1
#     img[img<0]=0
#     plt.imshow(img)
#     plt.show()


# for in1, in2, out in data.take(1):
#     x=[in1[0],in2[0],out[0]]
#     for i in range(3):
#         plt.subplot(1, 3, i + 1)
#         # getting the pixel values between [0, 1] to plot it.
#         drawable = np.array(x[i] * 0.5 + 0.5)
#         drawable[drawable < 0] = 0
#         drawable[drawable > 1] = 1
#
#         # plt.imshow(display_list[i] * 0.5 + 0.5)
#         plt.imshow(drawable)
#         plt.axis('off')
#     plt.show()

# def fun(**kwargs):
#     for key in kwargs:
#         print(key)
#
# fun(a=1,b=2,c=3)
# from extern import watermark_tool
# image = tf.read_file("./data/rio_w/train/1.jpg")
# image = tf.image.decode_jpeg(image)
# image=image[:,:,0:3]
# image=np.array(image)
# print(image.shape)
# w=watermark_tool.decode(image[:,0:128,:])
# plt.imshow(w)
# plt.show()
#
# import train_autoencoder
#
# img,b = train_autoencoder.load_mnist("./data/mnist/mnist.npy", train_num=1000, test_num=50)
# print(img)
# plt.imshow(list[3][0])
# plt.show()
# a="1234"
# print("67" in a)

# import train_autoencoder_key
# #
# a=train_autoencoder_key.create_key()
# print(a)
# import train_tool

# im=plt.imread("1.png")
# im=(im[:,:,0]+im[:,:,1]+im[:,:,2])/3

# x=np.zeros([32,32,3])
# x[:,:,0]=im
# x[:,:,1]=im
# x[:,:,2]=im

# x[x>0.5]=1
# x[x<=0.5]=0
# plt.ion()
# plt.figure(1)
# plt.imshow(x)
# # plt.show()
# plt.draw()

 
 

# # y=[1,2,3,4,5]
# def invoke_method_of_class(class_instance, method_name, params):
#     # get method
#     method = getattr(A, method_name)
#     # invoke
#     method(class_instance, params)
   
# class A:
#     def b(self,params):
#         print(params)

# a=A()
# m=getattr(a,"b")
# m({1:1})
# # invoke_method_of_class(a,"b",{1:1})


# a=np.loadtxt("1.txt")
# # b=np.loadtxt("2.txt")

# model = model_use.EncoderDecoder("./trained_models/auto_mnist")

# model.decoder()

# a=tf.convert_to_tensor(np.array([1,0,2,180]))
# b=tf.convert_to_tensor(np.array([1,230,2,180]))
# a=tf.cast(a>127.5, dtype=tf.int8)
# b=tf.cast(b>127.5, dtype=tf.int8)
# print(a-b)

# a=np.random.normal(0, scale=0.5, size=[128,128,3])

# # a=np.reshape(a,[128*128*3])
# # plt.hist(a)
# # plt.show()
# im=plt.imread("1.png")[:,:,0:3]
# plt.figure(1)
# plt.imshow(im)

# plt.figure(2)
# im=im+a
# im[im<0]=0
# im[im>1]=1
# plt.imshow(im)
# plt.show()

# print(a)

# for i in range(0,100):
#     print("\r你的输出详情:{}".format(i), end='', flush=True)

# from evaluate import evaluate, eval_all
# a=plt.imread("3.png")[:,:,0:3]*2-1
# b=plt.imread("2.png")[:,:,0:3]*2-1

# # plt.imshow(a)
# # plt.show()

# # plt.imshow(b)
# # plt.show()


# x=tf.reshape(a, shape=[1,32,32,3])

# a[0,0,0]=1
# y=tf.reshape(a, shape=[1,32,32,3])

# print(y)

# print()


# image1 = a*0.5+0.5
# image2 = b*0.5+0.5

# image1 = np.array(image1)
# image2 = np.array(image2)
# # filter
# image1[image1 >= 0.5] = 1
# image1[image1 < 0.5] = 0
# image2[image2 >= 0.5] = 1
# image2[image2 < 0.5] = 0
# ber
# e = np.sum(np.abs(image1-image2))
# e = np.mean(np.abs(image1-image2))
# print(e)


# e = evaluate(x, y, psnr_enable=False, ssim_enable=False, ber_enable=True)
# # 0.000514450553
# print(e)




# np.savetxt("1.txt",a[:,:,0],fmt="%.2f")
# np.savetxt("2.txt",b[:,:,0], fmt="%.2f")
# print(r)
# from evaluate import eval_all
# wm_results, wm_report = eval_all(data_path="./data/flower/train",
#                           model_path="./trained_models/flower_unet_l1_bw",
#                           watermark_path="./watermark/wm_binary.png",visual_result_dir="./results",
#                           watermark_binary=True, decode_path="./trained_models/auto_mnist")

# print(wm_report)
# print(wm_results["value_list"])

# plt.imshow(x)
# plt.show()

# plt.imshow(y)
# plt.show()



# from train_autoencoder_key import create_key

# # create_key()

# m=model_use.EncoderDecoder("./experiments/key_sensitive/models/train1", key_enable=True)

# d=m.decode_from_image(encode_image_path="./experiments/key_sensitive/1980_EN.png", secret_key_path="./experiments/key_sensitive/key.npy")

# plt.imshow(d)
# plt.show()



# a=np.ones([1,128,128,3])
# a[:,:,0:20,:]=-1
# plt.imshow(a)
# plt.show()

print(None in ("True","true"))

