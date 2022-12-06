import pickle
import numpy as np
import cv2
import os
import logging

# 打开cifar-10数据集文件目录
# 
def unpickle(batch):
    with open("cifar-10-batches-py/data_batch_" + str(batch), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


logger = logging.getLogger()
logger.setLevel(logging.INFO)

label_name = ['airplane', 'automobile', 'brid', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_workers = 5  # total num of workers

# create folder
for i in range(num_workers):
    if not os.path.exists('file2/data/cifar-10-pictures-' + str(i)):
        os.mkdir('file2/data/cifar-10-pictures-' + str(i))
        for j in range(10):
            os.mkdir('file2/data/cifar-10-pictures-' +
                    str(i) + '/' + str(label_name[j]))
if not os.path.exists('file2/data/cifar-10-pictures-test'):
    os.mkdir('file2/data/cifar-10-pictures-test')
    for j in range(10):
        os.mkdir('file2/data/cifar-10-pictures-test/' + str(label_name[j]))

order = [0] * 10
num_pics_one = 5000 // num_workers

# download train-batch and open
for batch in range(1, 6):
    data_batch = unpickle(batch)
    # print(data_batch)

    cifar_label = data_batch[b'labels']
    cifar_data = data_batch[b'data']

    # 把字典的值转成array格式，方便操作
    cifar_label = np.array(cifar_label)
    # print(cifar_label.shape)
    cifar_data = np.array(cifar_data)
    # print(cifar_data.shape)

    for i in range(10000):
        image = cifar_data[i]
        image = image.reshape(-1, 1024)
        r = image[0, :].reshape(32, 32)  # 红色分量
        g = image[1, :].reshape(32, 32)  # 绿色分量
        b = image[2, :].reshape(32, 32)  # 蓝色分量
        img = np.zeros((32, 32, 3))
        #RGB还原成彩色图像
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        j = order[cifar_label[i]] // num_pics_one
        cv2.imwrite("file2/data/cifar-10-pictures-" + str(j) + "/" + str(label_name[cifar_label[i]]) + "/" +
                    str(label_name[cifar_label[i]]) + "_" + str(order[cifar_label[i]]) + ".jpg", img)
        order[cifar_label[i]] += 1

# download test-batch and open
with open("cifar-10-batches-py/test_batch", 'rb') as fo:
    data_batch = pickle.load(fo, encoding='bytes')
# print(data_batch)

cifar_label = data_batch[b'labels']
cifar_data = data_batch[b'data']

# 把字典的值转成array格式，方便操作
cifar_label = np.array(cifar_label)
# print(cifar_label.shape)
cifar_data = np.array(cifar_data)
# print(cifar_data.shape)

for i in range(10000):
    image = cifar_data[i]
    image = image.reshape(-1, 1024)
    r = image[0, :].reshape(32, 32)  # 红色分量
    g = image[1, :].reshape(32, 32)  # 绿色分量
    b = image[2, :].reshape(32, 32)  # 蓝色分量
    img = np.zeros((32, 32, 3))
    #RGB还原成彩色图像
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    cv2.imwrite("file2/data/cifar-10-pictures-test/" + str(label_name[cifar_label[i]]) + "/" +
                str(label_name[cifar_label[i]]) + "_" + str(i) + ".jpg", img)

