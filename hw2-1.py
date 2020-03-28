import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result*256+int(b)
    return result


def image_load(file):
    f = gzip.open(file,'rb')
    file_content = f.read()
    file_list = list(file_content)
    #[0:4] magic number, [4:8]# of images, [8:12]# of rows, [12:16] # of cols
    num_img = bytes_to_int(file_list[4:8])
    num_row = bytes_to_int(file_list[8:12])
    num_col = bytes_to_int(file_list[12:16])
    img = np.array(file_list[16:]).reshape(num_img,num_row*num_col)
    return img

def label_load(file):
    f  = gzip.open(file,'rb')
    file_content = f.read()
    file_list = list(file_content)
    #[0:4]magic number, [4:8]# of labels, [8:]one word one label
    num_label = bytes_to_int(file_list[4:8])
    label = np.array(file_list[8:]).reshape(num_label,1)
    return label

def train_load(toggle_option, bin_tr_image, tr_label):
    lut = np.ones((10,784,32))
    p_num = []  #each number(0~9) probability
    for i in range(10):
        p_num.append(0.0)
    #for i in range(1):
    #    print(bin_tr_image[i])
    for i in range(bin_tr_image.shape[0]):   #60000
        l_num = tr_label[i][0]
        p_num[l_num] += 1
        for j in range(bin_tr_image.shape[1]):   #784
            bin_num = bin_tr_image[i][j]
            #print(l_num,j,bin_num)
            lut[l_num,j,bin_num] += 1
    #print(p_num)
    total = sum(p_num)
    for i in range(10):
        p_num[i] = p_num[i]/total
    #print(p_num)
    for i in range(10):
        for j in range(784):
            lut[i,j] = lut[i,j] / np.sum(lut[i,j])
    return lut , p_num




if __name__ == "__main__":
    toggle_option = int(input("0:Discrete mode, 1:Continuous mode: "))
    if toggle_option == 0 or toggle_option == 1:
        tr_image = image_load("train-images-idx3-ubyte.gz")
        tr_label = label_load("train-labels-idx1-ubyte.gz")
        bin_tr_image = tr_image//8
        lut = train_load(toggle_option, bin_tr_image, tr_label)

        for i in range(10):
            for j in range(784):
                lut[i,j] = lut[i,j]/np.sum(lut[i,j])
        #print(tr_label)
        

        '''
        for i in range(1):
            print(tr_label[i][0])
            #print(tr_image[i].shape)
            a = tr_image[i]//8
            print(a)
            #img_array = np.array(tr_image[i]).reshape(28,28)
            #img = Image.fromarray(img_array)
            #plt.imshow(img_array, cmap='gray')
            #plt.show()
        '''
