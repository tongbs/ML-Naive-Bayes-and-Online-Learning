import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result*256+int(b)
    return int(result)


def image_load(file):
    f = gzip.open(file,'rb')
    file_content = f.read()
    file_list = list(file_content)
    #[0:4] magic number, [4:8]# of images, [8:12]# of rows, [12:16] # of cols
    num_img = bytes_to_int(file_list[4:8])
    num_row = bytes_to_int(file_list[8:12])
    num_col = bytes_to_int(file_list[12:16])
    img = np.array(file_list[16:]).reshape((num_img,num_row*num_col))
    return img

def label_load(file):
    f  = gzip.open(file,'rb')
    file_content = f.read()
    file_list = list(file_content)
    #[0:4]magic number, [4:8]# of labels, [8:]one word one label
    num_label = bytes_to_int(file_list[4:8])
    label = np.array(file_list[8:]).reshape((num_label,1))
    return label

def train_load(toggle_option, tr_image, tr_label):
    if toggle_option == 0:
        bin_tr_image = tr_image//8
        lut = np.ones((10,784,32))
        p_num = []  #each number(0~9) probability
        for i in range(10):
            p_num.append(0.0)

        for i in range(bin_tr_image.shape[0]):   #60000
            l_num = tr_label[i][0]
            p_num[l_num] += 1
            for j in range(bin_tr_image.shape[1]):   #784
                bin_num = bin_tr_image[i][j]
                #print(l_num,j,bin_num)
                lut[l_num,j,bin_num] += 1
        total = sum(p_num)
        for i in range(10):
            p_num[i] = p_num[i]/10
        p_num = np.array(p_num)
        for i in range(10):
            for j in range(784):
                lut[i,j] = lut[i,j] / np.sum(lut[i,j])
        return lut , p_num

def log_likelihood(toggle_option, j, k, pixel, lut):
    if toggle_option == 0:
        return math.log(lut[j,k,pixel])
    else:
        print("continuous")

#p_num = P(C)
def test(toggle_option, ts_image, ts_label, lut, p_num):
    if toggle_option == 0:
        ts_image //= 8
    for i in range(10):
    #for i in range(ts_image.shape[0]):  #10000
        prob_img = []   #each test image for classify to [0~9] probability
        for j in range(10):
            log_p_num = math.log(p_num[j])
            #print(log_p_num) #prior
            for k in range(784):
                log_p_num = log_p_num + log_likelihood(toggle_option, j, k, ts_image[i,k], lut)
            prob_img.append(log_p_num)
        prob_img = np.array(prob_img)
        prob_img = prob_img / np.sum(prob_img)
        print(ts_label[i][0])
        for j in range(10):
            print(f'{j}: posterior:{prob_img[j]}')

        

if __name__ == "__main__":
    toggle_option = int(input("0:Discrete mode, 1:Continuous mode: "))
    if toggle_option == 0 or toggle_option == 1:
        #tr:train ts:test
        tr_image = image_load("train-images-idx3-ubyte.gz")
        tr_label = label_load("train-labels-idx1-ubyte.gz")
        ts_image = image_load("t10k-images-idx3-ubyte.gz")
        ts_label = label_load("t10k-labels-idx1-ubyte.gz")
        lut , p_num = train_load(toggle_option, tr_image, tr_label)
        test(toggle_option, ts_image, ts_label, lut, p_num)


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
