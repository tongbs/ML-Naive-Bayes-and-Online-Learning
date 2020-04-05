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
    ##[0:4] magic number, [4:8]# of images, [8:12]# of rows, [12:16] # of cols
    num_img = bytes_to_int(file_list[4:8])
    num_row = bytes_to_int(file_list[8:12])
    num_col = bytes_to_int(file_list[12:16])
    img = np.array(file_list[16:]).reshape((num_img,num_row*num_col))
    return img

def label_load(file):
    f  = gzip.open(file,'rb')
    file_content = f.read()
    file_list = list(file_content)
    ##[0:4]magic number, [4:8]# of labels, [8:]one word one label
    num_label = bytes_to_int(file_list[4:8])
    label = np.array(file_list[8:]).reshape((num_label,1))
    return label

def train_load(toggle_option, tr_image, tr_label):
    if toggle_option == 0:
        bin_tr_image = tr_image//8
        lut = np.ones((10,784,32))
        p_num = []  ##each number(0~9) probability
        for i in range(10):
            p_num.append(0.0)

        for i in range(bin_tr_image.shape[0]):   ##60000
            l_num = tr_label[i][0]
            p_num[l_num] += 1
            for j in range(bin_tr_image.shape[1]):   ##784
                bin_num = bin_tr_image[i][j]
                ##print(l_num,j,bin_num)
                lut[l_num,j,bin_num] += 1
        total = sum(p_num)
        for i in range(10):
            p_num[i] = p_num[i]/total
        p_num = np.array(p_num)
        for i in range(10):
            for j in range(784):
                lut[i,j] = lut[i,j] / np.sum(lut[i,j])
        return lut, p_num
    else:  ##continuous
        data = []
        lut = []
        for i in range(10):
            data.append([])
            lut.append([])
            for j in range(784):
                data[i].append([])
                lut[i].append({'std':0.0,'mean':0.0})
        lut = np.array(lut)
        p_num = []
        for i in range(10):
            p_num.append(0.0)
        
        for i in range(tr_image.shape[0]):
            l_num = tr_label[i][0]
            p_num[l_num] += 1
            for j in range(tr_image.shape[1]):
                bit = tr_image[i][j]
                data[l_num][j].append(bit)
        total = sum(p_num)
        for i in range(10):
            p_num[i] = p_num[i]/total
        p_num = np.array(p_num)
        for i in range(10):
            for j in range(784):
                lut[i][j]['std'] = np.std(data[i][j])
                lut[i][j]['mean'] = np.mean(data[i][j])
        return lut, p_num 
                
def log_likelihood(toggle_option, j, k, pixel, lut):
    if toggle_option == 0:
        return math.log(lut[j,k,pixel])
    else:
        ##print("continuous")
        std = lut[j][k]['std']
        if std == 0:
            std = 40
        mean = lut[j][k]['mean']
        ##print('mean',mean)
        ##try:
        ##    likelihood = math.log(1/math.sqrt(2*math.pi*(std**2))) + (-((pixel-mean)**2)/(2*std**2))
        ##except ZeroDivisionError:
        ##    likelihood = 0
        likelihood = math.log(1/math.sqrt(2*math.pi*(std**2))) + (-((pixel-mean)**2)/(2*std**2))
        return likelihood
        
def error_rate(result_list):
    correct = 0
    wrong = 0
    for i in range(len(result_list)):
        if result_list[i] == 1:
            correct+=1
        else:
            wrong+=1
    error_rate = wrong/(wrong+correct)
    return error_rate

def bayes_imagination_number(toggle_option, lut):
    for i in range(10):
        print(f'{i}:')
        for j in range(784):
            if toggle_option == 0:
                ##ex. number = 7,have 784 pixel, find the maximum value in 32 bins(在該pixel上出現最多次的bit是哪個值)
                bit = np.argmax(lut[i,j])*8
            else:
                bit = lut[i][j]['mean']
            if bit >= 128:
                print(1,end=' ')
            else:
                print(0, end=' ')
            if (j+1)%28 == 0:
                print()
        print()    

##p_num = P(C)
def test(toggle_option, ts_image, ts_label, lut, p_num):
    if toggle_option == 0:
        ts_image //= 8
    result_list = []

    for i in range(ts_image.shape[0]):  #10000
        prob_img = []   ##each test image for classify to [0~9] probability
        for j in range(10):
            log_p_num = math.log(p_num[j])
            ##print(log_p_num) #prior
            for k in range(784):
                log_p_num = log_p_num + log_likelihood(toggle_option, j, k, ts_image[i,k], lut)
            prob_img.append(log_p_num)
        prob_img = np.array(prob_img)
        prob_img = prob_img / np.sum(prob_img)
        ##print(ts_label[i][0])

        for j in range(10):
            print(f'{j}: Posterior (in log scale):{prob_img[j]}')
        min_index = np.argmin(prob_img)
        Prediction = min_index
        Ans = ts_label[i][0]
        print(f'Prediction: {Prediction}, Ans: {Ans}')
        print()
        if Prediction == Ans:
            result_list.append(1)
        else:
            result_list.append(0)
    ##print out imagination of number in Bayes classifier
    bayes_imagination_number(toggle_option, lut)
    ##calculate error rate
    error = error_rate(result_list)
    print(f'Error rate: {error}')

        

if __name__ == "__main__":
    toggle_option = int(input("0:Discrete mode, 1:Continuous mode: "))
    if toggle_option == 0 or toggle_option == 1:
        ##tr:train ts:test
        tr_image = image_load("train-images-idx3-ubyte.gz")
        tr_label = label_load("train-labels-idx1-ubyte.gz")
        ts_image = image_load("t10k-images-idx3-ubyte.gz")
        ts_label = label_load("t10k-labels-idx1-ubyte.gz")
        lut , p_num = train_load(toggle_option, tr_image, tr_label)
        test(toggle_option, ts_image, ts_label, lut, p_num)
