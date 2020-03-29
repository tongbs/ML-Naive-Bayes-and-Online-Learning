import math

def binary_count(line):
    num_1 = 0
    num_0 = 0
    for i in range(len(line)):
        if line[i] == '1':
            num_1 +=1
        else:
            num_0 += 1
    return num_1, num_0

def C(n,k):
    numerator = math.factorial(n)
    denominator = math.factorial(k)*math.factorial(n-k)
    ans = numerator/denominator
    return ans

def binomial(num_1, num_0):
    p = num_1/(num_1+num_0)
    pr = math.pow(p,num_1)*math.pow(1-p,num_0)*C(num_1+num_0,num_1)
    return pr


if __name__ == "__main__":
    a = int(input("a = "))
    b = int(input("b = "))

    f = open("testfile.txt","r")
    lines = f.readlines()

    for i , line in enumerate(lines):
        line = line.strip('\n')
        print(f'case {i}: {line}')
        num_1 , num_0 = binary_count(line)
        print(num_1,num_0)
        likelihood = binomial(num_1, num_0)
        print(f'Likelihood: {likelihood}')
        print(f'Beta prior:     a = {a} b = {b}')
        a+=num_1
        b+=num_0
        print(f'Beta posterior: a = {a} b = {b}')
        print()