

if __name__ == "__main__":
    a = int(input("a = "))
    b = int(input("b = "))

    f = open("testfile.txt","r")
    lines = f.readlines()

    for line in lines:
        print(line)
