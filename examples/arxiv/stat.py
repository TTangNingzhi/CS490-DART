num_1 = 0
num_0 = 0
with open('../avazu/train/train.csv') as f:  # ../criteo/train.txt, ../avazu/train/train.csv
    cnt = 0
    while True:
        line = f.readline()
        if not line:
            break
        cnt += 1
        print(cnt, '/40428967', end='\r')  # 45840617, 40428967
        if line[0] == '1':
            num_1 += 1
        else:
            num_0 += 1

print(num_1, num_0)
