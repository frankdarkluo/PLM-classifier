max_len=0
total_len=0

with open('./data/GYAFC/test.1') as of:
    datas=of.readlines()
    for data in datas:
        data=data.strip().split()
        total_len+=len(data)
        if len(data)>max_len:
            max_len=len(data)

print(max_len)
print(total_len/len(datas))
