import sys
import statistics

data1=float(sys.argv[1])
data2=float(sys.argv[2])
# reverse_ppl=1/float(sys.argv[3])

data_test=[data1,data2]

def geometric_mean(data):  # 计算几何平均数
    total=1
    for i in data:
        total*=i #等同于total=total*i
    return pow(total,1/len(data))

print(geometric_mean(data_test))
#print(statistics.harmonic_mean(data_test))