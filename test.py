import math
import numpy as np
import matplotlib.pyplot as plt
r=1
k=200
x = np.arange(201, 1000, 1)
ref = np.ones((len(x)))
ref2 = np.ones((len(x)))
nn = 2
alp_list = [0/nn,0.1/nn,0.35/nn,0.7/nn,0.9/nn]
# alp_list = [0.5,0.5,0.5,0.5,0.5]
can_list = [len(x)/5,len(x)*2/5,len(x)*3/5,len(x)*4/5,len(x)*5/5]
for j in range(0,len(x)):
    for i,kkk in enumerate(can_list):
        if j<kkk:
            ref2[j] = alp_list[i]
            break
        else:
            continue
for i in x :
    for n in range(k+1,i+1):
        ref[i-k-1] *= (n*np.exp(ref2[n-k-1])-1)/n/np.exp(ref2[n-k-1])

y = k / x * np.exp(-ref2)
# z=np.exp(-ref2*(x-k))*ref
stand = k/x
# 创建figure对象
plt.figure(figsize=(14,10))

# 设定为用1个图表表示
ax = plt.subplot(1, 1, 1)

#设定各方程式的线性和标记，标签，plot
##线形图
line_width = 4  # 线的宽度为2像素
plt.plot(x, y, marker='o', markersize=1, label='Samples in buffer',color='#2FA12F', linewidth=line_width )
plt.plot(x, ref, marker='o', markersize=1, label='Samples in the data stream',color='#FF8E2A', linewidth=line_width)
plt.plot(x, stand, marker='o', markersize=1, label='Baseline',color='#1F77B4', linewidth=line_width)
# 画虚线
line_color = 'red'  # 线的颜色为红色
xx = [x+200 for x in can_list]  # 竖线的x坐标
for xxx in xx :
    plt.axvline(xxx, color=line_color, linewidth=line_width, linestyle='dashed')
#显示图例
plt.legend()
#显示网格线
ax.yaxis.grid(True, linewidth=line_width)
ax.spines['bottom'].set_linewidth(line_width)
ax.spines['left'].set_linewidth(line_width)
# 设置图例样式
legend = ax.legend(loc='upper right', fontsize=24)
plt.xlabel('Samples', fontsize=24)
plt.ylabel('Probability of being stored in the buffer', fontsize=24)
# 设置坐标轴数据样式
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
plt.savefig('figure/figure.jpg', quality=100)
#显示图表
plt.show()

print(y)

# print(np.exp(-0.041))
# print(np.exp(-0.015))