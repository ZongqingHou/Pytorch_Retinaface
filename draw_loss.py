import matplotlib.pyplot as plt

file = open('./train_details/train_details_m68_5r3.txt')
# file = open('./train_details/0.txt')
data = file.readlines()
para_1 = []
para_2 = []
para_3 = []

for num in data:
    para_1.append(float(num.split(' ')[0]))
    para_2.append(float(num.split(' ')[1]))
    if float(num.split(' ')[2])<100:
        para_3.append(float(num.split(' ')[2]))

para_x=[]
for i in range(1,len(data)+1):
    para_x.append(int(i))

plt.plot(para_x,para_1,label='location')
plt.plot(para_x,para_2,label='classification')
plt.plot(para_3,label='landmarks')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=3, mode="expand", borderaxespad=0.)
plt.show()