import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
x1=[1,2,3,4,5]
y1=[57.4,72.2,76.5,78.0,78.8]

x2=[1,2,3,4,5]
y2=[55.8,50.0,44.9,43.1,42.6]

x3=[1,2,3,4,5]
y3=[56.5,59.12,56.45,55.5,55.21]

x4=[1,2,3,4,5]
y4=[56.6,60.125,58.416,57.98,57.87]
# x=np.arange(20,350)
l1=plt.plot(x1,y1,'r--',label='Acc%')
l2=plt.plot(x2,y2,'g--',label='BLEU')
l3=plt.plot(x3,y3,'b--',label='H2')
plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
plt.title('Acc%, BLEU, H2')
new_ticks = np.linspace(1, 5, 5)
plt.xticks(new_ticks)
plt.xlabel('Editing steps',fontsize=15)
plt.ylabel('Metrics',fontsize=15)
plt.legend()
plt.show()