from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import matplotlib.pyplot as plt

plt.scatter([0],[0],s=200,marker='*',color='C0')
plt.scatter([0],[1],s=200,marker='*',color='C1')
plt.scatter([1],[0],s=200,marker='>',color='C0')
plt.scatter([1],[1],s=200,marker='>',color='C1')

plt.show()
