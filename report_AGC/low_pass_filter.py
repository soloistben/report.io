import numpy as np
import matplotlib.pyplot as plt

'''low-pass'''
# x = np.arange(0.0, 2.0, 0.01)
# y1 = 1-0.5*x
# y2 = (1-0.5*x)**2
# y3 = (1-0.5*x)**3
# y4 = (1-0.5*x)**4
# y5 = (1-0.5*x)**5
# y6 = (1-0.5*x)**6
# plt.title('low-pass filter y=(1-1/2x)^k')
# plt.plot(x, y1, color='red', label='k=1')
# plt.plot(x, y2, color='blue', label='k=2')
# plt.plot(x, y3, color='green', label='k=3')
# plt.plot(x, y4, color='gray', label='k=4')
# plt.plot(x, y5, color='yellow', label='k=5')
# plt.plot(x, y6, color='skyblue', label='k=6')
# plt.legend() 
# plt.show()

'''high-pass'''
# x = np.arange(0.0, 2.0, 0.01)
# y1 = 0.5*x
# y2 = (0.5*x)**2
# y3 = (0.5*x)**3
# y4 = (0.5*x)**4
# y5 = (0.5*x)**5
# y6 = (0.5*x)**6
# plt.title('low-pass filter y=(1/2x)^k')
# plt.plot(x, y1, color='red', label='k=1')
# plt.plot(x, y2, color='blue', label='k=2')
# plt.plot(x, y3, color='green', label='k=3')
# plt.plot(x, y4, color='gray', label='k=4')
# plt.plot(x, y5, color='yellow', label='k=5')
# plt.plot(x, y6, color='skyblue', label='k=6')
# plt.legend() 
# plt.show()

'''median-pass
y = -t*x(x-2)
yk = k^2*y

f(x) = -t*x(x-2)

G = Uf(x)U^-1 = U(-tV(V-2E))U^-1 = -tU(V^2-2V)U^-1 = -t(L^2-2L)
Gk = k^2*G
'''
# x = np.arange(0.0, 2.0, 0.01)
# y1 = -1/16*x*(x-2)
# y2 = 2**2 *y1
# y3 = 3**2 *y1
# y4 = 4**2 *y1
# plt.plot(x, y1, color='red', label='k=1')
# plt.plot(x, y2, color='blue', label='k=2')
# plt.plot(x, y3, color='green', label='k=3')
# plt.plot(x, y4, color='gray', label='k=4')
# plt.show()
