import time
import numpy as np
from heapq import nlargest
'''
思路1：将序列排序后再寻找第k小的数字，可以用到快速排序或归并排序。
思路2：使用快速排序的切分思想，将序列切为大于，等于，小于pivot的三个数组，分别叫L，M，R。如果k小于L的长度，
	那么在L里面递归；如果k大于L+M的长度，在R里面寻找第（k - len(L+M)）大的数字。其余情况就是等于pivot，直接return就好。

偶数个数列，中间两个取较小值
'''
def median_list(L, k):
	# print(L)

	if len(L) == 1:
		return L[0]

	pivot = L[0]
	left, right = 0, len(L)-1

	while left < right:
		while L[right] >= pivot and left < right:
			right -= 1
		L[left] = L[right]

		while L[left] <= pivot and left < right:
			left += 1
		L[right] = L[left] # 快排划分

	L[left] = pivot	#放置正确位置

	if k > left:
		return median_list(L[left+1:], k-left-1)
	elif k < left:
		return median_list(L[:left], k)
	else:
		return L[k]

# L = [10,1,14,33,12,5,17,8,22]
# 分治法
L = np.random.random(2708)
# print('L size:', len(L), 'median size:',(len(L)-1)//2)
start_time = time.time()
print(median_list(L, (len(L)-1)//2))
print('time:{:.6f}'.format(time.time() - start_time))
print('----------------')

#sort
start_time = time.time()
L.sort()
# print(L)
mid = (len(L)-1)//2
# print('mid:',mid)
print(L[mid])
# print(L[:mid])
# print('left size:',len(L[:mid]))
# print(L[mid + 1:])
# print('right size:',len(L[mid + 1:]))
print('time:{:.6f}'.format(time.time() - start_time))
print('----------------')

# np.median
start_time = time.time()
print(np.median(L))
print('time:{:.6f}'.format(time.time() - start_time))


L = [10,1,14,33,12,5,17,8,22]
L = np.array(L)
result = map(L.tolist().index, nlargest(7, L))
print(list(result))
