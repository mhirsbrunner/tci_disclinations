import time
import numpy as np
import cupy as cp

loop = 0

cpu = []
gpu = []

while loop < 10:

	a = np.random.randint(10, size=(2000, 2000))
	a = (a + a.conj().T) / 2

	t=time.time()
	np.linalg.inv(a)
	cpu.append(time.time()-t)

	a=cp.array(a)
	t=time.time()
	cp.linalg.inv(a)
	gpu.append(time.time()-t)
	
	loop = loop + 1

print(f'cpu: {cpu}')
print(f'gpu: {gpu}')

cpu = np.sum(np.array(cpu)[1:])
gpu = np.sum(np.array(gpu)[1:])


print(f'cpu/gpu: {cpu/gpu}')
