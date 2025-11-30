

# import subprocess

# for i in range(1, 4):
#     for j in range(1, 7):
#         number = i*j
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={i-1} python data_making.py -n {number}', shell=True)


import subprocess
import time

processes = []

list = [i for i in range(1, 19)]

for idx, number in enumerate(list):
    if number % 3 == 0:
        cuda = 0 
    if number % 3 == 1:
        cuda = 1 
    if number % 3 == 2:
        cuda = 2 
    command = f'CUDA_VISIBLE_DEVICES={cuda} python data_making.py -n {number}'
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

# 等待所有进程完成
for process in processes:
    process.wait()

print("所有任务完成！")