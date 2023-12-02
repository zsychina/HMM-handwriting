from data import train_set, gesture_list

import random
import math
import numpy as np
import copy

K = 5

train_points = []

# # 训练集是所有数据
for gesture_name in gesture_list:
    for sample_i in range(len(train_set[gesture_name])):
        for point in train_set[gesture_name][sample_i]:
            train_points.append([point[0]] + [point[1]] + [gesture_name])

# 训练集是每个字母第一例数据
# for gesture_name in gesture_list:
#     for point in train_set[gesture_name][0]:
#         train_points.append([point[0]] + [point[1]] + [gesture_name])

mean_vectors = random.sample(train_points, K)

round = 0
for i in range(1000):
    clusters = [[] for _ in range(K)] 

    for j in range(len(train_points)):
        distances = []
        for i in range(K):
            distance = math.sqrt(
                (train_points[j][0] - mean_vectors[i][0]) ** 2 +
                (train_points[j][1] - mean_vectors[i][1]) ** 2
            )
            distances.append(distance)
        closet_cluster = np.argmin(distances)
        clusters[closet_cluster].append(train_points[j])

    old_mean_vectors = copy.deepcopy(mean_vectors)
    for i in range(K):
        if len(clusters[i]) > 0:
            new_mean_vector = [0 for _ in range(2)]
            x_sum = 0
            y_sum = 0
            for cluster_i_point in clusters[i]:
                x_sum += cluster_i_point[0]
                y_sum += cluster_i_point[1]
            new_mean_vector[0] = x_sum / len(clusters[i])
            new_mean_vector[1] = y_sum / len(clusters[i])
            if new_mean_vector[0] != mean_vectors[i][0] or new_mean_vector[0] != mean_vectors[i][0]:
                mean_vectors[i] = new_mean_vector

    changed = False
    for i in range(len(mean_vectors)):
        if mean_vectors[i][0] != old_mean_vectors[i][0] or \
            mean_vectors[i][1] != old_mean_vectors[i][1]:
            changed = True
         
    round += 1
    if not changed:
        print(f'ended at round {round}')
        break
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    colors = ['b', 'g', 'r', 'c', 'm']
    plt.figure(figsize=(4, 4))
    for k in range(K):
        for point in clusters[k]:
            plt.scatter(point[0], point[1], c=colors[k])
    plt.show()
    