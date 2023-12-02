from data import train_set, val_set, gesture_list
from kmeans import clusters, K
import copy
import math
from hmmlearn import hmm
import numpy as np

# def get_cluster_num(point):
#     # 使用k=1的KNN得到某个点的聚类编号
#     x = point[0]
#     y = point[1]
#     for cluster_num in range(len(clusters)):
#         for cluster_point in clusters[cluster_num]:
#             if cluster_point[0] == x and cluster_point[1] == y:
#                 return cluster_num
#     return K

def get_cluster_num(point):
    # 用k=1的KNN得到测试集每个点在Kmeans上的观测值
    x = point[0]
    y = point[1]
    distances_cluster_nums = []
    for clsuter_num in range(len(clusters)):
        for cluster_point in clusters[clsuter_num]:
            distance = math.sqrt(
                (cluster_point[0] - x) ** 2 +
                (cluster_point[1] - y) ** 2
            )
            distances_cluster_nums.append((distance, clsuter_num))
            
    min_distance_cluster_num = min(distances_cluster_nums, key=lambda dis: dis[0])
    return min_distance_cluster_num[1]

train_obs = {
    'a': [], 'e': [], 'i': [], 'o': [], 'u': [],
}

val_obs = {
    'a': [], 'e': [], 'i': [], 'o':[], 'u':[],
}

for gesture_name in gesture_list:
    for word in train_set[gesture_name]:
        word_obs = []
        for word_point in word:
            cluster_num = get_cluster_num(word_point)
            word_obs.append(cluster_num)
        train_obs[gesture_name].append(word_obs)


for gesture_name in gesture_list:
    for word in val_set[gesture_name]:
        word_obs = []
        for word_point in word:
            cluster_num = get_cluster_num(word_point)
            word_obs.append(cluster_num)
        val_obs[gesture_name].append(word_obs)


model = {}

# 隐藏状态数量
n_states = 3
# 观测状态数量（Kmeans的K值）
n_observations = K
for gesture_name in gesture_list:
    model[gesture_name] = hmm.CategoricalHMM(n_components=n_states, n_iter=100)
    model[gesture_name].n_features = n_observations
    observations = train_obs[gesture_name]
    observations = [np.array(obs).reshape(-1, 1) for obs in observations]
    lengths = [obs.shape[0] for obs in observations]
    # print(lengths)
    model[gesture_name].fit(np.concatenate(observations), lengths)


val_obs = {
    'a': [], 'e': [], 'i': [], 'o':[], 'u':[],
}

for gesture_name in gesture_list:
    for word in val_set[gesture_name]:
        word_obs = []
        for word_point in word:
            cluster_num = get_cluster_num(word_point)
            word_obs.append(cluster_num)
        val_obs[gesture_name].append(word_obs)

stat = {
    'a': [], 'e': [], 'i': [], 'o':[], 'u':[],
}

for gesture_name in gesture_list:
    for obs in val_obs[gesture_name]:
        obs_seq = np.array(obs).reshape(-1, 1)
        log_probs = []
        for model_gesture in model:
            log_prob = model[model_gesture].score(obs_seq)
            log_probs.append((log_prob, model_gesture))
            
        # 选出概率最大的模型
        max_prob_model_gesture = max(log_probs, key=lambda prob_tuple: prob_tuple[0])[1]
        stat[gesture_name].append(max_prob_model_gesture)

confusion_stat = {}
record_lengths = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0,} # 对于每一个字母测试用例的数量
for stat_key in stat:
    count = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0,}
    for record in stat[stat_key]:
        count[record] += 1
        record_lengths[stat_key] += 1
    confusion_stat[stat_key] = count
    
    
confusion_matrix = copy.deepcopy(confusion_stat)
for vowel_true in confusion_matrix:
    for vowel_infer in confusion_matrix[vowel_true]:
        confusion_matrix[vowel_true][vowel_infer] /= record_lengths[vowel_true]


# 画混淆矩阵
print(f'聚类数量为{K}，隐藏状态数量为{n_states}')
print('\ta\te\ti\to\tu')
for vowel_true in gesture_list:
    print(vowel_true + '\t', end='')
    for vowel_infer in gesture_list:
        print(confusion_matrix[vowel_true][vowel_infer], '\t', end='')
    print()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # colors = ['b', 'g', 'r', 'c', 'm']
    # plt.figure(figsize=(4, 4))
    # for k in range(5):
    #     for point in clusters[k]:
    #         plt.scatter(point[0], point[1], c=colors[k])
    # plt.show()

