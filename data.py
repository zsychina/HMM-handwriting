import xml.etree.ElementTree as ET

def get_one_dataset(path):
    tree = ET.parse(path)
    root = tree.getroot()
    word_dataset = []
    
    for training_example in root.findall('trainingExample'):
        example_data = []
        for coord in training_example.findall('coord'):
            x = float(coord.get('x'))
            y = float(coord.get('y'))
            t = int(coord.get('t'))
            example_data.append((x, y, t))
            
        word_dataset.append(example_data)
    return word_dataset

gesture_list = ['a', 'e', 'i', 'o', 'u']
dataset = {}

for gesture_name in gesture_list:
    dataset[gesture_name] = get_one_dataset(f'./project1-data/{gesture_name}.xml')
    
train_set = {}
val_set = {}

for key_gesture_name in dataset:
    sample_num = len(dataset[key_gesture_name])
    # 奇数训练，偶数测试
    train_set[key_gesture_name] = [dataset[key_gesture_name][i] for i in range(sample_num) if i % 2 == 1]
    val_set[key_gesture_name] = [dataset[key_gesture_name][i] for i in range(sample_num) if i % 2 == 0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']

    plt.figure(figsize=(4, 4))

    for color, gesture_name in zip(colors, gesture_list):
        for sample_i in range(len(train_set[gesture_name])):
            for point in train_set[gesture_name][sample_i]:
                ax.scatter(point[0], point[1], color=color, label=gesture_name)

    # for color, gesture_name in zip(colors, gesture_list):
    #     for point in train_set[gesture_name][0]:
    #         ax.scatter(point[0], point[1], color=color, label=gesture_name)

    plt.show()