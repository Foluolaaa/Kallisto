'''
本模块用于对结构化的数据进行转换，转换为pickle数据，会构造embedding
本模块会将数据以二进制格式的形式存储
'''

import json
import pickle
import numpy as np
from tqdm import tqdm
import os

from utils.gpt import embedding
from utils.utils import *

def convert_jsonl(filename):
    if os.path.exists(change_file_extension(filename, 'pickle')):
        print('file exist')
        return

    text_sum = 0
    dict = {}
    def id(text):
        nonlocal text_sum
        nonlocal dict

        if text in dict:
            return dict[text]
        dict[text] = text_sum
        text_sum += 1
        return text_sum - 1

    json_data = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f_in:
        for line in f_in:
            data = json.loads(line)

            data['name'] = id(data['name'])

            #对条中的每一行额外建模
            if data['structure'][-1] == '条':
                splits = split_law(data['texts'][-1])
                if len(splits) > 1:
                    data['index'] = []
                    for split in splits:
                        data['index'].append(id(split))

            # 处理texts字段
            texts = data['texts']
            for i in range(len(texts)):
                texts[i] = id(texts[i])

            data['texts'] = texts

            json_data.append(data)

    map_output = [''] * text_sum
    for key, value in dict.items():
        map_output[value] = key

    # 求解embedding数据
    embedding_list = []

    for key in tqdm(map_output):
        output = embedding(key)
        embedding_list.append(output)

    embedding_list = np.array(embedding_list).astype(np.float32)
    '''
    存储有三个数据段
    第一个数据段为常规的json数据段，除了所有的字符串都被搬走了以外
    第二个数据段为所有的字符串
    第三个字符串为embedding_list
    '''
    with open(change_file_extension(filename, 'pickle'), 'wb') as file:
        pickle.dump(json_data, file)
        pickle.dump(map_output, file)
        pickle.dump(embedding_list, file)

# 使用示例

directory_path='output-法律'
#file_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
file_paths = ['中华人民共和国专利法.txt']

for input_file in file_paths:
    print(input_file)
    convert_jsonl(input_file)

