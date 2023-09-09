import os
import json
import numpy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils.gpt import embedding


def solve(dic):
    content = dic["content"]
    # content带有”“
    json_str = json.dumps(content, ensure_ascii=False)
    vector = embedding(json_str)
    dic["embedding"] = vector
    return dic


def process_multithread_add_embedding(num):
    directory_path = "D:\\_MyCode\\python\\实验组\\5\\dataToDo"
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            # 拼接文件夹和文件名得到完整的文件路径
            print(filename)
            file_path = os.path.join(foldername, filename)
            print('solving:', file_path)
            embedding_dict = []
            with open(file_path, "r", encoding='utf-8', errors='error')as file:
                data = json.load(file)
            with ThreadPoolExecutor(max_workers=num) as executor:
                futures = [executor.submit(solve, dic) for dic in data]
                for i, future in enumerate(tqdm(futures, desc="Processing")):
                    embedding_dict.append(future.result())

            directory = "D:\\_MyCode\\python\\实验组\\5\\embedding_data_1"
            output_filename = os.path.join(directory, 'embedding_' + filename)

            # 将更新后的JSON对象写入到一个新的文件中
            with open(output_filename, "w", encoding='utf-8') as file:
                json.dump(embedding_dict, file, ensure_ascii=False, indent=4)


def print_content(file_path, indice):
    with open(file_path, "r", encoding='utf-8', errors='error') as file:
        dict_list = json.load(file)
    for index, item in enumerate(dict_list):
        if index == indice:
            print(dict_list[index]["title"], dict_list[index]["content"])


def get_sim_indices(file_path, threshold):
    with open(file_path, "r", encoding='utf-8', errors='error') as file:
        data = json.load(file)

    tem_list = []
    tem_list2 = []
    vector_list = []

    for data_dict in data:
        vector_list.append(data_dict["embedding"])
        tem_list.append(data_dict["title"])
        tem_list2.append(data_dict["content"])

    indices_to_remove = []

    for i, vector1 in enumerate(vector_list):
        for j, vector2 in enumerate(vector_list[i+1:], start=i+1):
            sim2 = numpy.dot(embedding(tem_list2[i]), embedding(tem_list2[j]))
            similarity = numpy.dot(vector1, vector2)
            if similarity > threshold:
                indices_to_remove.append(j)
                print_content(file_path, i)
                print_content(file_path, j)
                print(similarity, '\n')

    # 转换成集合去重，确保保存的索引不重复
    indices_to_remove = list(set(indices_to_remove))
    # 移除相似度高的向量
    print("相似度大的索引：", indices_to_remove)
    return indices_to_remove


def del_sim(filename, file_path, indices_to_remove):
    with open(file_path, "r", encoding='utf-8', errors='error') as file:
        dict_list = json.load(file)
    # 使用列表解析和 enumerate() 来创建一个新的字典列表，不包含要删除的索引位置的字典
    new_dict_list = [item for index, item in enumerate(dict_list) if index not in indices_to_remove]

    directory = "D:\\_MyCode\\python\\实验组\\5\\去重后"
    output_path = os.path.join(directory, 'deduplicate_' + filename)
    with open(output_path, "w", encoding='utf-8') as file:
        json.dump(new_dict_list, file, ensure_ascii=False, indent=4)


def process_deduplication(threshold):
    directory_path = "D:\\_MyCode\\python\\实验组\\5\\embedding_data"
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            embedding_file_path = os.path.join(directory_path, filename)
            indices_to_remove = get_sim_indices(embedding_file_path, threshold)
            del_sim(filename, embedding_file_path, indices_to_remove)


process_multithread_add_embedding(20)
# 设置相似度阈值，超过该阈值的文本将被删除
# threshold = 0.98
# process_deduplication(threshold)
# file_path = "D:\\_MyCode\\python\\实验组\\副本\\deduplicate_embedding_0.json"
# get_sim_indices(file_path, 0.8)
