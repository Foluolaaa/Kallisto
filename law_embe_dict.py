'''
本函数为法条存储综合存储函数
提供法条存储，语义查询，法条简称-全称查询，给定法名和编号输出法全称的查询，给问题和法条实现法条掺杂的函数
本函数计划进行扩展，以支持对书本的查询
相比上一版本的law_embe_dict,本函数不依赖GPT，在简称全程查询和embedding上，均采用GLM服务
本版本实现了更高的存储密度，同数据量下空间消耗压缩至原先的五分之一，初始化速度提升五十倍，查询速度提升十倍
'''

import json
import math
import random

from tqdm import tqdm
import heapq
from time import sleep
import os
import time
import numpy as np
import pickle
from utils.glm import *


class law_embe_dict:
    def __init__(self, file_names):
        self.length = 0  # 总长度
        self.structures = []  # 存储json的原始结构，注意str全部变成了id
        self.texts = []  # 存储每一条被切分后的文字，含切片后的原文，原文索引
        self.embeddings = []  # 存储全部的embedding，含切片后的原文，原文索引的embedding
        self.names_dict = {}  # 存储全部的法律名字，其中第i次出现的法律名字标记编号为i
        self.law_article_id = []  # 存储第i部法的第j条在structures中的位置
        if hasattr(file_names, 'str'):
            file_names = [file_names]
        for file_name in tqdm(file_names):
            with open(file_name, 'rb') as f:
                structures = pickle.load(f)
                texts = pickle.load(f)
                embeddings = pickle.load(f)

                self.texts += texts

                for structure in structures:
                    # 给每部法分配一个名字
                    law_name = texts[structure['name']]
                    if law_name not in self.names_dict:
                        key = len(self.names_dict)
                        self.names_dict[law_name] = key
                        self.law_article_id.append([0])

                    structure['name'] += self.length

                    structure['texts'] = [id + self.length for id in structure['texts']]

                    if 'index' in structure:
                        structure['index'] = [id + self.length for id in structure['index']]
                        #需要给index构建一个反向的编号
                    self.structures.append(structure)

                    # 由于条是连续的，所以可以这么赋值
                    if structure['structure'][-1] == '条':
                        self.law_article_id[self.names_dict[law_name]].append(structure['texts'][-1])

                self.length += len(embeddings)

                self.embeddings.append(embeddings)

        self.embeddings = np.concatenate(self.embeddings, axis=0)

    # 输入某个法的简称，输出所有包含了这个法的全称
    # 本模采用子序列法匹配，即简称必须是全称的子序列
    # 本函数仅对get_law_fullname提供服务，不对外提供
    def __get_law_fullname_list(self, law_name):
        # 判断subseq是否是seq的子序列，如果是则返回True
        def is_subsequence(subseq, seq):
            subseq_length = len(subseq)
            seq_length = len(seq)
            if subseq_length > seq_length:
                return False

            i = 0
            j = 0
            while i < subseq_length and j < seq_length:
                if subseq[i] == seq[j]:
                    i += 1
                j += 1

            return i >= subseq_length

        answer_list = []

        # 遍历所有名字
        for full_name in self.names_dict.keys():

            # 使用子序列匹配判断是否匹配
            if is_subsequence(law_name, full_name):
                answer_list.append(full_name)
            else:
                # 部分的xx解释会被命名为xx规定，特此打一个补丁。
                if '解释' in law_name and '规定' in full_name:
                    full_name_modify = full_name.replace('规定', '解释')

                    if is_subsequence(law_name, full_name_modify):
                        answer_list.append(full_name)

        # 在很多找法的时候，会吐出很多的司法解释，但我们没有要找这些材料，可以进行一波过滤

        if '解释' not in law_name:
            return_list = []
            for fullname in answer_list:
                if '解释' not in fullname and '规定' not in fullname:
                    return_list.append(fullname)
            answer_list = return_list

        return answer_list

    # 输入一个简称，输出与这个简称最为接近的全称
    def get_law_fullname(self, law_name):
        # 输入是一个字符串，输出是这个字符串，去掉全部的标点符号的串（包括中文的标点符号和英文的标点符号），并返回完成去除的串
        def remove_punctuation(text):
            import re
            import string
            # 去除英文标点符号
            text = text.translate(str.maketrans("", "", string.punctuation))

            # 去除中文标点符号
            text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)

            return text

        law_name = remove_punctuation(law_name)

        # 行政诉讼法专属补丁
        if '行诉法' in law_name:
            law_name = law_name.replace('行诉法', '行政诉讼法')

        # 民诉法也要打补丁
        if '民诉法' in law_name:
            law_name = law_name.replace('民诉法', '民事诉讼法')

        # 规定和解释经常会混淆，打一个补丁
        if '规定' in law_name:
            law_name = law_name.replace('规定', '解释')

        law_name_list = self.__get_law_fullname_list(law_name)

        # 如果只筛出一个或者没筛出，那么就直接返回即可，无需用GPT筛选
        if len(law_name_list) <= 1:
            if law_name_list:
                return law_name_list[0]
            return None

        # 构造查询的prompt，这个prompt与最接近法条的prompt一致，经测试二者可以复用
        def get_nearest_fullname(problem, laws):
            problem = '有问题:' + problem + '\n\n下面是法条:\n'
            output = problem + ' 第1条\n\n'.join(laws + ['请你找出与该问题最有联系的一条法条并输出。你输出法的名字和第几条即可'])
            return output

        # 用GPT从这若干个解释中，挑选出最接近的解释
        for trytimes in range(5):
            prompt = get_nearest_fullname(law_name, law_name_list)
            #print(prompt)
            answer = chat(prompt)
            #取出真正的答案段
            answer = answer.split(' ')[0]

            #GLM有时候会输出些别的，得判一判
            if answer in law_name_list:
                return answer

            #如果没找到，就random_shuffle一次后再来，最多试5次
            random.shuffle(law_name_list)

        # 如果什么都没找到就输出None
        return None

    # 输入法的名称和id，返回法的全文
    def get_article_by_name_and_id(self, name, id, output_name_and_id = False):
        name = str(name)
        origin_id = id
        id = int(id)
        if name in self.names_dict:
            key = self.names_dict[name]
            id = self.law_article_id[key][id]
            if output_name_and_id:
                return name + ' 第' + str(origin_id) + '条 ' + self.texts[id]
            return self.texts[id]
        return None

    # 整条法的搜索器，搜索与text最为接近的前top条法并输出
    # output_prob为True时会输出余弦相似度
    # output_prefix为True时会输出全部的前缀
    # prefix_index 可以调整前缀和后面部分的权重
    def get_full_from_embedding(self, text, top=5, output_prob = False, output_prefix = False, prefix_index = 0.1, min_name_sim = 0.7):

        # 计算每一条法的相似度，本相似度考虑了法名，章名，以及每一条的情况
        def calc_sim(line):
            sim_name = cos_sims[line['name']]
            sim_sum = sim_name
            sum_cnt = 1

            penalty = False

            # 名字相似度惩罚
            if sim_name < min_name_sim:
                penalty = True

            for i in line['texts'][:-1]:
                sim_sum += cos_sims[i]
                sum_cnt += 1

                if cos_sims[i] > min_name_sim + 0.05:
                    penalty = False

            sim_sum /= sum_cnt

            sim_max = cos_sims[line['texts'][-1]]
            if 'index' in line:
                for i in line['index']:
                    sim_max = max(sim_max, cos_sims[i])
                    if cos_sims[i] > 0.8:
                        penalty = False

            if penalty:
                sim_sum = 0

            return sim_sum * prefix_index + sim_max * (1 - prefix_index)

        start_time = time.time()
        target_embedding = np.array(embedding(text)).astype(np.float32)
        print("embedding耗时=", time.time() - start_time)
        law_heap = []
        start_time = time.time()

        cos_sims = self.embeddings.dot(target_embedding).tolist()

        end_time = time.time()
        print("搜索耗时=", end_time - start_time)

        for i, structure in enumerate(self.structures):
            if '条' not in structure['structure']:
                continue

            # 屏蔽掉奇奇怪怪的超长输出，比如最后附件下面的表格，或者野生动物清单等
            if len(self.texts[structure['texts'][-1]])>1024:
                continue

            cos_sim = calc_sim(structure)
            heapq.heappush(law_heap, (cos_sim, i))
            if len(law_heap) > top:
                heapq.heappop(law_heap)
        law_heap = sorted(law_heap, reverse=True)
        print("排序耗时=", time.time() - end_time)

        sim_laws = []
        for sim, i in law_heap:
            law_article = self.texts[self.structures[i]['texts'][-1]]

            if output_prefix:
                law_article = ' '.join([self.texts[id] for id in self.structures[i]['texts']])

            law_name = self.texts[self.structures[i]['name']]
            law_id = self.structures[i]['ids'][-1]
            output = law_name + ' 第' + str(law_id) + '条 ' + law_article
            sim_laws.append((sim, output))

        for sim, law in sim_laws:
            print(sim, law)

        if not output_prob:
            sim_laws = [law for sim, law in sim_laws]
        return sim_laws

    # 法部分段落的搜索器，搜索与text最为接近的前top款法并输出
    # 只考虑每一款本身的相似度
    def get_line_from_embedding(self, text, top=5, output_prob = False, clean = False):
        if clean:
            top *= 3

        # 计算每一条法的相似度，本相似度考虑了法名，章名，以及每一条的情况
        def get_structure_score(line):
            sim_sum = cos_sims[line['name']]
            sum_cnt = 1

            for i in line['texts'][:-1]:
                sim_sum += cos_sims[i]
                sum_cnt += 1

            sim_sum /= sum_cnt

            return sim_sum

        start_time = time.time()
        target_embedding = np.array(embedding(text)).astype(np.float32)
        print("embedding耗时=", time.time() - start_time)
        law_heap = []
        start_time = time.time()

        cos_sims = self.embeddings.dot(target_embedding).tolist()

        end_time = time.time()
        print("搜索耗时=", end_time - start_time)

        for i, structure in enumerate(self.structures):
            if '条' not in structure['structure']:
                continue

            #屏蔽掉奇奇怪怪的输出，比如最后附件下面的表格，或者野生动物清单等
            if len(self.texts[structure['texts'][-1]]) > 1024 or (('index' in structure) and (len(structure['index']) > 15) ):
                continue

            structure_score = get_structure_score(structure)

            articles = [structure['texts'][-1]]
            if 'index' in structure:
                articles += structure['index']

            for j, idx in enumerate(articles):
                cos_sim = structure_score * 0.1 + cos_sims[idx] * 0.9
                heapq.heappush(law_heap, (cos_sim, i, j)) #加入堆中的是相似度，第几条，第几款/项
                if len(law_heap) > top:
                    heapq.heappop(law_heap)
        law_heap = sorted(law_heap, reverse=True)
        print("排序耗时=", time.time() - end_time)

        sim_laws = []
        for sim, i, j in law_heap:
            law_article = self.texts[self.structures[i]['texts'][-1]]
            if 'index' in self.structures[i] and j>0:
                law_article = self.texts[self.structures[i]['index'][j-1]]
            law_name = self.texts[self.structures[i]['name']]
            law_id = self.structures[i]['ids'][-1]
            output = law_name + ' 第' + str(law_id) + '.' + str(j) + '条 ' + law_article
            sim_laws.append((sim, output))

        #这是一个去重策略，如果后面出现了法条的全文，则会在前面把单列的部分全部删除
        if clean:
            sum = 0
            answer = []
            for sim, law in sim_laws:
                id = law[law.find(' 第') + 2: law.find('条 ')]
                id_tiao, id_kuan = id.split('.')
                id_tiao = law[:law.find('第') + 1] + id_tiao + '.'

                if id_kuan != '0':
                    id_tiao += '0'
                    #如果以前就出现了.0，则直接跳过
                    for sim2, law2 in answer:
                        if id_tiao in law2:
                            break
                    else:
                        answer.append((sim, law))
                    continue

                #如果出现了整条的法条，就开始执行去重工作
                already_replace = False
                for i in range(len(answer)):
                    if id_tiao in answer[i][1]:
                        if not already_replace:
                            answer[i] = (answer[i][0], law)
                            already_replace = True
                        else:
                            answer[i] = (answer[i][0], '$delete')

                answer = [(sim2, law2) for sim2, law2 in answer if law2 != '$delete']

                if not already_replace:
                    answer.append((sim, law))

                if len(answer) * 3 == top:
                    break

            sim_laws = answer

        for sim, law in sim_laws:
            print(sim, law)

        if not output_prob:
            sim_laws = [law for sim, law in sim_laws]
        return sim_laws

    # 根据prompt，从embedding中，找出最为接近的top_input条法律，然后在这些法条中筛选出top_output条最为接近的法条输出
    def get_top_laws(self, prompt, top_input, top_output):
        chat_cache = {}

        # 构造查询的prompt
        def closest_law(problem, laws):
            #print(laws)
            problem = '有问题:' + problem + '\n\n下面是法条:\n'
            output = problem + '\n\n'.join(laws + ['请你找出与该问题最有联系的一条法条并输出。你输出法的名字和第几条即可。'])

            #这一版本的代码，支持了各个细分条，但模型又没有重训，所以做了一些修改
            output = output.replace('.0条 ', '条 ')
            return output

        # 根据prompt，从列表laws中，找出与prompt最为接近的法条，并输出全文
        def get_top_law(prompt, laws):
            if len(laws) == 1:
                return laws[0]

            # GLM的Token不建议传输太多，整体Token限制为1024
            if len(closest_law(prompt, laws)) < 1800:
                origin_prompt = prompt
                prompt = closest_law(prompt, laws)

                if prompt in chat_cache:
                    return chat_cache[prompt]

                # print(prompt)
                for trytime in range(5):
                    law_id = chat(prompt.replace('錕', '').replace('\n'*4, '\n'*2))
                    for law in laws:
                        if law.startswith(law_id[:-1]):
                            print('输出=', law_id)
                            chat_cache[prompt] = law
                            return law

                    if prompt in chat_cache:
                        chat_cache.pop(prompt)

                    #如果失败了，就执行random_shuffle进行随机扰动，然后再次尝试
                    print('err=', law_id)
                    random.shuffle(laws)
                    prompt = closest_law(origin_prompt, laws)
                return None

            law_group = [[]]

            for law in laws:
                if len(closest_law(prompt, law_group[-1] + [law])) < 1800:
                    law_group[-1].append(law)
                else:
                    law_group.append([law])

            law_answers = [get_top_law(prompt, group) for group in law_group]
            law_answers = [law for law in law_answers if law is not None]
            return get_top_law(prompt, law_answers)

        #laws = self.get_line_from_embedding(prompt, top_input, output_prob=False, clean=True)
        laws = self.get_full_from_embedding(prompt, top_input, output_prob=False)
        answers = []
        for i in range(top_output):
            print('step', i)
            law = get_top_law(prompt, laws)
            # 为了让其余的prompt尽可能不变，这里的删除结构比较特别
            laws[laws.index(law)] = '錕' * len(law)
            answers.append(law)

        # print(answers)
        return answers

    '''
    输入prompt
    exist_laws_and_id_list表示相关法的名字和第几条，仅支持字符串列表
    输出若干条法条，且进行随机打乱
    法条的总长度不超过max_length
    最多掺入max_add条法律
    format表示格式，有"law"(纯法条),"name"(法条名和ID),"full"(法名+ID+纯法条）
    random_shuffle 表示是否要随机打乱，默认为True
    everyone表示，对每个法条
    '''

    def mix_unused_laws(self, prompt, exist_laws_and_ids_list=[], max_length=1536, max_add=8, format="full",
                        random_shuffle=True, everyone=False):
        answer_list = list(exist_laws_and_ids_list)
        exist_laws_len = len(exist_laws_and_ids_list)
        if max_add > 0:
            laws = self.get_full_from_embedding(prompt, exist_laws_len + max_add)
            for law in laws:
                name_and_id = ' '.join(law.split(' ')[:2])
                if name_and_id not in exist_laws_and_ids_list:
                    max_add -= 1
                    answer_list.append(name_and_id)
                    if max_add <= 0:
                        break

        # 读取法条原文
        if format in ["law", "full"]:
            answer_list = [self.get_article_by_name_and_id(answer.split(' ')[0], answer.split(' ')[1][1:-1], output_name_and_id=True) for
                           answer in answer_list]

        if everyone:
            new_answer_list = []
            for i in range(exist_laws_len):
                new_answer_list.append([answer_list[i]] + answer_list[exist_laws_len:])
            answer_list = new_answer_list
        else:
            answer_list = [answer_list]

        for i in range(len(answer_list)):
            # 排除长度过长的情况
            while len(''.join(answer_list[i])) > max_length:
                answer_list[i].pop()

            # 进行随机打乱
            if random_shuffle:
                random.shuffle(answer_list[i])

        if not everyone:
            answer_list = answer_list[0]
        return answer_list

    # 输入一篇文档，文档按照句号和逗号切分，对于每一句话和相邻的话，判断是否是法条的原文。
    # 如果不是法条原文，则原样输出
    # 如果是原文，则会变为法条的名字和编号
    # 这个模块准确率有待提高，可以试着用一下
    def identify_laws_in_article(self, article):

        # 用动态规划算法求最长公共子序列
        def longest_common_subsequence_length(str1, str2):
            m = len(str1)
            n = len(str2)

            # 创建一个二维数组来保存子问题的解
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            # 填充数组dp，通过自底向上的方式计算最长公共子序列的长度
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            # 返回最长公共子序列的长度
            return dp[m][n]

        # 动态规划算法求最长公共子串
        def longest_common_substring_length(str1, str2):
            m = len(str1)
            n = len(str2)

            # 创建一个二维数组来保存子问题的解
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            # 用一个变量来记录最长公共子串的长度
            max_length = 0

            # 填充数组dp，通过自底向上的方式计算最长公共子串的长度
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        max_length = max(max_length, dp[i][j])

            # 返回最长公共子串的长度
            return max_length

        # 根据求出的参数，判断这段文本是否是法条
        # 其中S表示最长公共子串长度
        # Q表示最长公共子序列长度
        # len表示待比较字符串的长度
        def check_law(S, Q, len):
            if Q / len < 0.75:
                return False

            if S / len < 0.5 and Q / len < 0.9 and len - Q > 3:
                return False

            return True

        '''
        判断输入的文本，是否是法条中的原文
        采用的判断标准：
        1，短句子排除（长度小于10的排除）
        2，最长公共子串长度小于全文长度一半的排除
        3，打分低于0.75的排除
        '''

        def is_text_law(sentence):
            # 输入是一个字符串，输出是这个字符串，去掉全部的标点符号的串（包括中文的标点符号和英文的标点符号），并返回完成去除的串
            def remove_punctuation(text):
                import re
                import string
                # 去除英文标点符号
                text = text.translate(str.maketrans("", "", string.punctuation))

                # 去除中文标点符号
                text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)

                return text

            # 在去除标点符号的基础上，把空格和换行全部去掉
            def remove_punctuation_and_spaces(text):
                text = remove_punctuation(text)
                text = text.replace(' ', '').replace('\n', '').replace('\t', '').replace('\u3000', '')
                return text

            # 把字符串里面所有的阿拉伯数字，都转化为汉字的数字
            def convert_arabic_to_chinese(text):
                from cn2an import an2cn
                result = ""
                current_number = ""
                for char in text:
                    if char.isdigit():
                        current_number += char
                    else:
                        if current_number != "":
                            chinese_number = an2cn(current_number)
                            if char == '%':
                                result += '百分之'
                                result += chinese_number
                            else:
                                result += chinese_number
                                result += char
                            current_number = ""
                        else:
                            result += char
                if current_number != "":
                    chinese_number = an2cn(current_number)
                    result += chinese_number
                return result

            sentence = remove_punctuation_and_spaces(convert_arabic_to_chinese(sentence))
            # print(text)
            if len(sentence) < 20:
                return [False, 0, 0]

            #from modules.get_top_laws_from_embedding import get_top_laws_from_embedding
            laws = self.get_full_from_embedding(sentence, 10, output_prob=False, prefix_index=0.0, min_name_sim=0.5)
            #laws = self.get_top_laws_from_embedding(sentence, 10)
            for a in laws:
                law_and_id = ' '.join(a.split(' ')[:2])
                a = remove_punctuation_and_spaces(convert_arabic_to_chinese(a))
                # print(a)
                # print(len(text))
                ans_subs = longest_common_substring_length(sentence, a)
                ans_subq = longest_common_subsequence_length(sentence, a)
                # print(f'subs={ans_subs},subq={ans_subq},len={len(sentence)}')
                # is_law = (ans_subs / len(sentence) > 0.5 or ans_subq / len(sentence) > 0.9) and (ans_subq / len(sentence) > 0.75)
                is_law = check_law(ans_subs, ans_subq, len(sentence))
                if is_law:
                    is_law = law_and_id
                    return [is_law, ans_subs, ans_subq]
            return [False, 0, 0]

        # 去除多余的换行
        article = article.replace('\n\n', '\n')
        # 先根据句号进行切分
        sentences = article.replace('?', '。').replace('？', '。').split('。')[:-1]
        print(sentences)

        output, i = [], 0
        output_law = []
        while i < len(sentences):
            info, last_subs, last_subq = is_text_law(sentences[i])
            i += 1
            if not info:
                output.append(sentences[i - 1])
                print(sentences[i - 1])
                continue

            law_info = info
            sentence = sentences[i - 1]
            while i < len(sentences):
                if len(sentences[i]) > 20:
                    now_info, now_subs, now_subq = is_text_law(sentences[i])
                    if not check_law(now_subs, now_subq, len(sentences[i])):
                        break

                info, subs, subq = is_text_law(sentence + sentences[i])
                deltaQ = (subq - last_subq) / len(sentences[i])
                if deltaQ < 0.75:
                    break
                sentence += sentences[i]
                i += 1

            print(sentence)
            print(law_info)
            output_law.append(law_info)

        return output_law