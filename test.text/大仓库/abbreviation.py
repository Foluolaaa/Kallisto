import os


def find_full_name_by_abbreviation(abbreviation):
    current_dir = os.path.dirname(os.path.abspath(__file__))+"/res"


# 遍历当前目录下的所有txt文件
    for file_name in os.listdir(current_dir):
        if file_name.endswith(".txt"):
            full_name = file_name.replace(".txt", "")

            # 使用子序列匹配判断是否匹配
            if is_subsequence(abbreviation, full_name):
                return full_name

    # print(abbreviation, "未找到匹配的全称")
    return 'None'


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

    return i == subseq_length
