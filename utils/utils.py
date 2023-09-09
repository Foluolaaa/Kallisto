#修改文件的后缀名并输出
def change_file_extension(filename, new_extension):
    # 检查文件名是否具有后缀
    if '.' not in filename:
        return filename + '.' + new_extension
    else:
        # 找到最后一个点的索引
        last_dot_index = filename.rfind('.')

        # 提取原始文件名和后缀
        file_without_extension = filename[:last_dot_index]
        current_extension = filename[last_dot_index + 1:]

        # 构建新的文件名
        new_filename = file_without_extension + '.' + new_extension

        return new_filename


#一个法条切分函数，可以按自然段进行切分
def split_law(texts):
    texts = texts.replace('(', '（').replace(')', '）')
    texts = texts.split('\n')
    i = 0
    output = []
    while texts:
        text = texts.pop(0)
        if not text.endswith('：'):
            output.append(text)
            continue

        while texts and texts[0].startswith('（'):
            xiang = texts.pop(0)
            output.append(text + '\n' + xiang)

    show = '输出=\n' + '\n'.join(output)
    print(show)

    return output