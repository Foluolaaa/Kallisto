import json
import re
import fullname
import abbreviation
import main

chinese_num = {
    '零': 0, '一': 1, '⼀': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,'⼋':8
}

chinese_unit = {
    '十': 10, '百': 100, '千': 1000, '⼗': 10
}


def chinese_to_num(chinese_str):
    # 将字符串翻转，百，十，千
    reversed_str = chinese_str[::-1]
    num = 0
    section = 1
    for char in reversed_str:
        if char in chinese_unit:
            # 获取单位数值 10,100，1000
            unit = chinese_unit[char]
            section = unit
        else:
            num += chinese_num[char] * section
    return num


def get_name(text):
    name_list = re.findall("《(.+?)》", text)
    # 没有提到一个法名
    if not name_list:
        return 2
    fullname_list = []
    for i in name_list:
        name = abbreviation.find_full_name_by_abbreviation(i)
        # 找不到该法条的文件
        if name != 'None':
            # print(i, "找到了")
            fullname_list.append(name)
        else:
            fullname_list.append("")
    if all(not x for x in fullname_list):
        return 1
    law = ''
    if len(fullname_list) != 1:
        parts = re.split(r"《.*?》", text)
        i = 0
        for part in parts[1:]:
            # law +=
            if fullname_list[i] != '':
                return get_law(fullname_list[i], part)
            i += 1

    return get_law(fullname_list[0], summary)


def get_law(name, text):
    # 对应 条
    pattern = r'第[^\s：、.项章第和至与个下条本等]{0,7}条'
    matches = re.findall(pattern, text)
    # print(matches)
    # 没有写明是第几条
    if not matches:
        return 1
    law = ''
    res = 0
    for i in range(len(matches)):
        if matches[i][1:-1].isdigit():
            match_list = [int(matches[i][1:-1])]
            # 重复引用前面的条文
            if i > 0 and int(matches[i][1:-1]) < res:
                continue
            res = int(matches[i][1:-1])
        else:
            # 重复引用前面的条文
            match_list = [int(chinese_to_num(matches[i][1:-1]))]
            if i > 0 and int(chinese_to_num(matches[i][1:-1])) < res:
                continue
            res = int(chinese_to_num(matches[i][1:-1]))
        # print(main.find_text_by_id(name, match_list))
        if fullname.find_text_by_id(name, match_list) != 3:
            return fullname.find_text_by_id(name, match_list)
        law += "《"+name+"》第"+str(match_list[0])+"条"
    return law


with open('中国法律服务网20200807.json', encoding='utf-8') as f:
    # data_list 是一个包含多个字典的列表
    data_list = json.load(f)
f.close()
'''
# 将每个字典转换为字符串，并用正则表达式替换所有不可见字符
for i in range(len(data_list)):
    # 将字典转换为一个字符串
    data_str = json.dumps(data_list[i])

    clean_str = re.sub(r"\s+ \s+", " ", data_str)
    # 将清理过的字符串转换回字典，并替换原来的元素
    data_list[i] = json.loads(clean_str)
'''
new_data = []
for data in data_list:

    # 将一个字典中的值提取出来
    value_lists = [str(value) for value in data.values()]

    # summary
    a = value_lists[3]
    pattern = r'中国法律服务网.+?：'
    match = re.search(pattern, a)
    if match is not None:
        start_index = match.end()
        # +1没有换行
        summary = a[start_index+1:-56]
    else:
        summary = '暂无'

    # content
    if value_lists[2] == '':
        content = '问题暂无'
    else:
        content = "有问题：" + value_lists[2]

        if summary == '暂无':
            content += '\n\n' + summary
        elif get_name(summary) == 2:
            content += "\n\n无法条：" + summary
        else:
            content += "\n\n有法条："
            if get_name(summary) != 1:
                content += get_name(summary)
        content += "\n\n请你结合法条，回答我的问题。你要先给出结论（或者某些特定条件下的结论），然后引用法律（需要清晰地指出是什么法第一条），告诉我对应的规定，来回答问题。"

    new_dict = {
        "content": content,
        "summary": summary
    }
    new_data.append(new_dict)

with open('result.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=0, ensure_ascii=False)
f.close()
