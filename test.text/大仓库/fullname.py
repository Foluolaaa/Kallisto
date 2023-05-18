import json


def find_text_by_id(name, ids):
    # 获取文件路径
    file_path = name + ".txt"
    try:
        # 打开文件并读取内容，将每行json line数据转换为json格式
        with open(file_path, "r", encoding="utf-8") as file:
            data = []
            for line in file:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue

            # 根据ids的长度构造条款号字符串
            clause_number = ""
            if len(ids) == 1:
                clause_number += "第{}条".format(ids[0])
            if len(ids) == 2:
                clause_number += "第{}章第{}条".format(ids[0], ids[1])
            if len(ids) == 3:
                clause_number += "第{}章第{}节第{}条".format(ids[0], ids[1], ids[2])

            # 查找与ids相等的条款号，并输出内容
            for item in data:
                if item["ids"] == ids:
                    texts = item["texts"][-1]
                    print("{}{}：{}".format(name, clause_number, texts))
                    return "{}{}：{}".format(name, clause_number, texts)

            # print("未找到对应的内容")
            return 3

    except FileNotFoundError:
        print("文件不存在")
    except Exception as e:
        print("发生错误：", str(e))


'''
name = "人民检察院检察建议工作规定"
ids = [1, 2]
find_text_by_id(name, ids)
'''
