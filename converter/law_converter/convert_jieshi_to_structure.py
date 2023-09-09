#本模块可以实现将司法解释转化为结构化文本
#本模块转化数据来源为北大法宝txt格式文件，其他来源的文件应该也可以
#北大法宝txt格式存在bug，部分小标题会跟在末尾的位置

import json, re, copy, os

import cn2an

def extractTitleAndContents(lines):
    begin_idx, end_idx, title_idx = -1, -1, -1
    title_pattern = re.compile(r"法|条例|解释|办法|规则|规定")
    title = ""

    for idx, line in enumerate(lines):
        line = re.sub(r'[\s\u3000]+', '', line)
        if (begin_idx == -1):
            if (title_pattern.findall(line)):
                if len(title) == 0 and len(line)<60:
                    title = line
                    title_idx = idx
            elif (line == "目录"):
                begin_idx = idx + 1
        else:
            if (line == ""):
                end_idx = idx
                break

    if end_idx == -1:
        for idx, line in enumerate(lines):
            if line.strip().startswith('第一') or line.strip().startswith('一、'):
                end_idx = idx - 1
                break

    return title, lines[end_idx + 1:]

hanzi_list = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千']

#编章节条核心控制函数
def next_id(structure, texts, ids, text):
    cn_num = ''
    if text[0] == '附':
        if len(text) < 5 and structure[-1]=='条':
            return structure, texts, ids, 2

    if text[0] != '第':
        #如果是以(二）开头的司法解释，要可以进行处理
        if len(text) > 4 and text[0] in ['(','（'] and text[1] in hanzi_list and (('。' not in texts[-1] and '：' not in texts[-1]) or texts[-1].rfind('。') > texts[-1].rfind('：') ) and len(text) < 20 and '。' not in text and '；' not in text:
            text = '第' + text[1:2] + '节' + text[3:]
        else:
            for cn_idx in range(1, len(text)):
                if text[cn_idx]  not in hanzi_list:
                    if text[cn_idx] == '、' and text[0] in hanzi_list : #地方条例屏蔽这里
                        text = '第' + text[:cn_idx] + '编' + text[cn_idx+1:]
                        break
                    else:
                        try:
                            texts[-1] += '\n' + text
                        except:
                            print('qipa1')
                        return structure, texts, ids, False

    for cn_idx in range(1, len(text)):
        if text[cn_idx] in [' ', '\t', '\u3000']:
            continue
        if text[cn_idx] in hanzi_list:
            cn_num += text[cn_idx]
        else:
            break
    else:
        texts[-1] += '\n' + text
        return structure, texts, ids, False

    try:
        an_num = cn2an.cn2an(cn_num)
    except:
        print(cn_num)
    hirerachy = text[cn_idx]
    text = text[cn_idx + 1:]

    if hirerachy == '分':
        hirerachy += text[0]
        text = text[1:]

    text = text.strip()
    text = re.sub(r'[\s\u3000]+', '', text)

    rk = ['编', '分编', '章', '节', '条']

    #本处可以判断之一之二
    if (hirerachy not in rk) or (hirerachy == '条' and text[:1] == '之'):
    #if (hirerachy not in rk):
        try:
            texts[-1] += '\n' + text
        except:
            print('qipa2')
        return structure, texts, ids, False


    while structure and rk.index(structure[-1]) >= rk.index(hirerachy):
        structure.pop()
        texts.pop()
        ids.pop()
    structure.append(hirerachy)
    texts.append(text)
    ids.append(an_num)
    return structure, texts, ids, True

def parseLaw(lines):
    law_name, contents = extractTitleAndContents(lines)
    if law_name.find('(')!=-1:
        id = law_name.find('(')
        if law_name[id+1] in hanzi_list:
            id = law_name.find('(', id+1)
            if id != -1:
                law_name = law_name[:id]
        else:
            law_name = law_name[:id]
    structure, texts, ids = [], [], []
    last1, last2, last3 = [], [], []

    output = []

    for content in contents:
        structure, texts, ids, tf = next_id(structure, texts, ids, content)
        if tf == 2:
            break
        if tf and last1:
            output.append([last1, last2, last3])
        last1, last2, last3 = list(structure), list(texts), list(ids)

    output.append([last1, last2, last3])
    try:
        id = output[1:].index(output[0])
        output = output[id + 1:]
    except:
        None

    jsonl = []
    for line in output:
        jsonl.append({'name': law_name, 'structure': line[0], 'texts': line[1], 'ids': line[2]})

    return law_name, jsonl

def process_document(txt_path):
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.readlines()
    stripped_lines = txt[2:]
    input_lines = []
    for line in stripped_lines:
        line = line.strip()
        if line:
            match = re.search(r'。第.[分编章节]', line)
            if not match:
                match = re.search(r'[\u3000\s](?:十[一二三四五六七八九]|一|二|三|四|五|六|七|八|九)、', line)
            if match:
                key = match.group()
                id = line.find(key)
                input_lines.append(line[:id+1])
                input_lines.append(line[id+1:])
            else:
                input_lines.append(line)
    return parseLaw(input_lines)



if __name__ == '__main__':
    assets_path = "./assets/"
    output_path = "output/"

    # 获取目录下所有的 .docx 文件
    txt_files = [f for f in os.listdir(assets_path) if f.endswith('.txt')]

    for txt_file in txt_files:
        txt_path = os.path.join(assets_path, txt_file)
        out_file_name, output = process_document(txt_path)
        print(out_file_name, txt_path)
        out_file_name = output_path + out_file_name + '.txt'
        with open(out_file_name, "w", encoding="utf-8") as f:
            # 将输出的 JSON 对象写入文件，以 JSONL 格式存储
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
