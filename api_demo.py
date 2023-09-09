
api_url = 'http://43.139.170.242:7000/'

def get_full_name(name):
    import requests
    res = requests.post(api_url + 'fullname', json={"name": name})
    res = res.json()
    return res['response']

def get_article_by_name_and_id(name, id, full=False):
    import requests
    res = requests.post(api_url + 'article', json={"name": name, "id": id, "full": full})
    res = res.json()
    return res['response']

def get_laws_from_embedding(text, tops, full=False, output_prefix=False, prefix_index = 0.1):
    import requests
    res = requests.post(api_url + 'laws', json={"text": text, "tops": tops, "output_prefix": output_prefix, "full": full, "prefix_index": prefix_index})
    res = res.json()
    return res['response']

def get_accurate_laws(text, range = 20, tops = 1):
    import requests
    res = requests.post(api_url + 'accurate_laws', json={"text": text, "range": range, "tops": tops})
    res = res.json()
    return res['response']

def mix_unused_laws(prompt, exist_laws_and_ids_list=[], max_length=1536, max_add=8, format="full",
                        random_shuffle=True, everyone=False):
    import requests
    data = {
        "prompt": prompt,
        "exist_laws_and_ids_list": exist_laws_and_ids_list,
        "max_length": max_length,
        "max_add": max_add,
        "format": format,
        "random_shuffle": random_shuffle,
        "everyone": everyone
    }
    res = requests.post(api_url + 'mix_unused_laws', json=data)
    res = res.json()
    return res['response']


text = '秦汉时期的刑罚主要包括笞刑、徒刑、流放刑、肉刑、死刑、羞辱刑等，下列哪些选项属于徒刑?鬼薪白粲'
print(get_laws_from_embedding(text, 3, True, True, prefix_index=0.2))
#print(get_full_name(text))
#print(get_article_by_name_and_id(text, 42))