#本函数仅可在服务器上使用
def embedding(prompt):
    import requests
    res = requests.post('http://43.139.170.242:7000/embedding', json={"prompt": prompt})
    res = res.json()
    return res['response']

def chat(prompt):
    import requests
    res = requests.post('http://43.139.170.242:7000/chat', json={"prompt": prompt})
    res = res.json()
    return res['response']