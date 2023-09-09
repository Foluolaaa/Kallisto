import os
from law_embe_dict import law_embe_dict

directory_path = 'law_pickle'
file_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
file_paths = [file_path for file_path in file_paths if file_path.endswith('.pickle')]
a = law_embe_dict(file_paths)

from fastapi import FastAPI, Request
import uvicorn, json, datetime

app = FastAPI()

@app.post("/fullname")
async def get_law_fullname(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print('receive get_law_fullname', json_post_list)

    name = json_post_list['name']
    response = a.get_law_fullname(name)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + '", name:"' + name + '", response:"' + repr(response) + '"'
    print(log)

    return answer

@app.post("/article")
async def get_article_by_name_and_id(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print('receive get_article_by_name_and_id', json_post_list)

    name = json_post_list['name']
    id = json_post_list['id']
    output_name_and_id = True if bool(json_post_list['full']) else False
    response = a.get_article_by_name_and_id(name, id, output_name_and_id)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + '", input:"' + str(json_post_list) + '", response:"' + repr(response) + '"'
    print(log)

    return answer

@app.post("/laws")
async def get_laws_from_embedding(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print('get_laws_from_embedding', json_post_list)

    text = json_post_list['text']
    tops = int(json_post_list['tops'])
    output_prob = True if bool(json_post_list['full']) else False
    output_prefix = json_post_list.get('output_prefix', False)
    prefix_index = json_post_list.get('prefix_index', 0.1)
    response = a.get_full_from_embedding(text, tops, output_prob, output_prefix, prefix_index)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + '", input:"' + str(json_post_list) + '", response:"' + repr(response) + '"'
    print(log)

    return answer

@app.post("/identify_laws")
async def identify_laws_in_article(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print('get_laws_from_embedding', json_post_list)

    text = json_post_list['text']
    response = a.identify_laws_in_article(text)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + '", input:"' + str(json_post_list) + '", response:"' + repr(response) + '"'
    print(log)

    return answer

@app.post("/accurate_laws")
async def get_top_laws(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print('get_laws_from_embedding', json_post_list)

    text = json_post_list['text']
    range = int(json_post_list['range']) if 'range' in json_post_list else 20
    tops = int(json_post_list['tops']) if 'tops' in json_post_list else 1
    response = a.get_top_laws(text, range, tops)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + '", input:"' + str(json_post_list) + '", response:"' + repr(response) + '"'
    print(log)

    return answer

@app.post("/mix_unused_laws")
async def mix_unused_laws(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_dict = json.loads(json_post)

    print('mix_unused_laws', json_post_dict)

    prompt = json_post_dict['prompt']
    exist_laws_and_ids_list = json_post_dict.get('exist_laws_and_ids_list', [])
    max_length = int(json_post_dict.get('max_length', 1536))
    max_add = int(json_post_dict.get('max_add', 8))
    format = json_post_dict.get('format', 'full')
    random_shuffle = json_post_dict.get('random_shuffle', True)
    everyone = json_post_dict.get('everyone', False)

    response = a.mix_unused_laws(prompt, exist_laws_and_ids_list, max_length, max_add, format, random_shuffle, everyone)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + 'input:"' + str(json_post_dict) + '", response:"' + repr(response) + '"'
    print(log)

    return answer

@app.post("/chat")
async def get_article_by_name_and_id(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print('receive get_article_by_name_and_id', json_post_list)

    prompt = json_post_list['prompt']
    from utils.glm import chat
    response = chat(prompt)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + '", input:"' + str(json_post_list)
    print(log)

    return answer

@app.post("/embedding")
async def get_article_by_name_and_id(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print('receive get_article_by_name_and_id', json_post_list)

    prompt = json_post_list['prompt']
    from utils.glm import embedding
    response = embedding(prompt)

    now = datetime.datetime.now()
    time_tick = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time_tick
    }

    log = "[" + time_tick + "] " + '", input:"' + str(json_post_list)
    print(log)

    return answer

if __name__ == '__main__':
    uvicorn.run("api:app", host='0.0.0.0', port=8000, workers=4)