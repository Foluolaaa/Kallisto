#本代码为系统与GPT进行交互的封装接口
#该代码需在梯子环境下运行
import random

OPENAI_API_KEY = ['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','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']

#与ChatGPT进行单轮对话的模块
def chat(prompt):
    import openai
    openai.api_key = random.choice(OPENAI_API_KEY)
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )
            break
        except:
            from time import sleep
            sleep(0.2)

    answer = completion.choices[0].message['content']
    return answer


#基于ChatGPT求embedding的模块
def embedding(text):
    import openai
    openai.api_key = random.choice(OPENAI_API_KEY)

    while True:
        try:
            completion = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            break
        except:
            from time import sleep
            sleep(0.2)
    result = completion['data'][0]['embedding']
    return result
