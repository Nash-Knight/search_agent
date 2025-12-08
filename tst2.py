from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from openai import OpenAI
import requests
import os
import dotenv
import time
import json
dotenv.load_dotenv()
model_name = r"E:\LLM\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,            # 原来的 load_in_4bit=True
    bnb_4bit_compute_dtype="float16", 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto"
)

def filter(query, content):
    res = ""
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    prompt = f"""
    你是一个擅长提取核心信息的助手。接下来我会给你一个query和一大堆信息，请你帮我从中找出与query直接相关的信息进行总结并返回。
    注意，只要返回你的总结结果，不要输出任何思考过程。
    将你最核心的输出包裹在<response></response>中。
    以下是一个简单的例子：
    query：{query}
    info：{content}
    你的回答：<response>大语言模型是是一种基于深度学习的人工智能技术，主要用于生成和理解自然语言文本。</response>
    """
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in completion:
        res += chunk.choices[0].delta.content
    return res


def generation(query, info, round, model=model, tokenizer=tokenizer):
    tmp = ""
    sys_prompt = f"""
    你是一个可以使用外部搜索引擎的 AI 助手。

    你的规则如下：

    1. 如果你认为自己**能够直接回答**，或根据用户给入的额外信息进行**总结提取**后，能立即回答该问题，则直接回答，不要调用工具，不要输出<tool_response><search>。
    2. 如果你认为需要调用外部搜索引擎来获取信息，则停止正常回答，并严格输出：
    **额外信息**:{info}
    <tool_response><search>

    不要添加空格、不要添加解释说明、不要添加换行，仅输出这个固定字符串。
    <例> 我需要查找搜索引擎来回答用户的问题<tool_response><search>


    请务必严格遵守以上规则。
    """.strip()
    
    messages = [
            {"role": "system", "content": sys_prompt},
            # "role": "user", "content": info},
            {"role": "user", "content": query}
        ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # inputs: {'input_ids': tensor([[151644, 8948, ... ,77091, 198]]), 'attention_mask': tensor([[1, 1, ... ,1, 1]]
    # type_inputs: <class 'transformers.tokenization_utils_base.BatchEncoding'>

    past_key_values = None  # KV cache的注意力矩阵
    generated = inputs["input_ids"]  # 取出生成部分
    # print(generated) # tensor([[xxx, xxx, xxx]])

    # 用于拼接完整生成
    full_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    print(f"===========开始第{round}轮生成===========")
    for step in range(50):  # 最多生成 N 步，每步生成一个新的token
        # 第一步input_ids=generated因为没有KV cache，后面每次只传入新生成的token，因为有KV cache的存在
        # 一步一步生成 token
        outputs = model(
            input_ids=generated,
            past_key_values=past_key_values,
            use_cache=True
        )
        # print(outputs.logits.shape)  # (batch_size, seq_len, vocab_size)  -> torch.Size([1, 22, 151936])
        """outputs:
        ==================================
        CausalLMOutputWithPast(
        loss=None,
        logits=tensor([...]), # logits.shape == (batch_size, seq_len, vocab_size)  -> torch.Size([1, 22, 151936])
        past_key_values=DynamicCache(layers=[...]),
        hidden_states=None,
        attentions=None 
        )
        ==================================
        1. 生成步骤没有标签，因此loss = None
        2. logits: 模型对每个token的预测分布，包括已经生成的token。只需要采样最后一行，才是next token的分布
        3. past_key_values: KV cache
        """

        logits = outputs.logits[:, -1, :] # logits.shape = torch.Size(batch_size=1, seq_len, vocab_size)
        # print(logits.shape)  (batch_size, vocab_size)  -> torch.Size([1, 151936])

        # 更新KV缓存
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(logits, dim=-1).unsqueeze(0) 
        # dim=-1:最后一维
        # 模型下一轮生成期望输入是：[batch_size, seq_len]，因此要unsqueeze(0)
        generated = next_token # tensor([[]])

        decoded = tokenizer.decode(next_token[0])
        
        print(decoded, end="", flush=True)   # flush=True:强制立刻把输出写到终端，而不是放在缓冲区里等待
        tmp += decoded  # 维护tmp是因为终端打印远慢于赋值操作
        # 只保留结尾部分即可
        tmp = tmp[-20:]
        time.sleep(0.25)
        full_text += decoded
        
        
        if "<|im_end|>" in tmp:   # 正常输出结束
            print("\n===========模型输出结束===========")
            return
        if "<search>" in tmp:  # 发现llm开始调用工具
            print("\n开始调用外部工具，模型生成暂停。")
            info = search(query)
            print("info:", info)
            new_query = info + query
            new_round = round + 1
            torch.cuda.empty_cache()
            return generation(new_query, info, new_round)
    print('\n模型达到最大生成长度，强制结束。')
    print("\n===========模型输出结束===========")

def search(query:str):
    url = "https://s.jina.ai/?q=" + query + "&gl=CN&hl=zh-cn"

    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + os.getenv("JINA_API"),
        "X-Engine": "https://www.bing.com"
    }

    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
        resp_data = resp.json()
        result = resp_data.get("data")  # result:List[dict{'title', 'url', 'description', 'date', 'content', 'metadata', 'external', 'usage'}]
        content = result[0]['title'] + result[0]['description'] + result[0]['content']
        filtered_content = filter(query=query, content=content)
        return filtered_content
    
    
if __name__ == "__main__":
    query = "查询特斯拉的实时股价"
    generation(query=query, info="", round=1)