import requests
import json
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import traceback
from pathlib import Path

# =======================================================================
#                           JINA SEARCH 配置
# =======================================================================
# 替换为您真实的 Jina API 密钥
JINA_API_KEY = "jina_800f62ec9cc745e09f058c4652a961feziG6FeCa71toa9my7gXm3prQbJaF"
JINA_SEARCH_ENDPOINT = "https://s.jina.ai"

# 代理配置 (基于您的 Clash 设置 127.0.0.1:7890)
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}
# =======================================================================
#                           本地模型配置
# =======================================================================
# Qwen3-0.6B 模型路径
LOCAL_MODEL_PATH = "C:/Users/nashk/Documents/nashknight/search_agent/Qwen3-0.6B"
# =======================================================================


def jina_search_agent(query: str, use_proxy: bool = True) -> str:
    """
    使用 Jina Search API 查找网页内容，并将结果格式化为 LLM 上下文。
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Respond-With": "no-content"
    }
    # 使用正确的 URL 格式：在 URL 中直接包含查询参数
    url = f"{JINA_SEARCH_ENDPOINT}/?q={requests.utils.quote(query)}"
    proxies_config = PROXIES if use_proxy else None
    
    try:
        response = requests.get(
            url, 
            headers=headers, 
            proxies=proxies_config,
            timeout=15
        )
        response.raise_for_status()

        results_json = response.json()
        
        # 保存完整 JSON 到文件
        with open('jina_answer.json', 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON 响应已保存到: jina_answer.json")
        
        # 打印完整 JSON 结构以供调试
        print("\n" + "="*80)
        print("完整 JSON 响应结构:")
        print("="*80)
        print(json.dumps(results_json, indent=2, ensure_ascii=False))
        print("="*80 + "\n")
        
        search_results: List[Dict[str, Any]] = results_json.get('data', [])

        if not search_results:
             return "Jina Search successful, but no relevant results were returned."

        # 简洁格式化结果
        context = []
        for i, item in enumerate(search_results[:5]): # 只取前 5 条结果
            title = item.get('title', 'No Title')
            url = item.get('url', 'N/A')
            content = item.get('content', 'No content snippet.')
            
            # 使用更简洁的格式
            context.append(f"[SOURCE {i+1}] Title: {title}\nSnippet: {content.strip()}")
            
        return "\n---\n".join(context)

    except requests.exceptions.HTTPError as e:
        return f"[SEARCH ERROR] HTTP {e.response.status_code}: API Key or quota issue. Details: {e}"
    except requests.exceptions.RequestException as e:
        return f"[NETWORK ERROR] Connection failed. Check proxy (127.0.0.1:7890) and internet. Details: {e}"


# --- 您的模型推理函数（略作修改以使用全局路径） ---
def local_model_inference(prompt):
    """
    使用本地 Qwen3-0.6B 模型进行推理。
    """
    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"错误：模型路径不存在: {LOCAL_MODEL_PATH}")
        raise FileNotFoundError(LOCAL_MODEL_PATH)
    
    print(f"✅ 模型路径确认: {LOCAL_MODEL_PATH}")
    
    # 首次调用时加载模型和分词器
    if 'model' not in local_model_inference.__dict__:
        print("首次加载模型中，请稍候...")
        try:
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

            # 决定加载类型
            has_cuda = torch.cuda.is_available()
            device_map = "auto" if has_cuda else None
            dtype = torch.float16 if has_cuda else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                device_map=device_map,
                local_files_only=True,
                dtype=dtype,
            )
            local_model_inference.model = model
            local_model_inference.tokenizer = tokenizer
            print(f"模型加载完成，设备: {model.device}")

        except Exception:
            print("模型加载失败，请检查 transformers, torch, accelerate 等依赖是否正确安装。")
            traceback.print_exc()
            raise
    
    model = local_model_inference.model
    tokenizer = local_model_inference.tokenizer

    # 准备模型输入 (使用 Chat Template)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    # 生成文本
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512, # 增加 token 数量以适应 RAG 结果
        do_sample=True,      # 启用采样以获得更自然的回答
        temperature=0.7
    )
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 移除输入 Prompt 部分，只返回模型的回答
    output_text = output.split(text)[-1].strip() if text in output else output.strip()
    return output_text


# =======================================================================
#                           主程序 RAG 流程
# =======================================================================

if __name__ == "__main__":
    
    # 1. 定义查询内容
    USER_QUERY = "如何种一朵玫瑰花"
    
    print("--- 1. Jina.ai 搜索开始 ---")
    
    # 2. 调用 Jina Search Agent 获取上下文
    search_context = jina_search_agent(USER_QUERY, use_proxy=True)

    print("\n--- 2. 搜索结果摘要 ---")
    
    if search_context.startswith("[SEARCH ERROR]") or search_context.startswith("[NETWORK ERROR]"):
        print(f"搜索失败，将仅使用模型的内部知识。错误信息: {search_context}")
        rag_prompt = f"请详细回答用户的问题：{USER_QUERY}"
    else:
        print("搜索成功，已获取以下上下文。")
        # 3. 构建 RAG Prompt，将搜索结果嵌入
        rag_prompt = f"""
        以下是来自搜索引擎的上下文信息，用于指导你的回答：
        <search_context>
        {search_context}
        </search_context>

        请根据上述上下文信息，详细、全面地回答用户的问题。如果上下文信息不足，请明确指出。
        用户问题：{USER_QUERY}
        """

    print("\n--- 3. LLM RAG Prompt 构建完成 ---")
    print(rag_prompt)
    
    print("\n--- 4. 本地模型推理开始 ---")
    try:
        # 4. 调用本地模型进行推理
        final_answer = local_model_inference(rag_prompt)
        
        print("\n==============================================")
        print(f"✅ RAG Agent 最终答案:\n{final_answer}")
        print("==============================================")

    except Exception:
        print("\n❌ 模型推理或主流程失败。")
        traceback.print_exc()