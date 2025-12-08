import argparse
import re
import sys
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============ 基本配置 ============
LOCAL_MODEL_PATH = r"C:/Users/nashk/Documents/nashknight/search_agent/Qwen3-0.6B"

# 长度控制参数（与 notebook 保持一致）
MAX_SOURCES_PER_SEARCH = 5
MAX_SOURCE_DESC_LEN = 400
MAX_FORMATTED_SOURCES_LEN = 1500
MAX_RAW_DISPLAY_LEN = 1000

BASE_PROMPT = (
    "你是一个智能搜索助手。你的任务是分析用户问题,判断是否需要搜索来获取信息。\n\n"
    "**核心判断原则:**\n"
    "在分析问题时,如果遇到以下情况,必须使用<search>标签搜索:\n"
    "1. 需要实时数据(股价、天气、新闻、汇率等会变化的信息)\n"
    "2. 需要最新信息(当前状态、今天/现在的情况)\n"
    "3. 需要具体事实(某个公司的数据、某个地点的情况、某个产品的参数)\n"
    "4. 你不确定答案,或答案可能过时\n"
    "5. 用户明确要求查询、搜索、查找信息\n\n"
    "如果是以下情况,可以直接回答:\n"
    "1. 常识性问题(如何做某事、概念解释)\n"
    "2. 主观问题(建议、意见)\n"
    "3. 数学计算、逻辑推理\n\n"
    "**确定需要搜索时的输出格式**:\n"
    "<search>简短查询词</search>\n\n"
    "**不需要搜索时,直接给出答案。**\n\n"
    "**关键规则:**\n"
    "- <search>标签单独成行,内容3-10字\n"
    "- 没搜索前不要编造数字、日期等事实\n"
    "- 宁可多搜索,不要猜测\n"
    "- 如果已有足够信息,直接回答并列出参考来源,不要再输出<search>\n"
)

# Jina Search API 配置（保持与 notebook 一致，可按需修改为环境变量）
JINA_API_KEY = "jina_800f62ec9cc745e09f058c4652a961feziG6FeCa71toa9my7gXm3prQbJaF"
JINA_SEARCH_ENDPOINT = "https://s.jina.ai"
PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

# ============ 核心模型与工具函数（从 notebook 精简移植） ============

def load():
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return m, tok

model, tokenizer = load()


def generate(prompt, max_new_tokens=512):
    msgs = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inp = tokenizer(text, return_tensors="pt")
    for k, v in inp.items():
        inp[k] = v.to(model.device)
    ids = model.generate(**inp, max_new_tokens=max_new_tokens)
    gen_ids = ids[0][inp["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    clean_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_ids.tolist(), raw_text, clean_text


def clear_model_cache():
    if hasattr(model, "past_key_values"):
        model.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fetch_search_result(query: str, use_proxy: bool = True, max_sources: int = MAX_SOURCES_PER_SEARCH) -> dict:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Respond-With": "no-content",
    }
    url = f"{JINA_SEARCH_ENDPOINT}/?q={requests.utils.quote(query)}"
    proxies_cfg = PROXIES if use_proxy else None
    try:
        resp = requests.get(url, headers=headers, proxies=proxies_cfg, timeout=12)
        resp.raise_for_status()
        results_json = resp.json()
        data = results_json.get("data", [])
        if not data:
            return {"error": f"未找到结果: {query}", "sources": {}}
        sources = {}
        for i, item in enumerate(data[:max_sources], 1):
            title = (item.get("title") or "").strip()[:120]
            description = (item.get("description") or "").strip()[:MAX_SOURCE_DESC_LEN]
            url_link = (item.get("url") or "").strip()
            if description and url_link:
                sources[f"信息{i}"] = {"url": url_link, "description": description, "title": title}
        return {"sources": sources, "error": None} if sources else {"error": "结果无有效内容", "sources": {}}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "sources": {}}


def extract_search_query(raw: str):
    match = re.search(r"<search>\s*([^<\n]+?)\s*</search>", raw, re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        if query and len(query) <= 80:
            return query, raw[:match.end()]
    return None, None


def clean_final_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<search>.*?</search>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?(?:think|search)>", "", text, re.IGNORECASE)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text.strip()


def format_sources_for_prompt(sources_dict: dict, used_sources: dict) -> str:
    if not sources_dict:
        return "[无可用信息]"
    lines = ["**搜索结果:**"]
    for key, val in sources_dict.items():
        desc = val.get("description", "")[:300]
        url = val.get("url", "")
        title = val.get("title", "未命名")
        lines.append(f"{key}({title}): {desc}")
        lines.append(f"  URL: {url}")
        if url and url not in used_sources:
            used_sources[url] = title
    return "\n".join(lines)


def run_search_agent(user_query, max_rounds=5, max_new_tokens=512, use_proxy=True):
    clear_model_cache()
    rounds = []
    prompt = f"{BASE_PROMPT}\n用户问题: {user_query}"
    search_count = 0
    used_sources = {}

    for r in range(1, max_rounds + 1):
        _, raw, clean = generate(prompt, max_new_tokens=max_new_tokens)
        entry = {"round": r, "raw": raw, "clean": clean, "prompt": prompt}

        q, raw_trunc = extract_search_query(raw)
        if q:
            search_count += 1
            entry["raw"] = raw_trunc if raw_trunc else raw
            entry["search"] = q

            result_dict = fetch_search_result(q, use_proxy=use_proxy)
            entry["search_result"] = result_dict

            if result_dict.get("error"):
                print(f"⚠️ 搜索失败: {result_dict['error']}")
                rounds.append(entry)
                break

            sources = result_dict.get("sources", {})
            formatted_sources = format_sources_for_prompt(sources, used_sources)
            entry["formatted_sources"] = formatted_sources[:MAX_FORMATTED_SOURCES_LEN]
            rounds.append(entry)

            prompt = (
                f"{BASE_PROMPT}\n\n"
                f"=== 任务回顾 ===\n"
                f"用户的原始问题(query)是: {user_query}\n\n"
                f"你是一个智能搜索助手,负责为用户查询信息并给出专业答案。\n"
                f"你在第{search_count}轮搜索中已经查询了关键词'{q}',获得了以下搜索结果:\n\n"
                f"{formatted_sources[:MAX_FORMATTED_SOURCES_LEN]}\n\n"
                f"=== 当前任务(极其重要!) ===\n"
                f"现在你需要分析这些搜索结果,判断是否足够回答用户的问题:\n\n"
                f"**情况1: 信息已经足够 (这是最核心的环节!)**\n"
                f"如果以上搜索结果包含了足够的信息来回答'{user_query}',请务必按以下步骤操作:\n\n"
                f"第一步(最重要!必须执行!): 直接回答用户问题(query)\n"
                f"  - 用1-3句话给出核心答案,包含具体数字/事实，回答要围绕query，简明扼要\n"
                f"  - 例如用户问股价,你要说'特斯拉(TSLA)当前股价为XXX美元'\n"
                f"  - 严禁照抄输出示例！一定要根据先前的所有信息和分析自己总结！\n"
                f"  - 不要跳过这一步!这是用户最需要的!\n\n"
                f"第二步: 补充详细说明(可选)\n"
                f"  - 如果有额外有价值的信息,简要补充\n"
                f"  - 如涨跌幅、市值等相关数据\n\n"
                f"第三步: 列出参考来源\n"
                f"  - 格式:\n"
                f"    参考来源:\n"
                f"    [1] URL1\n"
                f"    [2] URL2\n\n"
                f"第四步: 不要再输出<search>标签\n\n"
                f"输出示例:\n"
                f"特斯拉(TSLA)当前股价为XXX美元,较前一交易日上涨XX%。\n\n"
                f"参考来源:\n"
                f"[1] https://...\n"
                f"[2] https://...\n\n"
                f"**情况2: 信息不足,需要继续搜索**\n"
                f"如果搜索结果不够详细,或者缺少关键信息:\n"
                f"1. 分析缺少什么信息\n"
                f"2. 输出<search>新的查询词</search>来获取更多细节\n"
                f"3. 注意不要重复搜索相同的关键词\n\n"
                f"=== 关键提醒 ===\n"
                f"- 这些搜索结果都是你自己查询得到的,不是用户提供的\n"
                f"- 你必须给出实质性答案,不能只列出链接!用户需要你的总结!\n"
                f"- 信息足够就立即回答,先答案后链接,不要只有链接没有答案!\n"
                f"- 回答要专业简洁,直接针对问题,不要说'我需要分析'这类话\n\n"
                f"现在请立即给出你的答案(记住:先回答问题,再列参考来源):"
            )
            continue

        # 无 <search> 标签,认为是最终答案
        entry["clean"] = clean_final_response(clean)
        entry["used_sources"] = used_sources.copy()
        rounds.append(entry)
        break

    return rounds, used_sources


def show_rounds(rounds, used_sources=None, user_query=None):
    if user_query:
        print(f"Query: {user_query}")

    for i, info in enumerate(rounds, 1):
        print(f"\n{'='*80}")
        print(f"=== Round {info['round']} ===")

        print('Raw_responses:')
        raw = info.get('raw', '')
        print(raw[:MAX_RAW_DISPLAY_LEN] + '...' if len(raw) > MAX_RAW_DISPLAY_LEN else raw)

        if 'search' in info:
            print(f"\n🔍 search_content: {info['search']}")

        if 'search_result' in info:
            result = info['search_result']
            if result.get('error'):
                print(f"\n⚠️ 搜索错误: {result['error']}")
            else:
                sources = result.get('sources', {})
                print(f"\n📚 搜索到 {len(sources)} 条信息（仅展示前3条）:")
                for key, val in list(sources.items())[:3]:
                    print(f"  {key}: {val.get('description', '')[:100]}...")

        print()
        is_last = (i == len(rounds))
        if is_last:
            print('✅ Clean_responses (最终答案):')
            print(info.get('clean') or '[无最终回答]')
            if used_sources:
                print(f"\n📎 本次对话所有追踪的链接 (共{len(used_sources)}个):")
                for idx, (url, title) in enumerate(list(used_sources.items())[:], 1):
                    print(f"  [{idx}] {title}\n      {url}")
        print('='*80)


# ============ CLI 入口 ============

def main():
    parser = argparse.ArgumentParser(description="Search Agent CLI (标签检测 + Jina 搜索)")
    parser.add_argument("-p", "--prompt", required=True, help="用户查询语句，例如：-p 查询特斯拉的实时股价")
    parser.add_argument("--max-rounds", type=int, default=5, help="最大搜索轮次，默认 5")
    parser.add_argument("--no-proxy", action="store_true", help="不使用代理访问 Jina")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="单轮最大生成 token 数")
    args = parser.parse_args()

    user_query = args.prompt
    use_proxy = not args.no_proxy
    rounds, sources = run_search_agent(user_query, max_rounds=args.max_rounds, max_new_tokens=args.max_new_tokens, use_proxy=use_proxy)
    show_rounds(rounds, sources, user_query)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
