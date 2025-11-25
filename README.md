# Search Agent

本仓库实现了一个基于本地 Qwen3-0.6B 模型的智能搜索代理（Search Agent），能够在必要时调用 Jina.ai 搜索接口获取实时信息并整合为带来源的答案。仓库包含两条互相独立的实现思路：

- `search_agent.ipynb`：推荐的「标签检测」模式（详细、易复现、便于汇报）。
- `tst2.py`：实验性的「流式生成 + KV cache」模式（更接近实时流式交互，需要额外 API）。

本 README 重点说明两部分的目的、运行方法与关键函数，便于汇报与演示。

**目录结构（简要）**
- `search_agent.ipynb` — 标签检测模式（含完整代码与说明）
- `tst2.py` — 流式生成模式（实验性脚本）
- `jina.py` — 简单的 Jina API 测试脚本
- `sample.py` — 模型加载与推理测试脚本
- `Qwen3-0.6B/` — 本地模型文件夹

---

**总体说明（高层）**

1. 设计目标：当用户问题需要最新/具体事实（如股价、天气、新闻等）时，自动触发外部搜索；否则直接使用本地模型回答。答案应包含引用来源并在内部与外部信息之间做清晰分离。
2. 两种实现的侧重点：
   - 标签检测（`search_agent.ipynb`）：模型输出包含 `<search>查询词</search>` 标签触发搜索，基于多轮 prompt 迭代生成最终答案。实现思路清晰、便于复现与演示。
   - 流式生成（`tst2.py`）：逐 token 生成并在检测到工具触发时暂停、调用搜索，然后继续生成（复用 KV cache）。延迟低但代码复杂且需要额外 API（例如 Qwen3-max）。

---

**快速开始（建议演示流程）**

1. 安装依赖：

```powershell
pip install transformers torch requests accelerate
```

（若使用 `tst2.py` 的 4-bit 量化/流式功能，还需：）

```powershell
pip install bitsandbytes openai python-dotenv
```

2. 准备模型：确认 `Qwen3-0.6B` 文件夹在 `C:/Users/nashk/Documents/nashknight/search_agent/Qwen3-0.6B`。

3. 运行（建议先在 Notebook 中按单元顺序执行 `search_agent.ipynb`）：

```powershell
# 启动 jupyter notebook 或 lab
jupyter notebook search_agent.ipynb
```

或者运行 `tst2.py`（实验脚本）：

```powershell
python tst2.py
```

注意：访问外网需设置代理（默认示例 `127.0.0.1:7890`）并配置 `JINA_API_KEY`、`JINA_API` 等环境变量或在脚本中设置常量。

---

**详细说明 — `search_agent.ipynb`**

本节分为三部分：整体工作流、每个重要函数的职责与实现要点、运行示例与注意事项。

1) 工作流概述

- 用户输入原始问题（`user_query`）。
- 使用 `BASE_PROMPT` 指导模型判断是否需要检索：若需要，模型应输出格式化标签 `<search>关键词</search>`。
- 系统检测到 `<search>` 标签后：调用 `fetch_search_result()` 拉取搜索结果并格式化为 `formatted_sources`；将该信息注入下一轮 prompt 并继续生成。
- 重复上述步骤直到模型不再输出 `<search>`；将模型生成的可见内容（移除内部标签后的）作为最终答案，并列出引用来源。

<div style="text-align:center">
  <img src="imgs/flowchart.png" alt="Search Agent 流程图" style="width:40%" />

  <p style="font-style:italic; margin-top:8px;">图：Search Agent 标签检测工作流（用户输入 → 模型判断是否搜索 → 调用 Jina → 整合信息 → 迭代/输出最终答案）</p>
</div>

2) 关键函数（按 notebook 实现顺序）

- `load()`
  - 目的：加载分词器与模型（`AutoTokenizer`, `AutoModelForCausalLM`）。
  - 关键点：使用 `local_files_only=True` 以保证从本地 `Qwen3-0.6B` 加载；如果有 GPU，脚本会自动使用 `device_map='auto'`。

- `generate(prompt, max_new_tokens=512)`
  - 目的：将 prompt 输入模型并产生后续 token，返回生成的 token ids、原始解码（含特殊 token）以及干净文本（跳过特殊 token）。
  - 实现要点：
    - 对话模板兼容：若 tokenizer 支持 `apply_chat_template`，会使用该方法构建模型输入；否则直接用 prompt。
    - 将输入张量移动到模型所在设备，并传入 `model.generate()`。
    - 仅取 `input_ids` 之后的新生成部分进行解码与返回，防止重复包含 prompt 内容。

- `clear_model_cache()`
  - 目的：清理模型 KV cache 与 GPU 缓存，保证每个查询独立（避免测试间干扰）。
  - 实现要点：检查并移除 `past_key_values`，在有 GPU 时调用 `torch.cuda.empty_cache()`。

- `fetch_search_result(query: str, use_proxy: bool = True, max_sources: int = MAX_SOURCES_PER_SEARCH) -> dict`
  - 目的：调用 Jina Search API（`https://s.jina.ai`）获取结构化搜索结果。
  - 返回：标准字典，形如 `{ 'sources': { '信息1': {'url':..., 'description':..., 'title':...}, ...}, 'error': None }`；出错时返回 `{ 'error': '错误信息', 'sources': {} }`。
  - 实现要点：
    - 使用 `requests.get()`，支持可选代理 `PROXIES`。
    - 解析 `data` 字段，截取前 `max_sources` 条并裁剪 `description` 长度以防上下文爆炸。
    - 处理网络/解析异常并以 `error` 字段返回错误信息。

- `extract_search_query(raw: str)`
  - 目的：从模型原始输出中提取 `<search>...</search>` 标签内容。
  - 实现要点：使用正则 `re.search(r'<search>\s*([^<\n]+?)\s*</search>')`，只接受长度合理（<=80）的查询词，返回 `(query, raw_trunc)`，其中 `raw_trunc` 是截断到标签结束位置的原始输出，便于展示。

- `clean_final_response(text: str) -> str`
  - 目的：将模型的最终输出清洗为用户可见的文本，移除内部标签 `<think>`、`<search>` 及其内容。
  - 实现要点：使用 `re.sub()` 删除这些标签与其包裹内容，并去除空行与多余空白，返回整洁文本。

- `format_sources_for_prompt(sources_dict: dict, used_sources: dict) -> str`
  - 目的：把 `fetch_search_result()` 返回的 sources 格式化为可以拼接回 prompt 的文本片段，并在 `used_sources` 中登记已使用的 URL（用于最终引用列表）。
  - 实现要点：对每条 source 截断 `description` 字段，生成易读的多行字符串并更新 `used_sources`。

- `run_search_agent(user_query, max_rounds=5, max_new_tokens=512, use_proxy=True)`
  - 目的：多轮驱动的核心函数，协调生成、检测 `<search>`、调用搜索并迭代直到得到最终答案或出错。
  - 核心逻辑：
    1. 清理模型缓存以确保独立运行。
    2. 初始化 `prompt`（将 `BASE_PROMPT` 与 `user_query` 拼接）。
    3. 循环最多 `max_rounds` 轮：每轮调用 `generate()` 产出 `raw`。
    4. 用 `extract_search_query()` 检测是否需要搜索；若需要，调用 `fetch_search_result()`，格式化结果并把信息注入新的 `prompt`，进入下一轮；若不需要搜索，则用 `clean_final_response()` 清理并返回最终结果。
    5. 在每轮中维护 `used_sources`（全局追踪），并把每轮信息以结构化字典追加到 `rounds` 列表，便于展示与调试。

- `show_rounds(rounds, used_sources=None, user_query=None)`
  - 目的：以可读的文本格式展示多轮执行细节（用于汇报与排查），包括每轮 raw 输出、搜索关键词、搜索到的部分信息与最终答案及引用列表。

3) 运行示例（汇报用）

- 不需要搜索的问法：`今天我心情真好` → 直接返回模型生成的答案。
- 需要搜索的问法：`查询特斯拉的实时股价` → 期望模型先输出 `<search>特斯拉实时股价</search>`，系统调用 Jina，整合后返回最终答案并列来源。

4) 注意事项与汇报要点

- 展示时强调「标签检测」方案的优势：实现简单、每轮 prompt 可见、便于调试与复现。
- 演示时务必展示 `rounds` 的内容与 `used_sources`，说明如何通过多轮查询逐步充实信息并最终回答。
- 网络请求与代理：在无代理或 API Key 错误时，`fetch_search_result()` 会返回 `error`，应在演示中说明如何配置 `JINA_API_KEY` 与 `PROXIES`。

---

**简要说明 — `tst2.py`**

1) 目标：实现真正的流式生成（token-by-token），在模型需要外部信息时即时触发工具调用并继续生成（利用 KV cache）。
2) 关键特点：
   - 使用 `BitsAndBytesConfig` 做 4-bit 量化以减少显存占用。
   - 手工按步调用模型前向，维护并复用 `past_key_values`（KV cache）。
   - 在检测到 `<search>` 或 `<tool_response>` 标记时暂停生成，调用 `search()`（同样使用 Jina），并用二级模型（示例中使用 Aliyun Dashscope / `qwen3-max`）对检索内容进行过滤与摘要。
3) 限制与注意事项：
   - 代码为实验性实现，直接使用 `torch.argmax` 作为解码策略（可替换为采样策略以获得更自然文本）。
   - 需要额外 API Key（例如 `DASHSCOPE_API_KEY`、`JINA_API`），并可能需要对 `tokenizer.apply_chat_template` 的调用兼容性进行调整。

---

如果你需要，我可以：
- 将 `search_agent.ipynb` 中的函数注释部分提取为一个单独的 `.py` 文件以便代码审阅；
- 为 `README.md` 再生成一个更简短的 PPT 演示要点（每页 3-5 bullet）以便汇报。 

---

作者与联系方式：Nash-Knight

许可证：MIT
