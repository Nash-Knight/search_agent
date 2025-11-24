# Search Agent

一个基于 Qwen3-0.6B 的智能搜索助手，支持自主调用 Jina.ai 搜索引擎获取实时信息并整合回答。提供两种实现方案：标签检测模式和流式生成模式。

## 项目简介

Search Agent 实现了一个能够自主判断何时需要搜索、调用真实搜索引擎、整合多轮信息并给出带引用来源答案的智能体系统。

**核心特性：**
- 🤖 基于 Qwen3-0.6B 本地模型推理
- 🔍 自动检测搜索需求并触发 Jina.ai API
- 🌐 获取实时网络信息（股价、天气、新闻等）
- 🔄 支持多轮搜索与信息整合
- 📎 自动追踪并列出参考来源 URL
- 💡 两种实现模式：标签检测 vs 流式生成

## 核心文件

### 1. `search_agent.ipynb` - 标签检测模式（推荐）

**实现原理：**
- 模型生成包含 `<search>查询词</search>` 标签的输出
- 系统检测标签并提取查询内容
- 调用 Jina API 获取结构化结果
- 将搜索结果注入下一轮 prompt 继续生成
- 直到模型不再输出 `<search>` 标签，给出最终答案

**核心组件：**

```python
# 长度控制参数
MAX_SOURCES_PER_SEARCH = 5        # 每次搜索返回的最大信息条数
MAX_SOURCE_DESC_LEN = 400          # 单条信息描述的最大长度
MAX_FORMATTED_SOURCES_LEN = 1500   # 格式化搜索结果的最大长度
MAX_RAW_DISPLAY_LEN = 1000         # 显示原始响应的最大长度
```

**主要函数：**
- `generate(prompt, max_new_tokens=512)`: 模型推理生成
- `clear_model_cache()`: 清空KV缓存，确保测试用例独立
- `fetch_search_result(query)`: 调用 Jina API 返回结构化字典
- `extract_search_query(raw)`: 正则提取 `<search>` 标签内容
- `clean_final_response(text)`: 清理内部标签，返回用户可见答案
- `format_sources_for_prompt(sources_dict, used_sources)`: 格式化信息为 prompt
- `run_search_agent(user_query, max_rounds=5)`: 多轮主循环
- `show_rounds(rounds, used_sources, user_query)`: 格式化输出结果

**搜索结果结构：**
```python
{
  "sources": {
    "信息1": {
      "url": "https://...",
      "description": "具体内容描述（400字内）",
      "title": "标题"
    },
    "信息2": {...}
  },
  "error": None  # 或错误信息
}
```

**使用示例：**
```python
# 需要搜索的查询
rounds, sources = run_search_agent("查询特斯拉的实时股价", max_rounds=5)
show_rounds(rounds, sources, user_query="查询特斯拉的实时股价")

# 不需要搜索的查询
rounds, sources = run_search_agent("今天我心情真好")
show_rounds(rounds, sources, user_query="今天我心情真好")
```

**输出格式：**
```
Query: 查询特斯拉的实时股价

================================================================================
=== Round 1 ===
Raw_responses:
<search>特斯拉实时股价</search>

🔍 search_content: 特斯拉实时股价

📚 搜索到 5 条信息（仅展示前3条）:
  信息1: 特斯拉(TSLA)当前股价为...

================================================================================
=== Round 2 ===
Raw_responses:
特斯拉(TSLA)当前股价为XXX美元...

✅ Clean_responses (最终答案):
特斯拉(TSLA)当前股价为XXX美元，较前一交易日上涨XX%。

参考来源:
[1] https://...
[2] https://...

📎 本次对话所有追踪的链接 (共3个):
  [1] 标题1
      https://...
```

**Prompt 设计精髓：**

1. **BASE_PROMPT**: 定义搜索触发原则
   - 必须搜索：实时数据、最新信息、具体事实、不确定答案
   - 直接回答：常识、主观问题、数学逻辑

2. **后续轮次 Prompt**: 极其详细的任务说明
   ```python
   === 任务回顾 ===
   用户的原始问题(query)是: {user_query}
   
   === 当前任务(极其重要!) ===
   情况1: 信息已经足够
     第一步: 直接回答用户问题(必须执行!)
     第二步: 补充详细说明
     第三步: 列出参考来源
     第四步: 不要再输出<search>
   
   情况2: 信息不足，继续搜索
   ```

**关键特性：**
- ✅ 自动清空缓存，防止测试用例间污染
- ✅ 追踪所有引用URL，避免重复
- ✅ 分离内部思考和最终答案
- ✅ 支持最多5轮迭代搜索
- ✅ 搜索结果长度限制，防止上下文溢出

### 2. `tst2.py` - 流式生成模式（实验性）

**实现原理：**
- 使用 `model.generate()` 的流式输出
- 逐 token 检测特殊标记 `<tool_response><search>`
- 检测到标记时中断生成，调用搜索
- 使用 KV cache 继续生成（而非重新构造 prompt）
- 集成 Qwen3-max 进行信息过滤和总结

**核心特性：**
- 🔥 4-bit 量化加载（BitsAndBytes）
- 🌊 流式逐 token 生成（支持实时打印）
- 🎯 KV cache 复用（避免重复计算）
- 🧠 二级 LLM 过滤（Qwen3-max API）

**主要函数：**
- `filter(query, content)`: 调用 Qwen3-max 提取核心信息
- `generation(query, info, round)`: 流式生成主循环
  - 逐 token 生成并打印
  - 检测 `<tool_response><search>` 触发搜索
  - 维护 KV cache 进行连续生成
- `search(query)`: 调用 Jina API 并用 filter 提取

**工作流程：**
```
1. 用户输入 query
2. 模型开始流式生成
3. 检测到 "<tool_response><search>" → 暂停生成
4. 调用 Jina API 获取搜索结果
5. 用 Qwen3-max 过滤提取核心信息
6. 将过滤后信息拼接到原 query 前
7. 利用已有 KV cache 继续生成
8. 重复 2-7 直到自然结束 "<|im_end|>"
```

**配置要求：**
```python
# 环境变量
DASHSCOPE_API_KEY = "sk-xxx"  # 阿里云百炼 API Key
JINA_API = "jina_xxx"          # Jina API Key

# 模���配置
model_name = "Qwen3-0.6B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)
```

**使用示例：**
```python
query = "查询特斯拉的实时股价"
generation(query=query, info="", round=1)

# 输出:
# ===========开始第1轮生成===========
# 我需要查找搜索引擎来回答用户的问题<tool_response><search>
# 开始调用外部工具，模型生成暂停。
# info: 特斯拉(TSLA)当前股价为XXX美元...
# ===========开始第2轮生成===========
# 根据搜索结果，特斯拉当前股价为XXX美元...
# ===========模型输出结束===========
```

**优势：**
- ⚡ 真正的流式输出，实时看到生成过程
- 💾 KV cache 复用，效率高于重新生成
- 🎯 二级 LLM 过滤，信息质量更高

**劣势：**
- 🔧 需要额外配置 Qwen3-max API
- ⚙️ 需要手动管理 KV cache 和生成逻辑
- 🐛 中断-继续机制比标签检测更复杂

**对比总结：**
| 特性 | search_agent.ipynb | tst2.py |
|------|-------------------|---------|
| 实现方式 | 标签检测 + 重新生成 | 流式生成 + KV cache |
| 用户体验 | 轮次清晰，易于调试 | 实时流式，更直观 |
| 效率 | 每轮重新计算 | KV cache 复用 |
| 复杂度 | 简单，易于理解 | 复杂，需手动管理 |
| 依赖 | 仅 Jina API | Jina + Qwen3-max API |
| 推荐场景 | 开发、测试、多轮复杂查询 | 生产环境、单轮快速查询 |

## 辅助脚本

### 3. `jina.py` - Jina API 测试脚本

**功能：**
- 独立测试 Jina.ai Search API
- 保存完整 JSON 响应到 `jina_answer.json`
- 可选：集成本地模型进行 RAG 流程验证

**使用示例：**
```bash
python jina.py
# 将查询"如何种一朵玫瑰花"并保存结果到 jina_answer.json
```

**返回的 JSON 结构：**
```json
{
  "code": 200,
  "status": 20000,
  "data": [
    {
      "title": "标题",
      "url": "链接",
      "description": "内容描述（主要信息）",
      "date": "日期",
      "content": "",
      "usage": {"tokens": 1000}
    }
  ],
  "meta": {"usage": {"tokens": 10000}}
}
```

### 4. `sample.py` - 模型加载测试

**功能：**
- 验证 Qwen3-0.6B 模型加载
- 测试基本推理功能
- 确保环境配置正确

**使用示例：**
```bash
python sample.py
```

## 环境配置

### 依赖安装

```bash
# 基础依赖（search_agent.ipynb + jina.py）
pip install transformers torch requests accelerate

# tst2.py 额外依赖
pip install bitsandbytes openai python-dotenv
```

### API 配置

**search_agent.ipynb / jina.py:**
```python
JINA_API_KEY = "jina_xxx"
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}
```

**tst2.py:**
```bash
# .env 文件
DASHSCOPE_API_KEY=sk-xxx  # 阿里云百炼 API Key
JINA_API=jina_xxx          # Jina API Key
```

### 模型路径

确保 Qwen3-0.6B 模型已下载:
```
# search_agent.ipynb
C:/Users/nashk/Documents/nashknight/search_agent/Qwen3-0.6B

# tst2.py
E:/LLM/models--Qwen--Qwen3-0.6B/snapshots/xxx
```

## 快速开始

### 方案一：标签检测模式（推荐新手）

```bash
# 1. 启动 Jupyter Notebook
jupyter notebook search_agent.ipynb

# 2. 依次执行单元
#    - 导入依赖
#    - 加载模型
#    - 运行 Search Agent v2.0 core
#    - 执行测试用例

# 3. 自定义查询
rounds, sources = run_search_agent("查询特斯拉的实时股价", max_rounds=5)
show_rounds(rounds, sources, user_query="查询特斯拉的实时股价")
```

### 方案二：流式生成模式（推荐进阶）

```bash
# 1. 配置环境变量
echo "DASHSCOPE_API_KEY=sk-xxx" > .env
echo "JINA_API=jina_xxx" >> .env

# 2. 运行脚本
python tst2.py

# 3. 修改 query
# 编辑 tst2.py 最后一行的 query 变量
```

## 工作原理对比

### search_agent.ipynb 工作流

```
用户输入 → 模型生成 → 检测<search>标签 
         ↓
     调用 Jina API
         ↓
   格式化搜索结果
         ↓
  构造新 prompt → 模型再次生成 → 检测<search>
         ↓
     无<search>
         ↓
    输出最终答案
```

### tst2.py 工作流

```
用户输入 → 模型流式生成 token by token
         ↓
  检测<tool_response><search>
         ↓
     暂停生成
         ↓
  调用 Jina API → Qwen3-max 过滤
         ↓
  信息拼接 → 继续生成（复用 KV cache）
         ↓
  检测<|im_end|>
         ↓
    输出结束
```

## 技术细节

### Prompt 设计原则

1. **明确触发条件**：实时数据、最新信息、具体事实
2. **示例引导**：提供正确和错误的输出示例
3. **强调关键指令**："极其重要！"、"必须执行！"
4. **分步骤说明**：第一步...第二步...第三步...
5. **防止幻觉**："不要编造数字"、"宁可多搜索"

### 数据结构

**rounds 列表：**
```python
[
  {
    "round": 1,
    "raw": "<search>特斯拉股价</search>",
    "clean": "",
    "search": "特斯拉股价",
    "search_result": {...},
    "formatted_sources": "**搜索结果:**\n信息1:...",
    "prompt": "完整的输入 prompt"
  },
  {
    "round": 2,
    "raw": "特斯拉当前股价为XXX美元...",
    "clean": "特斯拉当前股价为XXX美元...",
    "used_sources": {"url1": "title1"}
  }
]
```

**used_sources 字典：**
```python
{
  "https://example.com/1": "标题1",
  "https://example.com/2": "标题2"
}
```

## 常见问题

**Q: 为什么需要 clear_model_cache()?**
A: 防止测试用例之间的 KV cache 污染，确保每次查询独立。

**Q: 模型不输出 <search> 标签怎么办？**
A: 检查 BASE_PROMPT 是否包含清晰的触发条件，尝试增强关键词提示。

**Q: 为什么 tst2.py 比 search_agent.ipynb 快？**
A: tst2.py 复用 KV cache，避免重复计算已生成的 token。

**Q: 搜索结果为空怎么办？**
A: 检查 Jina API Key、代理配置、查询词是否合理。

**Q: 如何限制搜索次数？**
A: 修改 `run_search_agent(max_rounds=5)` 参数。

## 注意事项

1. **代理配置**：访问外网需确保代理正常（默认 127.0.0.1:7890）
2. **API 配额**：Jina API 有 token 限制，控制 MAX_SOURCES_PER_SEARCH
3. **模型限制**：Qwen3-0.6B 轻量级，复杂推理能力有限
4. **上下文长度**：搜索结果会截断，防止超出模型窗口
5. **响应质量**：小模型可能不严格遵循指令，需多轮 prompt 调优

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- GitHub: Nash-Knight/search_agent