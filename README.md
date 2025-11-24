# Search Agent

一个基于 Qwen3-0.6B 的多轮搜索智能体，支持自主调用 Jina.ai 搜索引擎获取实时信息并整合回答。

## 项目简介

Search Agent 实现了一个能够自主判断何时需要搜索、调用真实搜索引擎、整合多轮信息并给出带引用来源答案的智能体系统。

**核心特性：**
- 🤖 基于 Qwen3-0.6B 本地模型推理
- 🔍 自动检测 `<search>查询</search>` 标签触发搜索
- 🌐 集成 Jina.ai Search API 获取实时网络信息
- 🔄 支持多轮搜索与信息整合
- 📎 自动追踪并列出参考来源 URL
- 💡 内部思考（raw）与最终答案（clean）分离展示

## 版本历史

### v1.0 - 基础架构
- 实现基本的 `<search>` 标签检测
- 固定搜索结果注入（占位符）
- 单轮搜索 + 整合输出

### v2.0 - 真实搜索集成（当前版本）
- 接入 Jina.ai Search API
- 结构化搜索结果（字典格式：`{信息1: {url, description, title}}`）
- 支持多轮搜索（模型可根据信息充分性决定是否继续搜索）
- URL 引用追踪与自动汇总
- 优化 prompt 指导模型分析 description 并选取有用信息

## 文件说明

### 核心文件

#### `search_agent.ipynb`
主要实现脚本，包含完整的 Search Agent v2.0 逻辑。

**主要组件：**
- `BASE_PROMPT`: 搜索智能体行为规则
- `fetch_search_result(query)`: 调用 Jina API 并返回结构化结果
- `extract_search_query(raw)`: 从模型输出中提取 `<search>` 标签内容
- `format_sources_for_prompt(sources_dict, used_sources)`: 格式化信息为 prompt 并追踪 URL
- `run_search_agent(user_query, ...)`: 多轮主循环，支持迭代搜索与整合
- `show_rounds(rounds, used_sources)`: 格式化输出每轮结果与最终引用

**使用示例：**
```python
# 需要搜索的查询
rounds, sources = run_search_agent("查询特斯拉的实时股价", max_rounds=3)
show_rounds(rounds, sources)

# 不需要搜索的查询
rounds, sources = run_search_agent("今天我心情真好")
show_rounds(rounds, sources)
```

#### `jina.py`
Jina.ai Search API 的独立测试脚本，用于验证搜索功能与 JSON 结构。

**功能：**
- 调用 Jina Search API（支持代理）
- 保存完整 JSON 响应到 `jina_answer.json`
- 集成本地模型推理（可选 RAG 流程）

**使用示例：**
```bash
python jina.py
# 将查询"如何种一朵玫瑰花"并保存结果到 jina_answer.json
```

### 辅助文件

#### `download_model.py`
自动下载 Qwen3-0.6B 模型到本地 `Qwen3-0.6B/` 目录。

**用法：**
```bash
python download_model.py
```

#### `sample.py`
简单测试脚本，验证 Qwen3-0.6B 模型加载与基本推理功能。

**用法：**
```bash
python sample.py
```

#### `jina_answer.json`
Jina API 返回的示例 JSON 文件，包含搜索结果的完整结构。

**JSON 结构：**
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

## 环境配置

### 依赖安装

```bash
pip install transformers torch requests accelerate
```

### API 配置

在 `search_agent.ipynb` 或 `jina.py` 中配置：

```python
JINA_API_KEY = "your_jina_api_key"
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}
```

### 模型路径

确保 Qwen3-0.6B 模型位于：
```
C:/Users/nashk/Documents/nashknight/search_agent/Qwen3-0.6B
```

或修改 `LOCAL_MODEL_PATH` 变量。

## 使用指南

### 1. 启动 Jupyter Notebook

```bash
jupyter notebook search_agent.ipynb
```

### 2. 依次执行单元

1. 导入依赖
2. 加载模型
3. 运行 Search Agent v2.0 core 单元
4. 执行测试用例

### 3. 自定义查询

```python
# 实时信息查询（会触发搜索）
rounds, sources = run_search_agent("查询特斯拉的实时股价")
show_rounds(rounds, sources)

# 多轮搜索示例
rounds, sources = run_search_agent("如何种植玫瑰花？需要什么工具？", max_rounds=5)
show_rounds(rounds, sources)
```

## 工作原理

### 搜索触发机制

模型在内部思考时，若判断需要外部信息，会生成 `<search>查询内容</search>` 标签：

```
<think>
用户问特斯拉股价，这需要实时数据。
<search>特斯拉实时股价</search>
</think>
```

### 多轮流程

1. **Round 1**: 模型生成 `<search>` → 调用 Jina API → 获取结构化信息
2. **Round 2**: 模型分析信息 → 判断是否充分 → 可能继续 `<search>` 或给出最终答案
3. **Round N**: 直到模型不再输出 `<search>` 标签

### 输出格式

```
=== Round 1 ===
🔍 search_content: 特斯拉实时股价
📚 搜索到 5 条信息:
  信息1: 特斯拉(TSLA)股价...
  
=== Round 2 ===
✅ Clean_responses (最终答案):
特斯拉当前股价为 XXX 美元...

📎 本次对话引用的来源 (共3个):
  [1] 标题1
      https://example.com/1
  [2] 标题2
      https://example.com/2
```

## 技术细节

### Prompt 设计

```python
BASE_PROMPT = (
    "你是一个搜索智能体。规则:"
    "\n1. 能直接回答时直接回答，不用<search>."
    "\n2. 需要实时事实时使用<search>查询短语</search>."
    "\n3. 触发关键词：实时、当前、最新、股价、报价..."
    "\n4. <search>内部只放纯查询短语，不放句子。"
    "\n5. 分析每个description，选取有用信息。"
    "\n6. 最终答案必须列出参考来源URL。"
)
```

### 信息结构

```python
# 搜索结果
{
  "sources": {
    "信息1": {
      "url": "https://...",
      "description": "...",
      "title": "..."
    }
  }
}

# URL 追踪
used_sources = {
  "https://url1": "标题1",
  "https://url2": "标题2"
}
```

## 注意事项

1. **代理配置**：如需访问外网，确保代理正常运行（默认 127.0.0.1:7890）
2. **API 配额**：Jina API 有 token 限制，注意控制 `max_sources` 参数
3. **模型限制**：Qwen3-0.6B 为轻量级模型，复杂推理能力有限
4. **Token 长度**：搜索结果会截断以防止超出模型上下文窗口

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- GitHub: Nash-Knight/search_agent