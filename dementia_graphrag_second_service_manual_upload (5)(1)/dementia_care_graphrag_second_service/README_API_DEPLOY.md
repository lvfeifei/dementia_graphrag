# 失智老人居家照护 GraphRAG API 最小部署包说明

项目名称：dementia_care_graphrag_api_deploy_minimal

本部署包用于在服务器上部署“失智老人居家照护 GraphRAG 智能体”的 FastAPI 后端接口。小程序、网页前端或其他业务系统可以通过 HTTP API 调用本服务，获得照护问答、PDF 照护建议单和 ICS 日历提醒文件。

---

## 一、部署包定位

本部署包是最小 API 部署包，只包含 GraphRAG API 运行所需文件。

本部署包不包含 Neo4j 知识图谱重建文件，也不负责导入知识图谱。

默认前提是：

1. 服务器上已经部署好 Neo4j；
2. v02_final 失智照护知识图谱已经导入 Neo4j；
3. 部署人员已获得 Neo4j 的 Bolt 地址、用户名和密码；
4. 部署人员已准备好 DeepSeek API Key；
5. 小程序只调用本 API 服务，不直接连接 Neo4j。

推荐架构如下：

~~~text
小程序 / 前端
    ↓ HTTP / HTTPS
GraphRAG FastAPI 服务
    ↓ Bolt
Neo4j v02_final 失智照护知识图谱
    ↓ 本地文件
FAISS 向量索引 + corpus 语料
    ↓ API
DeepSeek 大模型服务
~~~

---

## 二、当前目录结构

~~~text
dementia_care_graphrag_api_deploy_minimal/
├── api_server.py
├── requirements.txt
├── data
│   └── corpus
│       ├── kg_advice_corpus.jsonl
│       ├── kg_corpus_summary.json
│       └── problem_catalog_v02_final.json
├── outputs
│   ├── answers
│   ├── calendars
│   ├── eval_results
│   ├── indexes
│   │   ├── advice_bgem3_embeddings.npy
│   │   ├── advice_bgem3_flatip.faiss
│   │   ├── advice_bgem3_meta.jsonl
│   │   ├── problem_bgem3_embeddings.npy
│   │   ├── problem_bgem3_flatip.faiss
│   │   ├── problem_bgem3_meta.jsonl
│   │   └── vector_index_summary.json
│   ├── logs
│   ├── pdfs
│   ├── retrieval_debug
│   └── sessions
└── src
    ├── 03_llm_client.py
    ├── 04_problem_mapper.py
    ├── 05_retrieve.py
    ├── 06_generate_answer.py
    ├── 07_session_memory.py
    ├── 08_graphrag_agent.py
    ├── 09_pdf_generator.py
    └── 10_calendar_generator.py
~~~

---

## 三、核心文件说明

### 1. api_server.py

api_server.py 是 FastAPI 服务入口，对外提供：

~~~text
GET  /health
POST /chat
POST /pdf
POST /ics
GET  /download/{kind}/{filename}
~~~

它会调用 src/08_graphrag_agent.py 执行完整 GraphRAG 问答流程。

### 2. requirements.txt

requirements.txt 是 Python 依赖文件。部署服务器需要先安装这些依赖，才能运行 API 服务。

### 3. data/corpus/

data/corpus/ 保存 GraphRAG 运行所需的本地语料。

其中：

- problem_catalog_v02_final.json：保存 58 个标准照护问题，用于问题映射。
- kg_advice_corpus.jsonl：保存从知识图谱导出的照护建议语料。
- kg_corpus_summary.json：保存语料统计摘要，用于版本核查和排错。

### 4. outputs/indexes/

outputs/indexes/ 保存已经构建好的 BGE-M3 + FAISS 向量索引。

其中：

- advice_bgem3_flatip.faiss 和 advice_bgem3_meta.jsonl：用于照护建议语义检索。
- problem_bgem3_flatip.faiss 和 problem_bgem3_meta.jsonl：用于标准 Problem 问题映射的候选召回。
- vector_index_summary.json：保存向量索引构建摘要。

该目录是 API 运行必需目录，不建议删除。

### 5. outputs/answers/

用于保存每次问答生成的回答文件。

运行后会自动生成：

~~~text
last_answer.md
last_answer_result.json
answer_<timestamp>.md
answer_result_<timestamp>.json
~~~

### 6. outputs/retrieval_debug/

用于保存每次检索的调试结果。

运行后会自动生成：

~~~text
last_retrieval_result.json
retrieval_result_<timestamp>.json
~~~

这些文件可用于查看本轮识别了哪些 Problem、检索了哪些 Advice、证据来源是什么。

### 7. outputs/pdfs/

用于保存 PDF 照护建议单。

调用 /pdf 后会生成：

~~~text
last_care_plan.pdf
dementia_care_plan_<timestamp>.pdf
~~~

### 8. outputs/calendars/

用于保存 ICS 日历提醒文件。

调用 /ics 后会生成：

~~~text
last_care_reminders.ics
dementia_care_reminders_<timestamp>.ics
~~~

### 9. outputs/sessions/

用于保存多轮会话记忆。

每个 session 会保存为一个 JSON 文件，例如：

~~~text
outputs/sessions/user_001.json
~~~

它记录用户历史提问、系统回答摘要、活跃照护问题和风险线索。

### 10. src/

src/ 是 GraphRAG 核心代码目录。

各文件作用如下：

~~~text
03_llm_client.py
封装 DeepSeek API 调用。

04_problem_mapper.py
将自然语言照护问题映射到标准 Problem ID。

05_retrieve.py
GraphRAG 检索核心，负责 Neo4j 图检索、corpus 兜底检索、关键词检索、向量检索和融合排序。

06_generate_answer.py
基于检索证据生成结构化照护回答。

07_session_memory.py
管理本地多轮会话记忆。

08_graphrag_agent.py
统一智能体主流程，串联记忆、检索、回答生成和输出保存。

09_pdf_generator.py
根据回答结果生成 PDF 照护建议单。

10_calendar_generator.py
根据检索结果生成 ICS 日历提醒文件。
~~~

---

## 四、部署前准备

### 1. 确认 Neo4j 已部署

本 API 服务不会导入知识图谱。部署前需要确认：

- v02_final 失智照护知识图谱已经导入 Neo4j；
- Neo4j 服务已启动；
- Bolt 端口可连接；
- 已获得 Neo4j 用户名和密码。

推荐使用独立 Neo4j 实例部署 v02_final 图谱，避免影响原有图谱。

### 2. 创建 .env 文件

在部署包根目录新建 .env 文件：

~~~bash
touch .env
~~~

然后写入以下内容，并根据服务器实际情况修改：

~~~env
# Neo4j v02_final graph service
NEO4J_URI=bolt://127.0.0.1:7688
NEO4J_USER=neo4j
NEO4J_LOCAL_PASSWORD=please_fill_neo4j_password

# DeepSeek API
DEEPSEEK_API_KEY=please_fill_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT=60
DEEPSEEK_MAX_RETRIES=3

# Embedding / FAISS
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=16
FAISS_INDEX_TYPE=FlatIP

# Optional API protection
# 如果填写了 API_AUTH_TOKEN，调用接口时需要在 header 中传入：
# X-API-Key: <API_AUTH_TOKEN>
API_AUTH_TOKEN=

# CORS
API_CORS_ORIGINS=*
~~~

其中最重要的是：

~~~env
NEO4J_URI=bolt://127.0.0.1:7688
NEO4J_USER=neo4j
NEO4J_LOCAL_PASSWORD=please_fill_neo4j_password
DEEPSEEK_API_KEY=please_fill_deepseek_api_key
~~~

如果电子院的 Neo4j 使用默认端口，NEO4J_URI 可能是：

~~~env
NEO4J_URI=bolt://127.0.0.1:7687
~~~

如果部署在远程服务器，也可以写成：

~~~env
NEO4J_URI=bolt://服务器IP:7688
~~~

---

## 五、安装依赖

建议先创建独立 Python 环境。

~~~bash
conda create -n dementia_graphrag_api python=3.10 -y
conda activate dementia_graphrag_api
~~~

进入部署包目录：

~~~bash
cd dementia_care_graphrag_api_deploy_minimal
~~~

安装依赖：

~~~bash
python -m pip install -r requirements.txt
~~~

如果 FAISS 安装失败，可以使用 conda 安装：

~~~bash
conda install -c conda-forge faiss-cpu -y
~~~

---

## 六、启动 API 服务

本地测试启动：

~~~bash
uvicorn api_server:app --host 127.0.0.1 --port 8000
~~~

服务器部署启动：

~~~bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
~~~

启动成功后会看到类似：

~~~text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
~~~

浏览器打开：

~~~text
http://127.0.0.1:8000/docs
~~~

可以查看自动生成的 API 文档。

---

## 七、接口说明

### 1. 健康检查接口

请求：

~~~text
GET /health
~~~

示例：

~~~bash
curl http://127.0.0.1:8000/health
~~~

如果 success=true，说明核心文件存在，API 服务基本可用。

### 2. 问答接口

请求：

~~~text
POST /chat
~~~

请求示例：

~~~bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user_001","question":"奶奶晚上总起来，还说有人偷她的钱，怎么办？"}'
~~~

常用请求参数：

~~~json
{
  "session_id": "user_001",
  "question": "奶奶晚上总起来，还说有人偷她的钱，怎么办？",
  "reset_session": false,
  "memory_context_mode": "auto",
  "include_active_problem_ids": false,
  "temperature": 0.2,
  "max_tokens": 3200
}
~~~

小程序主要使用返回中的：

~~~text
answer_md
selected_problem_ids
evidence_sources
retrieval_counts
paths
~~~

### 3. PDF 生成接口

请求：

~~~text
POST /pdf
~~~

请求示例：

~~~bash
curl -X POST "http://127.0.0.1:8000/pdf" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user_001"}'
~~~

返回中会包含：

~~~text
timestamp_pdf_path
last_pdf_path
download_url
~~~

### 4. ICS 日历提醒接口

请求：

~~~text
POST /ics
~~~

请求示例：

~~~bash
curl -X POST "http://127.0.0.1:8000/ics" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user_001","days":7}'
~~~

返回中会包含：

~~~text
timestamp_ics_path
last_ics_path
download_url
event_count
events
~~~

如果 success=false，且 error 提示没有识别到可生成的提醒，一般不是系统错误，而是说明本轮问题不适合生成日历提醒。

### 5. 文件下载接口

PDF 下载：

~~~text
GET /download/pdfs/<filename>
~~~

示例：

~~~bash
curl -L "http://127.0.0.1:8000/download/pdfs/dementia_care_plan_xxxxxx.pdf" -o care_plan.pdf
~~~

ICS 下载：

~~~text
GET /download/calendars/<filename>
~~~

示例：

~~~bash
curl -L "http://127.0.0.1:8000/download/calendars/dementia_care_reminders_xxxxxx.ics" -o reminders.ics
~~~

---

## 八、小程序推荐调用流程

推荐小程序调用流程如下：

1. 用户输入照护问题；
2. 小程序调用 POST /chat；
3. 页面展示 answer_md；
4. 用户点击“生成照护建议单”；
5. 小程序调用 POST /pdf；
6. 小程序根据 download_url 下载 PDF；
7. 用户点击“生成日历提醒”；
8. 小程序调用 POST /ics；
9. 小程序根据 download_url 下载 ICS。

小程序不应直接连接 Neo4j，也不应保存 Neo4j 密码或 DeepSeek API Key。

---

## 九、安全与权限建议

### 1. API Token

可以在 .env 中设置：

~~~env
API_AUTH_TOKEN=please_fill_a_random_long_token
~~~

设置后，调用接口时需要加：

~~~text
X-API-Key: please_fill_a_random_long_token
~~~

示例：

~~~bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: please_fill_a_random_long_token" \
  -d '{"session_id":"user_001","question":"奶奶晚上总起来，还说有人偷她的钱，怎么办？"}'
~~~

### 2. 不要暴露敏感信息

不要把以下内容写入小程序前端：

- Neo4j 用户名；
- Neo4j 密码；
- DeepSeek API Key；
- 服务器 .env 文件；
- API_AUTH_TOKEN。

这些内容只应保存在服务器后端。

---

## 十、常见问题排查

### 1. /health 返回 success=false

可能原因：

- 核心代码文件缺失；
- data/corpus 缺失；
- outputs/indexes 缺失；
- 部署包没有完整复制。

处理方式：

- 检查部署包目录结构；
- 确认 outputs/indexes 中 FAISS 和 meta 文件存在。

### 2. /chat 返回 500

可能原因：

- Neo4j 没启动；
- NEO4J_URI 配置错误；
- Neo4j 密码错误；
- DeepSeek API Key 错误；
- outputs/indexes 缺失；
- 网络无法访问 DeepSeek。

处理方式：

- 检查 .env；
- 检查 Neo4j 是否可连接；
- 检查 DeepSeek API Key；
- 查看 API 服务终端日志。

### 3. PDF 生成失败

可能原因：

- 还没有调用过 /chat；
- 没有生成 answer_result JSON；
- outputs/answers 目录不可写；
- reportlab 依赖未安装。

处理方式：

- 先调用 /chat；
- 确认 outputs/answers 下有 answer_result 文件；
- 检查 requirements 是否安装完整。

### 4. ICS 生成失败

可能原因：

- 本轮问题不适合生成日历提醒；
- 检索结果中没有明确时间性任务；
- 还没有调用过 /chat。

处理方式：

- 先调用 /chat；
- 换一个更适合生成提醒的问题，例如睡眠、夜间起床、外跑、作息安排、环境检查等问题。

---

## 十一、运行边界说明

本 API 服务用于失智老人居家照护教育支持。

系统回答不能替代：

- 医生诊断；
- 护士专业评估；
- 记忆门诊判断；
- 急诊处理；
- 线下护理服务。

如老人出现以下情况，应及时寻求专业帮助：

- 走失；
- 跌倒；
- 噎食；
- 严重呛咳；
- 误服；
- 自伤；
- 伤人；
- 意识改变；
- 持续严重幻觉或妄想；
- 明显攻击行为。

---

## 十二、部署包总结

dementia_care_graphrag_api_deploy_minimal 是一个面向小程序后端部署的 GraphRAG API 服务包。

它不包含 Neo4j 知识图谱重建文件，只包含 API 运行所需代码、corpus 语料、FAISS 索引、输出目录和依赖文件。

部署后，小程序可通过 /chat、/pdf、/ics 和 /download 接口完成照护问答、PDF 建议单生成和日历提醒生成。
