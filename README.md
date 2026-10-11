# 糖尿病出院患者智能随访助手

> 基于 **LangGraph + FastAPI + MySQL + Milvus + Redis + DeepSeek** 的医疗随访系统。
> 由两个可独立运行、又通过 **MCP 协议** 协作的子项目组成：医学知识 **RAG 问答服务** 与 **双 Agent 随访服务**。
> 内置 **JWT 认证**、**会话持久化**、**高频问答快车道**、**BERT 意图识别**、**检索策略选择**、**父子块检索**、**Human-in-the-Loop 人工介入（含医生接管式连续会话）**、**SSE 流式对话**、**结构化日志**、**Docker 一键部署**、**单元/集成测试**。

---

## 一、项目背景

糖尿病患者出院后缺乏持续、个性化指导；出院小结 / 病历专业性强，患者难以理解；上报血糖后不知数值好坏、是否需要就医。医疗问答容错成本极高，AI 答错后果严重，必须有人工兜底。

本项目通过 **两个显式入口** 将患者请求分发给两个 Agent，并用 **Human-in-the-Loop** 保证安全边界。

```
患者 App
   │
   ├─ 「我要咨询」 ──► 问答 Agent   RAG（公共指南 + 个人出院小结/病例 双源混合检索）
   │                    拿不准 / 高危 / 患者主动说「我要找医生」→ interrupt() 暂停 → 医生接管
   │                    接管后：患者消息直达医生（可多轮），医生「结束诊断」→ 自动恢复 AI
   │
   └─ 「记录血糖」 ──► 血糖管理 Agent  确定性计算（eAG / HbA1c / TIR / TAR / TBR / CV）+ 高低血糖告警
```

---

## 二、子项目

| 子项目 | 目录 | 职责 | 对外能力 |
|-------|------|------|---------|
| RAG 知识问答服务 | [rag_service](./rag_service) | 医学指南解析、混合检索、重排、RAG 生成、FAQ 快车道 | REST API + **MCP Server** |
| 双 Agent 随访服务 | [agent_service](./agent_service) | LangGraph 双 Agent 编排、JWT 认证、入口分流、人工介入、文档管理 | REST API + 静态前端 |

两个项目通过 **MCP 协议**协作：Agent 服务的问答 Agent 作为 MCP 客户端，调用 RAG 服务的 `search_knowledge_base` 工具完成知识检索。

---

## 三、整体架构

```
┌──────────────────────────────────────────────────────────┐
│                    前端（static/index.html）              │
│      对话流（患者）+ 会话工作台（医生）+ 血糖 / 病历        │
└──────────────────────────┬───────────────────────────────┘
                           │ HTTP（JWT Bearer Token + mode 路由）
┌──────────────────────────▼───────────────────────────────┐
│               Agent 服务 (FastAPI :8000)                  │
│   认证模块(JWT)                                          │
│   问答 Agent (LangGraph)        血糖管理 Agent (LangGraph)│
│   retrieve → route → generate    record → interpret       │
│        │ interrupt() 转人工 ──► 医生接管 / 「结束诊断」      │
└──────────┬───────────────────────────────────────────────┘
           │ MCP (Streamable HTTP / JSON-RPC) → :8001/rag/mcp
┌──────────▼───────────────────────────┐
│   RAG 服务 (FastAPI :8001)            │
│   · REST API                          │
│   · MCP Server（挂载于 /rag/mcp）      │
│   FAQ 快车道 → BERT 意图识别 →         │
│   检索策略 → BGE-M3 稠密+稀疏融合 →     │
│   bge-reranker 精排 → DeepSeek 生成    │
│   Redis 缓存                           │
│             │                         │
│   Milvus（公共指南向量库）              │
└───────────────────────────────────────┘
      MySQL（患者档案 / 血糖记录 / 会话历史 / 人工介入 /
             人工接管会话 human_sessions + human_messages / 个人文档 / FAQ）
      + Redis（结果缓存）
```

---

## 四、技术栈

| 层面 | 选型 |
|------|------|
| Web 框架 | FastAPI + SSE-Starlette |
| 认证 | JWT（PyJWT）+ bcrypt 密码哈希 |
| Agent 框架 | LangGraph（StateGraph + MemorySaver + interrupt 人工介入 + 医生接管式连续会话） |
| 人工介入 | 单次转人工（`hitl_pending`）+ 接管式连续会话（`human_sessions` / `human_messages`），支持主动转人工关键词 |
| RAG 检索 | BGE-M3 语义向量 + 稀疏向量（RRF / WeightedRanker 融合）+ bge-reranker 重排 |
| 文档解析 | PDF / Word / Markdown / TXT + OCR，父子块切分（父块供上下文、子块供检索） |
| 意图识别 | 自训练 BERT 分类器（通用知识 / 专业咨询）+ 领域关键词兜底 |
| 检索策略 | LLM 策略选择（直接检索 / HyDE / 子查询 / 回溯） |
| 高频快车道 | MySQL 问答表 + BM25 打分 + Redis 缓存，命中直返标准答案 |
| 大模型 | DeepSeek（OpenAI 兼容接口） |
| 向量库 | Milvus |
| 关系库 | MySQL 8.0（Agent 用 SQLAlchemy 异步 aiomysql；RAG 用 PyMySQL 同步） |
| 缓存 | Redis（问答结果缓存） |
| MCP | FastMCP Server ＋ JSON-RPC 客户端（tenacity 重试） |
| 可观测性 | 结构化 JSON 日志（含 trace_id）+ tenacity 重试 |
| 工程化 | Dockerfile × 2 + docker-compose 一键部署 + pytest |

---

## 五、快速开始

### 0) 环境要求

- Python 3.11 / 3.12
- Docker & Docker Compose（中间件）
- DeepSeek API Key
- 本地模型权重：BGE-M3、bge-reranker-large、bert-base-chinese（见第九节）

### 1) 启动中间件

```bash
# 在 my_project 根目录
docker compose up -d mysql redis etcd minio milvus
# MySQL(3307) / Redis(6380) / Milvus(19531) / etcd(2379) / MinIO(9002、9003)
```

### 2) 启动 RAG 服务

```bash
cd rag_service
pip install -r requirements.txt
cp .env.example .env                       # 配置 DEEPSEEK_API_KEY、模型路径、MySQL / Redis 等

# 导入知识库（公共指南 PDF → 解析 → 分块 → 向量化 → Milvus）
python -m app.ingestion.build --dir ./data

# 导入高频问答对（CSV/TXT → faq_qa 表，供快车道使用）
python scripts/import_faq.py --file ./data/faq_qa.csv

# 启动 RAG 服务（REST 与 MCP 同进程，无需另起；MCP 端点为 /rag/mcp）
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

> Agent 的 `RAG_MCP_SERVER_URL=http://localhost:8001/rag`，即调用 8001 上的 MCP。
> `python -m app.mcp.server`（独立 8002 端口）为可选的单独 MCP 进程，本项目默认不需要。

### 3) 启动 Agent 服务

```bash
cd ../agent_service
pip install -r requirements.txt
cp .env.example .env                       # 配置 JWT_SECRET、MySQL、RAG MCP 地址
python scripts/seed_patient.py             # 写入示例患者 / 医生 / 血糖记录

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4) 访问

| 地址 | 说明 |
|------|------|
| http://localhost:8000 | 患者 / 医生演示前端 |
| http://localhost:8000/docs | Agent 服务接口文档 |
| http://localhost:8001/docs | RAG 服务接口文档 |
| http://localhost:8001/rag/mcp | RAG 服务 MCP 端点（Agent 调用） |

---

## 六、认证与 API 使用

所有业务接口需携带 **Bearer Token**。

```bash
# 1) 注册（role=patient / doctor，注册成功即返回 token，注册即登录）
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"patient_001","password":"123456","name":"张三","role":"patient"}'

# 2) 登录
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"patient_001","password":"123456"}'

# 3) 携带令牌调用问答接口
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{"question":"出院后能吃什么水果？"}'
```

### Agent 服务主要接口

| 方法 | 路径 | 说明 | 鉴权 |
|------|------|------|------|
| POST | /api/v1/auth/register | 注册患者/医生（返回 token） | 否 |
| POST | /api/v1/auth/login | 登录换令牌 | 否 |
| POST | /api/v1/auth/refresh | 刷新令牌 | 否 |
| POST | /api/v1/chat | 咨询问答（入口分流；命中「找医生 / 转人工」等关键词则直接转人工） | 患者 |
| POST | /api/v1/chat/stream | 咨询问答（SSE 流式：status / meta / token / interrupted） | 患者 |
| POST | /api/v1/chat/unified/stream | 统一聊天入口（SSE，自动路由 qa / glucose / clarify / 人工） | 患者 |
| GET  | /api/v1/chat/hitl/{thread_id} | 患者端轮询单次人工处理结果 | 患者 |
| POST | /api/v1/glucose | 记录血糖（确定性计算 + 告警） | 患者 |
| POST | /api/v1/documents | 上传病例 / 出院小结（解析为纯文本入库） | 患者 |
| GET  | /api/v1/documents | 查看个人文档列表 | 患者 |
| DELETE | /api/v1/documents/{doc_id} | 删除个人文档 | 患者 |
| GET  | /api/v1/hitl/pending | 待人工介入列表（尚未接管） | 医生 |
| POST | /api/v1/hitl/resume | 医生首次回复：恢复图并**自动接管** | 医生 |
| GET  | /api/v1/hitl/sessions | 进行中的人工会话列表（含完整对话） | 医生 |
| POST | /api/v1/hitl/session/reply | 医生回复患者（接管后多轮） | 医生 |
| POST | /api/v1/hitl/session/close | 结束诊断，患者恢复 AI 问答 | 医生 |
| GET  | /api/v1/hitl/session/me | 患者端轮询人工会话（是否接管 / 已结束 + 对话） | 患者 |

### RAG 服务主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/rag/query | 非流式问答（快车道 → RAG → 兜底三态） |
| POST | /api/v1/rag/query/stream | SSE 流式问答 |
| POST | /api/v1/rag/session | 创建会话 ID |
| GET  | /api/v1/rag/history/{session_id} | 读取会话历史 |
| DELETE | /api/v1/rag/history/{session_id} | 清除会话历史 |
| POST | /api/v1/kb/build | 知识库重建 |
| GET  | /api/v1/kb/count | 向量库文档数 |
| MCP | /rag/mcp | `search_knowledge_base` 工具（Agent 调用） |

---

## 七、测试

```bash
# Agent 服务（单元测试，零依赖）
cd agent_service
pytest tests/test_calculator.py tests/test_auth.py

# Agent 服务（接口集成测试，需 MySQL 已就绪）
pytest tests/test_api.py

# RAG 服务（冒烟测试，无需 Milvus / 模型 / LLM）
cd ../rag_service
pytest tests/test_health.py
```

---

## 八、Docker 一键部署

```bash
# 在 my_project 根目录
docker compose up -d --build
```

- `rag_service` 容器需挂载本地模型权重（`./rag_service/models`），见下节。
- 生产环境请修改 `agent_service/.env` 中的 `JWT_SECRET` 与数据库密码。

> 说明：默认开发模式下，**基础设施（MySQL / Redis / Milvus / etcd / MinIO）跑在 Docker**；
> **业务服务（RAG / Agent）在本地 conda 环境启动**，避免容器内加载 GB 级模型权重的额外配置。
> 两种模式网络配置不同，请勿混用。

---

## 九、本地模型权重

| 模型 | 用途 | 体积 |
|------|------|------|
| bge-m3 | 稠密 + 稀疏双路向量编码 | GB 级 |
| bge-reranker-large | 精排 | GB 级 |
| bert-base-chinese | 查询分类器底座 / 分词器 | 中 |
| bert_query_classifier | **本项目自训练**的意图分类器 | 小 |
| nlp_bert_document-segmentation_chinese-base | 语义分块（可选，默认关闭） | 中 |

下载好 bge-m3 / bge-reranker-large 后，放到 `rag_service/models/`，并在 `rag_service/.env` 中配置：

```bash
# rag_service/.env
BGE_M3_MODEL_PATH=./models/bge-m3
RERANKER_MODEL_PATH=./models/bge-reranker-large
```

### 训练意图分类器

`bert_query_classifier` 由本项目自建语料微调而来（非外部拷入），复现方式：

```bash
cd rag_service
python scripts/train_classifier.py --data ./data/classifier_train.jsonl
```

---

## 十、工程化亮点

- **MySQL 迁移适配**：PostgreSQL 的 `ADD COLUMN IF NOT EXISTS` 语法 MySQL 不支持；改为先查 `information_schema` 判断存在性再执行 DDL，保持幂等。
- **人工介入双模式**：
  - 单次转人工：`interrupt()` 暂停图 → `hitl_pending` 表持久化 → 医生端轮询 → `Command(resume)` 恢复，不重跑前置节点；
  - **接管式连续会话**：医生首次回复即自动接管，之后患者咨询消息**直达医生**（可多轮），医生「结束诊断」后自动恢复 AI。会话与对话落 `human_sessions` / `human_messages`，跨进程、重启不丢。
- **主动转人工**：患者消息命中「找医生 / 转人工 / 人工客服」等关键词，跳过检索直接转人工（实测 < 0.1s）。
- **流式体验**：问答走 SSE 逐 token 输出；检索阶段先推送 `status` 事件（「正在检索知识库…」），避免用户面对空白干等。
- **双源检索**：公共指南（Milvus）+ 个人出院小结/病例（父子块，纯文本）合并后统一重排。
- **单例缓存**：BGE-M3 编码器、重排器、MySQL 客户端、FAQ 检索器均单例，避免重复加载。
- **检索性能调优**：`HYBRID_CANDIDATE_K` 控制重排候选数（CPU 环境下调至 10，检索延迟约减半、置信度不变）。

---

## 十一、简历一句话描述

> 糖尿病出院患者智能随访助手：基于 LangGraph + FastAPI，通过「咨询问答 / 记录血糖」两个显式入口将患者请求分发至「问答 Agent」（公共指南 + 个人出院小结双源混合检索 RAG，含 FAQ 快车道、BERT 意图识别与检索策略选择，拿不准 / 高危 / 患者主动要求时 interrupt() 暂停图转人工，医生接管后可连续多轮问答、结束时自动恢复 AI，前端 SSE 流式对话）与「血糖管理 Agent」（血糖记录、eAG / HbA1c / TIR / CV 确定性计算、高低血糖主动告警）；通过 MCP 协议实现 Agent 与 RAG 知识服务的解耦协作，内置 JWT 认证、MySQL 会话持久化、结构化日志、Docker 一键部署与单元/集成测试。
