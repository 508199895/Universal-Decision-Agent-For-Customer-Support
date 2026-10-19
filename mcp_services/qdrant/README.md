# 项目内 Qdrant MCP 服务

本目录保留上游 `mcp-server-qdrant` 的运行代码及 Apache-2.0 许可证；测试、开发命令和容器构建文件不随服务部署。

项目通过 `agentic/tools/mcp_client.py` 以 stdio 启动 `server.py`。服务连接现有 Qdrant 数据库，使用项目评估过的 Dense + BM25 + RRF，并通过 `BAAI/bge-reranker-base` 对结果重排。

在项目根目录 `.env` 配置：

```dotenv
QDRANT_URL=http://127.0.0.1:6333
COLLECTION_NAME=uda-hub-retrieval-eval
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SPARSE_EMBEDDING_MODEL=Qdrant/bm25
QDRANT_SEARCH_LIMIT=3
```

如 Qdrant 服务需要认证，另加 `QDRANT_API_KEY`。可通过 `RERANKER_MODEL` 指定其他 FastEmbed 重排模型。

运行依赖列于项目根目录 `requirements.txt`。服务启动时若没有显式设置 `FASTEMBED_CACHE_PATH`，会使用 `mcp_services/qdrant/.cache/fastembed`，模型缓存不会进入 Git。当前缓存包含 MiniLM、BM25 和 BAAI reranker。可通过项目 `.env` 中的 `FASTEMBED_CACHE_PATH` 覆盖默认目录。
