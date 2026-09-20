# Qdrant MCP 重排能力调研

调研日期：2026-09-17  
资料范围：仅使用 Qdrant 官方仓库源码、README、Qdrant 官方文档与 API 文档。

## 结论

是的，当前官方 `qdrant/mcp-server-qdrant` 对 MCP 客户端只暴露两个业务工具：`qdrant-store` 和 `qdrant-find`，没有独立的 rerank/重排工具或配置项。[README 的 Tools 清单](https://github.com/qdrant/mcp-server-qdrant#tools)只列出这两个工具；[`mcp_server.py`](https://github.com/qdrant/mcp-server-qdrant/blob/master/src/mcp_server_qdrant/mcp_server.py#L77-L149) 中的 `setup_tools()` 也只注册存储与查询流程。

当前 `qdrant-find` 是单路 dense 向量检索，不包含 RRF、ColBERT 或 cross-encoder 重排：它先用一个 embedding provider 生成单个查询向量，再调用一次 `query_points(query=query_vector, using=vector_name, limit=...)`。[`qdrant.py` 的查询实现](https://github.com/qdrant/mcp-server-qdrant/blob/master/src/mcp_server_qdrant/qdrant.py#L85-L127)没有使用 `prefetch`、`FusionQuery` 或第二阶段 scorer；默认的 [`FastEmbedProvider`](https://github.com/qdrant/mcp-server-qdrant/blob/master/src/mcp_server_qdrant/embeddings/fastembed.py#L8-L46)也只使用 `TextEmbedding`。其 embedding 抽象接口只表达“文档 dense 向量、查询 dense 向量、向量名、维度”，没有 sparse、multivector 或 reranker 接口。[`EmbeddingProvider` 源码](https://github.com/qdrant/mcp-server-qdrant/blob/master/src/mcp_server_qdrant/embeddings/base.py#L3-L24)

不过，“官方 MCP server 没内置”不等于“Qdrant 不支持”。可以扩展该 MCP server，主要有三条路线：

1. 用 Query API 的 `prefetch + RRF` 做 dense/sparse 混合检索与名次融合；
2. 用 `prefetch + ColBERT multivector` 在 Qdrant 内做 late-interaction 二阶段重排；
3. 先由 Qdrant 召回候选，再在 MCP 服务的应用层调用 FastEmbed cross-encoder 重排。

需要区分：**RRF 是多个候选列表的排名融合，不是读取 query-document 文本并重新判断相关性的模型 reranker。**

## 方案一：Query API `prefetch + RRF`

Qdrant Query API 自 v1.10.0 起支持混合查询和多阶段查询。存在 `prefetch` 时，Qdrant 先执行一个或多个预取查询，再在这些候选上执行顶层 `query`；`prefetch` 还可以嵌套。[Hybrid and Multi-Stage Queries](https://qdrant.tech/documentation/search/hybrid-queries/)；[Query points API](https://api.qdrant.tech/master/api-reference/search/query-points)

如果同时保存 dense 与 sparse named vectors，可以各自召回候选，再用 RRF 合并：

```python
results = await client.query_points(
    collection_name="docs",
    prefetch=[
        models.Prefetch(
            query=sparse_query_vector,
            using="sparse",
            limit=50,
        ),
        models.Prefetch(
            query=dense_query_vector,
            using="dense",
            limit=50,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
    with_payload=True,
)
```

官方示例同样使用两个 `prefetch`，再用顶层 RRF 取最终 Top-K。[RRF Python 示例](https://qdrant.tech/documentation/search/hybrid-queries/#reciprocal-rank-fusion-rrf) Qdrant 对 RRF 的定义是按每个候选在各结果列表中的名次计算融合分数，而不是直接混合原始 dense/sparse 分数；这正适合两种检索器分数尺度不同的情况。[RRF 说明与公式](https://qdrant.tech/documentation/search/hybrid-queries/#reciprocal-rank-fusion-rrf)

这条路线会扩大召回面并改善融合顺序，但不是 cross-encoder 意义上的语义精排。如果 Collection 是多 shard，为做全局融合，应把 fusion 放在顶层 `query`；放在 `prefetch` 内只会先在各 shard 内局部融合。[Fusion in Distributed Collections](https://qdrant.tech/documentation/search/hybrid-queries/#fusion-in-distributed-collections)

对当前 MCP server 的影响：需要把 Collection 从单一 dense named vector 扩展为至少 dense+sparse 两种表示；写入时生成两类向量；查询时把现有单次 `query_points` 改成包含多个 `prefetch` 和 RRF 顶层查询。

## 方案二：Qdrant 内部 ColBERT late-interaction 重排

Qdrant 支持用 Query API 先粗召回，再以 multivector 进行第二阶段评分。官方多阶段示例是先预取 100 个 dense 候选，再用名为 `colbert` 的 multi-vector 取最终 10 个结果。[Multi-stage queries 示例](https://qdrant.tech/documentation/search/hybrid-queries/#multi-stage-queries)

也可以先并行执行 dense 与 sparse 召回，然后直接以 ColBERT late-interaction 作为最终排名信号：

```python
results = await client.query_points(
    collection_name="docs",
    prefetch=[
        models.Prefetch(query=dense_query, using="dense", limit=20),
        models.Prefetch(query=sparse_query, using="sparse", limit=20),
    ],
    query=colbert_query_multivector,
    using="multi",
    with_payload=True,
    limit=10,
)
```

Qdrant 官方教程明确将这一流程描述为：dense 与 sparse 各取候选，再由 ColBERT late-interaction 模型重排合并后的候选。[Hybrid Search with Reranking：Rerank](https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/#rerank)

这是一种真正的模型化二阶段重排，并且评分在 Qdrant Query API 内完成，但代价是：

- 文档写入时必须额外生成并存储 ColBERT multivector；
- Collection 必须配置 multivector 与 `MAX_SIM`；
- late-interaction 向量占用更多存储，计算成本也高于简单融合；
- 当前 MCP server 的单 dense `EmbeddingProvider` 接口和自动建表逻辑不足以直接承载，需同时改写 embedding、Collection schema、写入和查询链路。

Qdrant 官方 multivector 教程给出的 rerank-only 配置使用 `MultiVectorComparator.MAX_SIM`，并可将该向量的 HNSW `m` 设为 `0`，因为它只用于对预取候选评分，不需要独立执行 HNSW 首检。[Multivector Representations for Reranking](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/)

## 方案三：应用层 external cross-encoder reranker

这是对当前 MCP server 改动最小、也最容易先验证收益的方案：

```text
qdrant-find
  -> dense 检索 Top-N（例如 30～50）
  -> 取 payload 中的 document 文本
  -> cross-encoder 对 query-document 对逐一打分
  -> 按新分数降序排列
  -> 返回 Top-K（例如 5～10）
```

Qdrant 官方 FastEmbed 教程将 reranker 定义为：先用 BM25 或 dense embedding 快速召回较小候选集，再用更精确但更慢、更重的模型重新评估；因此应只在有限候选集上执行。[Reranking with FastEmbed：Rerankers](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/#rerankers)

官方示例使用 `TextCrossEncoder`，先让 Qdrant 返回 Top-10 文本，再执行 `reranker.rerank(query, description_hits)` 并在客户端排序。[First-stage retrieval 与 rerank 示例](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/#first-stage-retrieval)

在当前服务中可以沿用相同结构：

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder

reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")

# 现有 Qdrant dense 检索扩大候选数
hits = await retrieve_candidates(query, limit=50)
documents = [hit.content for hit in hits]

scores = list(reranker.rerank(query, documents))
reranked = [
    hit
    for hit, _ in sorted(
        zip(hits, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:10]
]
```

cross-encoder 不在 Qdrant 数据库进程内部执行，而是在 MCP server/应用层对 Qdrant 返回的候选文本评分。它不要求重建现有 Collection 或额外存储 multivector，适合先做小范围改造；代价是增加模型推理延迟，并且必须保留足够大的首阶段候选集。官方教程列出的可用模型包含 `BAAI/bge-reranker-base`、多种 MiniLM 与 `jinaai/jina-reranker-v2-base-multilingual`；具体模型与许可证应在部署前核对。[FastEmbed 支持模型与示例](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/#setup)

## 针对当前项目的建议

| 目标 | 建议方案 | 原因 |
|---|---|---|
| 最小改动、快速验证重排收益 | 应用层 cross-encoder | 不改 Collection schema；只需扩大初检 Top-N、加 reranker、截取最终 Top-K |
| 同时解决中文语义与编号/专有词召回 | dense+sparse `prefetch` + RRF | 混合召回，避免只靠 dense；但 RRF 本身不等于模型精排 |
| 追求更高精度且能接受存储/计算成本 | dense+sparse 召回 + ColBERT late interaction | Qdrant 内一次 Query API 完成候选召回和模型化重排 |

推荐迭代顺序：

1. 先保留现有 dense Collection，在 `qdrant-find` 内增加 external cross-encoder：初检 Top-30～50，重排后返回 Top-5～10。
2. 用固定评估集比较重排前后的 MRR/NDCG、Recall@K 与 P95 延迟。
3. 如果主要问题是 dense 漏掉错误码、产品型号或精确术语，再升级 Collection 为 dense+sparse 并加入 RRF。
4. 只有 cross-encoder 延迟或部署方式不合适、且能承担重建索引与 multivector 存储时，再评估 ColBERT server-side late interaction。

无论哪种重排，都只能重排候选集，无法找回首阶段完全没有召回的文档。因此应同时监测初检 Recall@N 与重排后的排序指标，而不能只看最终 Top-K。

