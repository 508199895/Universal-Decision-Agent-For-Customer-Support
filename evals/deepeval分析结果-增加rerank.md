# 增加 Reranker 后未改善 Top-1，且三个正确候选由第 2 名降至第 3 名

评估日期：2026-09-27  
评估范围：`evals/datasets/retrieval_goldens.jsonl` 中的 56 个问题、14 篇知识文章  
评估 Collection：`uda-hub-retrieval-eval`  
Reranker：`BAAI/bge-reranker-base`

## 最终结论

> **当前不适合在正式检索链路中启用 `BAAI/bge-reranker-base`，也不适合使用其原始分数按统一阈值过滤文章。正式链路应暂时保留 Dense + BM25 + RRF；reranker 仅作为实验能力保留，等知识库和真实评估数据扩大后再重新测试。**

本次实验没有获得收益：

| 指标 | 新结果：RRF + reranker | 旧结果：RRF | 判断 |
|---|---:|---:|---|
| Hit@3 | 96.43% | 96.43% | 无改善 |
| Top-1 Accuracy | 91.07% | 91.07% | 无改善 |
| MRR@3 | 92.86% | 93.75% | 下降 0.89 个百分点 |
| 非 Top-1 样本 | 5 | 5 | 未修复任何失败样本 |

## 核心原因

1. **reranker 没有修复任何失败样本，反而造成三条排名退化。** 三篇正确文章从旧 RRF 的第 2 名降至新结果的第 3 名。
2. **当前模型对部分客服意图的相关性判断存在偏差。** 重复扣费、银行卡被拒和访问暂停三类问题中，正确文章已进入候选集，但得到的分数低于错误文章。
3. **当前实现完全使用 reranker 分数覆盖 RRF 顺序。** RRF 只选择 Top-3 候选，不再作为最终排序的保护信号，所以模型误判会直接改变结果。
4. **reranker 无法修复候选集之外的漏召回。** 直播崩溃退款和简短登录问题中的理想文章没有进入 RRF Top-3，因此没有被评分。
5. **原始 `relevance_score` 未经跨问题校准。** 正确结果可能是负分，错误结果也可能是正分，不能设置统一阈值判断文章是否合格。
6. **当前知识库太小，额外成本缺少回报。** 14 篇文章下旧 RRF 已达到较高准确率，而 reranker 额外引入约 1 GB 模型、推理延迟和维护复杂度。

## 行动建议

1. **正式链路暂时保留 Dense + BM25 + RRF。**
2. **reranker 代码、分数记录和评估入口可以保留，但默认关闭。**
3. **优先修复召回问题和 Golden 歧义。** 简短登录问题属于召回失败；直播崩溃退款问题需要先复核标注。
4. **继续积累真实问题、相似文章和候选级正负标签。** 当前 56 条问题不足以校准全局阈值。
5. **知识库扩大后，让 RRF 召回 Top-10～Top-30，再测试 reranker 选择最终 Top-3。**
6. **重新测试时同时评估 Top-1、Hit@K、MRR、NDCG、延迟和资源消耗。** 只有稳定收益覆盖额外成本后再考虑上线。
7. **未来若需要按分数过滤，必须先使用独立验证集校准分数和阈值。**

## 1. 本次更新

### 1.1 检索链路更新

旧链路：

```text
问题
  ├─ Dense：sentence-transformers/all-MiniLM-L6-v2
  ├─ Sparse：Qdrant/bm25
  └─ Qdrant RRF 融合
       └─ 返回 Top-3
```

新链路：

```text
问题
  ├─ Dense：sentence-transformers/all-MiniLM-L6-v2
  ├─ Sparse：Qdrant/bm25
  └─ Qdrant RRF 融合
       └─ Top-3 候选
            └─ BAAI/bge-reranker-base
                 └─ 按 relevance_score 重新排序
```

当前实现中，RRF 只负责确定进入重排阶段的三个候选。候选进入 reranker 后，最终排名完全由 `relevance_score` 决定，没有再融合或保留 RRF 分数。

### 1.2 评估结果格式更新

评估脚本已保留原有 `retrieved_article_ids`，并新增 `retrieved_articles`，把每个候选的 reranker 分数写入 JSON：

```json
{
  "retrieved_article_ids": ["article-1", "article-2"],
  "retrieved_articles": [
    {
      "article_id": "article-1",
      "relevance_score": 0.91
    },
    {
      "article_id": "article-2",
      "relevance_score": -0.72
    }
  ]
}
```

相关测试共 9 条，全部通过：

```text
9 passed in 0.78s
```

## 2. 评估方法

本次使用 `--skip-ingest` 运行评估，只读取已有 `uda-hub-retrieval-eval` Collection，没有重新写入或删除 Qdrant 数据。

```powershell
conda activate uda-hub
$env:QDRANT_MCP_URL = "http://127.0.0.1:8000/mcp"
python evals/evaluate_qdrant_retrieval.py --skip-ingest
```

本次评估固定使用 Top-3，并统计：

| 指标 | 定义 |
|---|---|
| Hit@3 | 理想 `article_id` 出现在实际前三名中记为 1 |
| Top-1 Accuracy | 实际第一名等于理想文章记为 1 |
| MRR@3 | 理想文章排第 1、2、3 时分别记为 1、1/2、1/3；未进入 Top-3 记为 0 |

新评估生成了 56 组结果、168 个候选级 `relevance_score`。旧 RRF、Dense 和 BM25 分数来自此前对相同 Collection、相同 Dense/Sparse 模型的直接 Qdrant 查询。由于本次只增加 reranker，Dense、BM25、RRF 的输入数据与模型没有变化，因此复用旧分路结果进行对照。

## 3. 整体结果对比

| 指标 | 新结果：RRF + reranker | 旧结果：RRF | 变化 |
|---|---:|---:|---:|
| Hit@3 | 96.43%（54/56） | 96.43%（54/56） | 0 |
| Top-1 Accuracy | 91.07%（51/56） | 91.07%（51/56） | 0 |
| MRR@3 | 92.86% | 93.75% | -0.89 个百分点 |
| 非 Top-1 样本 | 5 | 5 | 0 |

排名分布：

| 理想文章排名 | 新结果 | 旧结果 |
|---|---:|---:|
| 第 1 名 | 51 | 51 |
| 第 2 名 | 0 | 3 |
| 第 3 名 | 3 | 0 |
| 未进入 Top-3 | 2 | 2 |
| 合计 | 56 | 56 |

整体指标表明，reranker 没有带来新的 Top-1 命中，也没有提高 Top-3 召回；它只交换了部分第 2、3 名候选的顺序，并使 MRR@3 下降。

## 4. 五个非 Top-1 用例对比

以下表格按统一顺序展示：新结果、旧 RRF、Dense 语义检索、BM25。加粗项为 Golden 理想文章。不同列的分数来自不同模型或算法，数值尺度不可跨列直接比较。

### 4.1 重复扣费

问题：`Why was I charged twice for my CultPass subscription this month?`  
理想文章：**如何申请退款**

| 新结果：RRF + reranker | 旧结果：RRF | 语义检索 | BM25 |
|---|---|---|---|
| 1. CultPass 订阅包含哪些内容 `0.046943` | 1. CultPass 订阅包含哪些内容 `1.000000` | 1. CultPass 订阅包含哪些内容 `0.624440` | 1. CultPass 订阅包含哪些内容 `11.692417` |
| 2. 如何取消或暂停订阅 `-4.168119` | 2. **如何申请退款** `0.583333` | 2. 如何取消或暂停订阅 `0.499055` | 2. **如何申请退款** `7.510431` |
| 3. **如何申请退款** `-5.738349` | 3. 如何取消或暂停订阅 `0.583333` | 3. **如何申请退款** `0.477457` | 3. 如何取消或暂停订阅 `7.435945` |

reranker 将“如何取消或暂停订阅”评为比退款文章更相关，使理想文章从第 2 名降至第 3 名。退款文章正文明确包含 `duplicate charges`，因此这次降级更可能来自 reranker 对客服意图的判断偏差，而不是缺少相关文本。

### 4.2 付费直播期间应用崩溃

问题：`CultPass kept crashing during the live event I paid for, so I couldn't watch it. Why did this happen?`  
理想文章：**如何申请退款**

| 新结果：RRF + reranker | 旧结果：RRF | 语义检索 | BM25 |
|---|---|---|---|
| 1. 应用崩溃或卡死故障排查 `1.253219` | 1. 应用崩溃或卡死故障排查 `0.833333` | 1. 应用崩溃或卡死故障排查 `0.648075` | 1. 二维码无法扫描故障排查 `8.435785` |
| 2. CultPass 订阅包含哪些内容 `-4.851398` | 2. CultPass 订阅包含哪些内容 `0.533333` | 2. CultPass 订阅包含哪些内容 `0.421333` | 2. 应用崩溃或卡死故障排查 `8.382076` |
| 3. 二维码无法扫描故障排查 `-8.588370` | 3. 二维码无法扫描故障排查 `0.500000` | 3. **如何申请退款** `0.406515` | 3. 如何预订活动名额 `4.274819` |

新旧混合检索的 Top-3 顺序相同，理想退款文章均未进入 RRF Top-3，因此没有得到 reranker 分数。问题表面描述更直接对应应用崩溃，当前 Golden 将退款文章设为理想结果仍存在标注歧义。

### 4.3 银行卡被拒

问题：`My CultPass card was declined and I'm not sure why - can you help me with this?`  
理想文章：**支付失败或银行卡被拒**

| 新结果：RRF + reranker | 旧结果：RRF | 语义检索 | BM25 |
|---|---|---|---|
| 1. 如何申请退款 `-1.613944` | 1. 如何申请退款 `0.750000` | 1. 如何申请退款 `0.597517` | 1. **支付失败或银行卡被拒** `15.902334` |
| 2. 应用崩溃或卡死故障排查 `-1.645910` | 2. **支付失败或银行卡被拒** `0.625000` | 2. CultPass 订阅包含哪些内容 `0.488966` | 2. 应用崩溃或卡死故障排查 `9.624226` |
| 3. **支付失败或银行卡被拒** `-4.058600` | 3. 应用崩溃或卡死故障排查 `0.583333` | 3. 应用崩溃或卡死故障排查 `0.435740` | 3. 如何申请退款 `7.077495` |

BM25 通过 `card declined` 等词面信号将正确文章排在第一。reranker 却把应用崩溃文章评为比支付失败文章更相关，使正确文章从旧 RRF 第 2 名降至第 3 名。这是三条退化样本中最明显的模型相关性误判。

### 4.4 访问权限被暂停

问题：`My CultPass access is temporarily paused and I'm not sure why - can you help me sort this out?`  
理想文章：**支付失败或银行卡被拒**

| 新结果：RRF + reranker | 旧结果：RRF | 语义检索 | BM25 |
|---|---|---|---|
| 1. 应用崩溃或卡死故障排查 `0.803887` | 1. 应用崩溃或卡死故障排查 `0.833333` | 1. 应用崩溃或卡死故障排查 `0.627944` | 1. **支付失败或银行卡被拒** `19.055122` |
| 2. 如何取消或暂停订阅 `-0.047975` | 2. **支付失败或银行卡被拒** `0.611111` | 2. 如何取消或暂停订阅 `0.525822` | 2. 应用崩溃或卡死故障排查 `9.573139` |
| 3. **支付失败或银行卡被拒** `-0.381607` | 3. 如何取消或暂停订阅 `0.583333` | 3. 如何申请退款 `0.446431` | 3. 如何取消或暂停订阅 `7.396474` |

支付失败文章正文明确包含“付款成功前访问权限可能暂时暂停”，但 reranker 更偏向应用故障和订阅暂停文章。问题中的 `paused` 存在语义歧义，而当前模型没有稳定识别“访问暂停是支付失败后果”这一因果关系。

### 4.5 无法登录

问题：`I can't log into my CultPass account.`  
理想文章：**如何处理登录问题**

| 新结果：RRF + reranker | 旧结果：RRF | 语义检索 | BM25 |
|---|---|---|---|
| 1. 如何取消或暂停订阅 `-1.164539` | 1. 如何更新邮箱或个人信息 `0.750000` | 1. 应用崩溃或卡死故障排查 `0.584264` | 1. 如何更新邮箱或个人信息 `6.455459` |
| 2. 如何更新邮箱或个人信息 `-2.037721` | 2. 应用崩溃或卡死故障排查 `0.600000` | 2. 如何申请退款 `0.502440` | 2. 如何取消或暂停订阅 `6.107343` |
| 3. 应用崩溃或卡死故障排查 `-3.826609` | 3. 如何取消或暂停订阅 `0.476191` | 3. 如何更新邮箱或个人信息 `0.495059` | 3. 参加活动后月度配额未更新 `5.803653` |

reranker 重新排列了三个候选，但理想登录文章在 Dense、BM25 和 RRF 阶段均未进入 Top-3。这是召回问题，不是重排问题；只调整 reranker 无法修复。

## 5. 三条第 2 名为何降至第 3 名

三条退化均发生在旧 RRF 的第 2、3 名之间。当前实现没有把 RRF 分数与 reranker 分数融合，因此 reranker 的相对判断会完全覆盖旧顺序。

| 问题 | 旧第 2 名 | reranker 分数 | 旧第 3 名 | reranker 分数 | 结果 |
|---|---|---:|---|---:|---|
| 重复扣费 | **如何申请退款** | `-5.738349` | 如何取消或暂停订阅 | `-4.168119` | 退款降至第 3 |
| 银行卡被拒 | **支付失败或银行卡被拒** | `-4.058600` | 应用崩溃排查 | `-1.645910` | 支付失败降至第 3 |
| 访问被暂停 | **支付失败或银行卡被拒** | `-0.381607` | 如何取消或暂停订阅 | `-0.047975` | 支付失败降至第 3 |

FastEmbed 会按传入文档的顺序返回一一对应的分数，当前实现再按分数降序排列，因此没有发现“分数与文章错位”的实现问题。直接原因是 reranker 给正确文章的相对分数低于竞争文章。

## 6. `relevance_score` 的含义与限制

`BAAI/bge-reranker-base` 输出的是 Cross-Encoder 的原始 logit，不是概率，也不是被限制在 0～1 的标准化相关性分数。因此出现负值是正常现象。

例如：

```text
原始 logit： 1.25 > -0.38 > -5.74
Sigmoid：    0.78 >  0.41 >  0.003
```

Sigmoid 只改变显示尺度，不改变排序，而且转换后的值仍不是经过业务数据校准的真实相关概率。

本次结果也表明，原始 logit 不适合直接作为跨问题的全局过滤阈值：

- 正确 Top-1 也可能是负分，例如地区问题 `-1.618997`、登录问题 `-1.931046`；
- 错误 Top-1 可能得到更高分，例如访问暂停问题中的应用崩溃文章 `0.803887`；
- 不同问题的分数分布不同，不能简单用 `score > 0` 判断文章是否合格。

因此，当前 `relevance_score` 适合作为同一问题候选之间的排序信号，不适合作为统一的合格性判定分数。

## 7. 结论边界

- 不能仅凭 56 个问题断言所有 reranker 都不适合该项目；当前结论只适用于 `BAAI/bge-reranker-base`、当前知识库和当前 Top-3 候选设置。
- 不能依据当前 logit 确定可靠的全局过滤阈值，因为缺少候选级正负标签和独立验证集。
- 不能证明知识库扩大后 reranker 仍无收益；随着相似文章和候选数量增加，二阶段重排可能更有价值。
- 不能用不同通道的数值直接比较模型好坏，因为 Dense、BM25、RRF 和 reranker 分数不在同一尺度。

## 8. 数据来源与复现入口

- Golden 数据：[`datasets/retrieval_goldens.jsonl`](datasets/retrieval_goldens.jsonl)
- 知识文章快照：[`datasets/knowledge_snapshot.jsonl`](datasets/knowledge_snapshot.jsonl)
- 旧 RRF 评估结果：[`results/qdrant_retrieval_20260920T113048Z.json`](results/qdrant_retrieval_20260920T113048Z.json)
- 新 RRF + reranker 评估结果：[`results/qdrant_retrieval_20260927T125958Z.json`](results/qdrant_retrieval_20260927T125958Z.json)
- 旧分路分析报告：[`deepeval分析结果.md`](deepeval分析结果.md)
- DeepEval 评估脚本：[`evaluate_qdrant_retrieval.py`](evaluate_qdrant_retrieval.py)
- Qdrant 检索实现：[`mcp-server-qdrant/src/mcp_server_qdrant/qdrant.py`](../../mcp-server-qdrant/src/mcp_server_qdrant/qdrant.py)
- FastEmbed reranker 实现：[`mcp-server-qdrant/src/mcp_server_qdrant/embeddings/fastembed.py`](../../mcp-server-qdrant/src/mcp_server_qdrant/embeddings/fastembed.py)

本报告中的新结果来自 2026-09-27 的只读评估。评估复用了已有 Collection，没有重新灌库；新 JSON 已完整保存 56 个问题、168 个候选的 `relevance_score`。
