# DeepEval 总结

DeepEval 的核心是由评估数据、执行轨迹、评分指标以及评估调度共同组成的闭环。

| 概念 | 核心职责 | 类比 |
|---|---|---|
| Golden | 描述输入和期望行为 | 考题与参考答案 |
| Trace | 记录真实执行过程 | 答题过程 |
| TestCase | 承载一次实际运行的数据 | 实际答卷 |
| Metric | 定义如何评分 | 评分标准 |
| `evaluate()` / `assert_test()` | 调度指标、汇总结果或触发失败 | 阅卷系统 |

完整流程如下：

```text
Golden
  ↓ 驱动系统运行
Trace
  ↓ 产生实际输入、输出、上下文、工具轨迹
TestCase
  ↓ 交给 Metric
Metric score + reason
  ↓
evaluate() 汇总分析
或 assert_test() 作为测试门禁
```

## 不同场景需要的组件

不是每次评估都需要使用所有组件。

| 场景 | 必要组件 |
|---|---|
| 单条输出快速评分 | TestCase + Metric |
| 批量离线评估 | TestCase 集合 + Metric + `evaluate()` |
| 回归测试门禁 | Golden/TestCase + Metric + `assert_test()` |
| Agent 轨迹评估 | Golden + Trace + 轨迹 Metric |
| 生产可观测性 | Trace，按需增加 Metric |
| 组件故障定位 | Trace + Span 级 Metric |

## 三个层次

```text
数据层
├── Golden
├── TestCase
└── Trace / Span

评分层
└── Metric

执行与报告层
├── evaluate
├── assert_test
├── evals_iterator
└── inspect / 报告平台
```

## 使用注意

- Trace 不是所有评估都必需。简单的输入输出评估可以直接构造 TestCase。
- Golden 也不是必需。无参考指标可以只使用实际输入和输出。
- 真正不可缺少的是可评估的数据和 Metric。
- `evaluate()` 与 `assert_test()` 主要提供工程化的调度、汇总、报告和门禁能力。

## 本质

> DeepEval = 标准化评估数据结构 + 现成或自定义评分器 + Trace 采集 + 批量执行、门禁和报告工具。

其底层仍然是数据采集、规则或 Judge LLM 评分、阈值比较与结果聚合。DeepEval 的主要价值，是把这些环节组织成统一、可复用的工程体系。
