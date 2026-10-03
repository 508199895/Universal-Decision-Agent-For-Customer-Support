from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from deepeval.test_case import LLMTestCase

from evals.evaluate_qdrant_retrieval import (
    GoldenQuery,
    HitAtKMetric,
    MRRAtKMetric,
    Top1AccuracyMetric,
    expected_rank,
    load_retrieval_goldens,
    parse_qdrant_find_result,
    retrieve_queries,
)


def test_load_retrieval_goldens_deduplicates_articles(tmp_path):
    path = tmp_path / "goldens.jsonl"
    metadata = {"article_id": "article-1", "title": "Article One"}
    rows = [
        {
            "input": "First question",
            "expected_output": "Ignored answer",
            "context": ["Title: Article One\n\nKnowledge"],
            "additional_metadata": metadata,
        },
        {
            "input": "Second question",
            "expected_output": "Another ignored answer",
            "context": ["Title: Article One\n\nKnowledge"],
            "additional_metadata": metadata,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    articles, queries = load_retrieval_goldens(path)

    assert [article.article_id for article in articles] == ["article-1"]
    assert [query.input for query in queries] == [
        "First question",
        "Second question",
    ]
    assert all(query.expected_article_id == "article-1" for query in queries)


def test_load_retrieval_goldens_rejects_conflicting_contexts(tmp_path):
    path = tmp_path / "goldens.jsonl"
    rows = [
        {
            "input": "First question",
            "context": ["First content"],
            "additional_metadata": {"article_id": "article-1"},
        },
        {
            "input": "Second question",
            "context": ["Changed content"],
            "additional_metadata": {"article_id": "article-1"},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inconsistent"):
        load_retrieval_goldens(path)


def test_parse_qdrant_find_result_keeps_rank_and_top_three():
    blocks = [{"type": "text", "text": "Results for the query 'refund'"}]
    for index in range(1, 5):
        metadata = json.dumps({"article_id": f"article-{index}"})
        blocks.append(
            {
                "type": "text",
                "text": (
                    f"<entry><content>Content {index}</content>"
                    f"<metadata>{metadata}</metadata></entry>"
                ),
            }
        )

    retrieved = parse_qdrant_find_result(blocks, top_k=3)

    assert [article.article_id for article in retrieved] == [
        "article-1",
        "article-2",
        "article-3",
    ]


def test_parse_qdrant_find_result_accepts_fastmcp_json_list_content():
    entries = ["Results for the query 'refund'"]
    for index in range(1, 4):
        metadata = json.dumps({"article_id": f"article-{index}"})
        entries.append(
            f"<entry><content>Content {index}</content>"
            f"<metadata>{metadata}</metadata></entry>"
        )
    content = [SimpleNamespace(text=json.dumps(entries))]

    retrieved = parse_qdrant_find_result(content, top_k=3)

    assert [article.article_id for article in retrieved] == [
        "article-1",
        "article-2",
        "article-3",
    ]


@pytest.mark.asyncio
async def test_retrieve_queries_preserves_relevance_scores():
    entries = []
    for index, score in enumerate((0.91, 0.72, 0.35), start=1):
        metadata = json.dumps(
            {
                "article_id": f"article-{index}",
                "relevance_score": score,
            }
        )
        entries.append(
            f"<entry><content>Content {index}</content>"
            f"<metadata>{metadata}</metadata></entry>"
        )

    class FakeClient:
        async def call_tool(self, name, arguments):
            assert name == "qdrant-find"
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(entries))])

    rows = await retrieve_queries(
        FakeClient(),
        "evaluation-collection",
        [GoldenQuery(input="Question", expected_article_id="article-2")],
        top_k=3,
        max_concurrent=1,
    )

    assert rows[0]["retrieved_articles"] == [
        {"article_id": "article-1", "relevance_score": 0.91},
        {"article_id": "article-2", "relevance_score": 0.72},
        {"article_id": "article-3", "relevance_score": 0.35},
    ]


@pytest.mark.parametrize(
    ("retrieved", "rank", "hit_at_3", "top_1", "mrr_at_3"),
    [
        (["ideal", "other-1", "other-2"], 1, 1.0, 1.0, 1.0),
        (["other-1", "ideal", "other-2"], 2, 1.0, 0.0, 0.5),
        (["other-1", "other-2", "ideal"], 3, 1.0, 0.0, 1 / 3),
        (["other-1", "other-2", "other-3"], None, 0.0, 0.0, 0.0),
    ],
)
def test_article_id_metrics(retrieved, rank, hit_at_3, top_1, mrr_at_3):
    test_case = LLMTestCase(
        input="question",
        actual_output=json.dumps(retrieved),
        metadata={
            "expected_article_id": "ideal",
            "retrieved_article_ids": retrieved,
        },
    )

    assert expected_rank("ideal", retrieved) == rank
    assert HitAtKMetric(3).measure(test_case) == hit_at_3
    assert Top1AccuracyMetric().measure(test_case) == top_1
    assert MRRAtKMetric(3).measure(test_case) == pytest.approx(mrr_at_3)
