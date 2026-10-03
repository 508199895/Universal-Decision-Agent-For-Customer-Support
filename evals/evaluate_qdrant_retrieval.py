from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from fastmcp import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDENS_PATH = PROJECT_ROOT / "evals" / "datasets" / "retrieval_goldens.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
DEFAULT_COLLECTION_NAME = "uda-hub-retrieval-eval"
DEFAULT_TOP_K = 3
_ENTRY_PATTERN = re.compile(
    r"<entry><content>(.*?)</content><metadata>(.*?)</metadata></entry>",
    re.DOTALL,
)


@dataclass(frozen=True)
class KnowledgeArticle:
    article_id: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GoldenQuery:
    input: str
    expected_article_id: str


@dataclass(frozen=True)
class RetrievedArticle:
    article_id: str
    content: str
    metadata: dict[str, Any]


def load_retrieval_goldens(
    path: Path,
) -> tuple[list[KnowledgeArticle], list[GoldenQuery]]:
    """Load questions and deduplicate their ideal contexts into articles."""
    articles_by_id: dict[str, KnowledgeArticle] = {}
    queries: list[GoldenQuery] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("input")
            metadata = row.get("additional_metadata")
            contexts = row.get("context")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_number} has no valid input")
            if not isinstance(metadata, dict):
                raise ValueError(f"Line {line_number} has no additional_metadata")
            article_id = metadata.get("article_id")
            if not isinstance(article_id, str) or not article_id:
                raise ValueError(f"Line {line_number} has no article_id")
            if not isinstance(contexts, list) or len(contexts) != 1:
                raise ValueError(
                    f"Line {line_number} must contain exactly one ideal context"
                )
            content = contexts[0]
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Line {line_number} has an empty ideal context")

            article = KnowledgeArticle(
                article_id=article_id,
                content=content,
                metadata=dict(metadata),
            )
            previous = articles_by_id.get(article_id)
            if previous is not None and previous != article:
                raise ValueError(
                    f"Article {article_id!r} has inconsistent context or metadata"
                )
            articles_by_id[article_id] = article
            queries.append(
                GoldenQuery(
                    input=question.strip(),
                    expected_article_id=article_id,
                )
            )

    if not queries:
        raise ValueError(f"No retrieval goldens found in {path}")
    return list(articles_by_id.values()), queries


def _extract_text_blocks(result: Any) -> list[str]:
    if isinstance(result, str):
        return [result]
    if isinstance(result, list):
        texts: list[str] = []
        for item in result:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                texts.append(item.text)
        return texts
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return [result["text"]]
    if isinstance(getattr(result, "text", None), str):
        return [result.text]
    raise ValueError(f"Unsupported Qdrant MCP result type: {type(result).__name__}")


def parse_qdrant_find_result(result: Any, top_k: int = DEFAULT_TOP_K) -> list[RetrievedArticle]:
    """Parse the official mcp-server-qdrant entry format in ranked order."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    text_blocks: list[str] = []
    for text_block in _extract_text_blocks(result):
        try:
            decoded = json.loads(text_block)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list) and all(
            isinstance(item, str) for item in decoded
        ):
            text_blocks.extend(decoded)
        else:
            text_blocks.append(text_block)

    joined = "\n".join(text_blocks)
    retrieved: list[RetrievedArticle] = []
    for content, metadata_text in _ENTRY_PATTERN.findall(joined):
        try:
            metadata = json.loads(metadata_text) if metadata_text else {}
        except json.JSONDecodeError as error:
            raise ValueError("Qdrant returned invalid entry metadata JSON") from error
        article_id = metadata.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError("Qdrant result is missing metadata.article_id")
        retrieved.append(
            RetrievedArticle(
                article_id=article_id,
                content=content,
                metadata=metadata,
            )
        )
    return retrieved[:top_k]


def expected_rank(
    expected_article_id: str,
    retrieved_article_ids: Sequence[str],
) -> int | None:
    try:
        return retrieved_article_ids.index(expected_article_id) + 1
    except ValueError:
        return None


class _ArticleIdMetric(BaseMetric):
    threshold = 1.0
    async_mode = False
    verbose_mode = False
    include_reason = False

    def _score(self, test_case: LLMTestCase) -> float:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        self.error = None
        self.score = self._score(test_case)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            self.success = self.score is not None and self.score >= self.threshold
        return self.success

    @staticmethod
    def _metadata(test_case: LLMTestCase) -> tuple[str, list[str]]:
        metadata = test_case.metadata or {}
        expected = metadata.get("expected_article_id")
        retrieved = metadata.get("retrieved_article_ids")
        if not isinstance(expected, str) or not isinstance(retrieved, list):
            raise ValueError("Retrieval metric metadata is incomplete")
        if not all(isinstance(article_id, str) for article_id in retrieved):
            raise ValueError("retrieved_article_ids must contain only strings")
        return expected, retrieved


class HitAtKMetric(_ArticleIdMetric):
    def __init__(self, k: int = DEFAULT_TOP_K) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def _score(self, test_case: LLMTestCase) -> float:
        expected, retrieved = self._metadata(test_case)
        return float(expected in retrieved[: self.k])

    @property
    def __name__(self) -> str:
        return f"Hit@{self.k}"


class Top1AccuracyMetric(_ArticleIdMetric):
    def _score(self, test_case: LLMTestCase) -> float:
        expected, retrieved = self._metadata(test_case)
        return float(bool(retrieved) and retrieved[0] == expected)

    @property
    def __name__(self) -> str:
        return "Top-1 Accuracy"


class MRRAtKMetric(_ArticleIdMetric):
    def __init__(self, k: int = DEFAULT_TOP_K) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.threshold = 1 / k

    def _score(self, test_case: LLMTestCase) -> float:
        expected, retrieved = self._metadata(test_case)
        rank = expected_rank(expected, retrieved[: self.k])
        return 0.0 if rank is None else 1 / rank

    @property
    def __name__(self) -> str:
        return f"MRR@{self.k}"


async def ingest_articles(
    client: Client,
    collection_name: str,
    articles: Iterable[KnowledgeArticle],
) -> None:
    for article in articles:
        await client.call_tool(
            "qdrant-store",
            {
                "information": article.content,
                "collection_name": collection_name,
                "metadata": article.metadata,
            }
        )


async def retrieve_queries(
    client: Client,
    collection_name: str,
    queries: Sequence[GoldenQuery],
    *,
    top_k: int,
    max_concurrent: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def retrieve(index: int, golden: GoldenQuery) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            result = await client.call_tool(
                "qdrant-find",
                {"query": golden.input, "collection_name": collection_name}
            )
        retrieved = parse_qdrant_find_result(result.content, top_k=top_k)
        retrieved_ids = [article.article_id for article in retrieved]
        retrieved_articles = [
            {
                "article_id": article.article_id,
                "relevance_score": article.metadata.get("relevance_score"),
            }
            for article in retrieved
        ]
        rank = expected_rank(golden.expected_article_id, retrieved_ids)
        return index, {
            "input": golden.input,
            "expected_article_id": golden.expected_article_id,
            "retrieved_article_ids": retrieved_ids,
            "retrieved_articles": retrieved_articles,
            "expected_rank": rank,
            "hit_at_3": float(rank is not None),
            "top_1_accuracy": float(rank == 1),
            "reciprocal_rank_at_3": 0.0 if rank is None else 1 / rank,
        }

    indexed_rows = await asyncio.gather(
        *(retrieve(index, golden) for index, golden in enumerate(queries))
    )
    return [row for _, row in sorted(indexed_rows)]


def evaluate_with_deepeval(
    retrieval_rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> None:
    test_cases = [
        LLMTestCase(
            input=row["input"],
            actual_output=json.dumps(row["retrieved_article_ids"]),
            metadata={
                "expected_article_id": row["expected_article_id"],
                "retrieved_article_ids": row["retrieved_article_ids"],
            },
        )
        for row in retrieval_rows
    ]
    evaluate(
        test_cases=test_cases,
        metrics=[
            HitAtKMetric(top_k),
            Top1AccuracyMetric(),
            MRRAtKMetric(top_k),
        ],
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
        ),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )


def build_report(
    collection_name: str,
    article_count: int,
    retrieval_rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    count = len(retrieval_rows)
    if count == 0:
        raise ValueError("Cannot build a report without retrieval rows")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "article_count": article_count,
        "query_count": count,
        "top_k": top_k,
        "summary": {
            f"hit_at_{top_k}": sum(row["hit_at_3"] for row in retrieval_rows)
            / count,
            "top_1_accuracy": sum(
                row["top_1_accuracy"] for row in retrieval_rows
            )
            / count,
            f"mrr_at_{top_k}": sum(
                row["reciprocal_rank_at_3"] for row in retrieval_rows
            )
            / count,
        },
        "results": list(retrieval_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Qdrant knowledge collection and evaluate article-ID retrieval."
    )
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> Path:
    if args.top_k != DEFAULT_TOP_K:
        raise ValueError("This evaluation is fixed to top_k=3")
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")
    if not args.goldens.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {args.goldens}")

    articles, queries = load_retrieval_goldens(args.goldens)
    collection_name = args.collection_name
    mcp_url = os.getenv("QDRANT_MCP_URL", "http://127.0.0.1:8000/mcp")
    async with Client(mcp_url) as client:
        tools = {tool.name for tool in await client.list_tools()}
        required_tools = {"qdrant-find"}
        if not args.skip_ingest:
            required_tools.add("qdrant-store")
        missing_tools = required_tools - tools
        if missing_tools:
            raise RuntimeError(
                f"Qdrant MCP is missing tools: {sorted(missing_tools)}"
            )
        if not args.skip_ingest:
            await ingest_articles(client, collection_name, articles)

        rows = await retrieve_queries(
            client,
            collection_name,
            queries,
            top_k=args.top_k,
            max_concurrent=args.max_concurrent,
        )
    evaluate_with_deepeval(rows, top_k=args.top_k)
    report = build_report(
        collection_name,
        len(articles),
        rows,
        top_k=args.top_k,
    )

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_RESULTS_DIR / f"qdrant_retrieval_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")
    output_path = asyncio.run(async_main(parse_args()))
    report = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(report["summary"], indent=2))
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
