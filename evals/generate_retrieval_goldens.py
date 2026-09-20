from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from deepeval.models import DeepSeekModel
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.models.udahub import Knowledge


DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "core" / "udahub.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "datasets"
# Before changing StylingConfig, archive it in datasets/StylingConfig历史.md.
STYLING_CONFIG_VERSION = "v3_semantic_legacy_keyword_split"


def build_synthesizer_configs(
    model: DeepSeekModel,
    question_type: Literal["semantic", "keyword"] = "semantic",
) -> dict[str, Any]:
    """Build DeepEval settings for natural customer-support retrieval data."""
    from deepeval.synthesizer.config import (
        EvolutionConfig,
        FiltrationConfig,
        StylingConfig,
    )

    scenario = (
        "真实 CultPass 用户向客服咨询自己遇到的具体问题。"
        "用户使用自然、口语化表达，不使用客服内部术语。"
    )
    expected_output_format = (
        "一段客服直接回复用户的英文回答。"
        "只回答当前场景；优先采用上下文中的建议话术；"
        "明确用户下一步动作；不暴露内部授权或升级流程。"
    )

    if question_type == "semantic":
        task = (
            "根据知识文章生成客服检索测试数据。"
            "每条问题只描述一个具体用户场景，不包含解决措施。"
            "默认生成普通口语问题；仅当文章确实适合关键词检索时，"
            "才补充自然的关键词型问题。"
        )
        input_format = (
            "一条简短的英文用户消息。"
            "使用第一人称或自然疑问句，不包含答案提示、排查步骤、"
            "授权条件或升级处理方案。"
        )
    else:
        task = (
            "根据知识文章生成关键词型客服检索测试数据。每条问题必须自然保留"
            "至少一个来自标题或正文的精确检索锚点，例如功能名、状态、错误提示、"
            "标准术语、错误码或业务标识。同一篇文章生成的两条问题应尽量使用"
            "不同的检索锚点或不同场景。允许自然使用 reservation_id、"
            "transaction ID 等字段，也可以使用 RES-12345 等合理虚构值。不得"
            "输出 [date]、[type of event]、[platform/system] 等未填充模板，"
            "不得机械堆叠关键词，也不得包含答案或解决措施。问题描述的现象必须"
            "由原文直接支持，不得通过声称某个按钮、功能、状态或选项不存在来"
            "编造新的故障场景。每条问题只描述一个现象和一个诉求，不得同时"
            "询问政策资格、所需材料、排查步骤或完整处理流程。"
        )
        input_format = (
            "一条不超过 35 个英文单词的自然用户问题，不是关键词列表。问题中必须包含至少"
            "一个与原文一致或高度一致的关键状态、错误提示、功能名或业务标识，"
            "且不得包含未填充的方括号模板或原文没有描述的新故障。只描述问题"
            "本身，不询问应提供哪些资料、应执行哪些步骤或政策是否适用。"
        )

    return {
        "filtration_config": FiltrationConfig(
            synthetic_input_quality_threshold=0.3,
            max_quality_retries=3,
            critic_model=model,
        ),
        "evolution_config": EvolutionConfig(num_evolutions=0),
        "styling_config": StylingConfig(
            scenario=scenario,
            task=task,
            input_format=input_format,
            expected_output_format=expected_output_format,
        ),
    }


class CompatibleDeepSeekModel(DeepSeekModel):
    """Use zero when DeepEval has no pricing metadata for a DeepSeek model."""

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return super().calculate_cost(input_tokens, output_tokens) or 0.0


class GoldenLike(Protocol):
    input: str
    expected_output: str | None
    context: list[str] | None
    source_file: str | None


class SynthesizerLike(Protocol):
    def generate_goldens_from_contexts(
        self,
        *,
        contexts: list[list[str]],
        include_expected_output: bool,
        max_goldens_per_context: int,
        source_files: list[str],
    ) -> list[GoldenLike]: ...


def load_knowledge_records(db_path: Path) -> list[dict[str, Any]]:
    """Load article-level knowledge records without changing their content."""
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    session_local = sessionmaker(bind=engine)

    try:
        with session_local() as session:
            articles = (
                session.query(Knowledge)
                .order_by(Knowledge.article_id)
                .all()
            )

            return [
                {
                    "content": article.content,
                    "metadata": {
                        "article_id": article.article_id,
                        "account_id": article.account_id,
                        "title": article.title,
                        "tags": article.tags,
                        "created_at": (
                            article.created_at.isoformat()
                            if article.created_at is not None
                            else None
                        ),
                    },
                }
                for article in articles
            ]
    finally:
        engine.dispose()


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_versioned_goldens(
    output_dir: Path,
    rows: Iterable[dict[str, Any]],
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Save an immutable result copy tied to the active StylingConfig version."""
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must include timezone information")
    timestamp_text = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = (
        output_dir
        / "history"
        / f"retrieval_goldens_{STYLING_CONFIG_VERSION}_{timestamp_text}.jsonl"
    )
    write_jsonl(path, rows, overwrite=False)
    return path


def generate_retrieval_goldens(
    records: list[dict[str, Any]],
    synthesizer: SynthesizerLike,
    max_goldens_per_context: int,
) -> list[dict[str, Any]]:
    """Generate questions and expected answers grounded in original content."""
    contexts = [
        [f"Title: {record['metadata']['title']}\n\n{record['content']}"]
        for record in records
    ]
    article_ids = [record["metadata"]["article_id"] for record in records]
    metadata_by_article_id = {
        record["metadata"]["article_id"]: record["metadata"]
        for record in records
    }

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        source_files=article_ids,
        include_expected_output=True,
        max_goldens_per_context=max_goldens_per_context,
    )

    rows: list[dict[str, Any]] = []
    for golden in goldens:
        article_id = golden.source_file
        if article_id not in metadata_by_article_id:
            raise ValueError(
                f"Generated golden has an unknown source article: {article_id!r}"
            )

        rows.append(
            {
                "input": golden.input,
                "expected_output": golden.expected_output,
                "context": golden.context,
                "additional_metadata": metadata_by_article_id[article_id],
            }
        )

    return rows


def merge_generation_passes(
    records: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    keyword_rows: list[dict[str, Any]],
    expected_per_type: int = 2,
) -> list[dict[str, Any]]:
    """Validate and merge semantic and keyword rows without changing schema."""
    article_ids = [record["metadata"]["article_id"] for record in records]
    expected_ids = set(article_ids)

    def group_rows(
        rows: list[dict[str, Any]],
        question_type: str,
    ) -> dict[str, list[dict[str, Any]]]:
        grouped = {article_id: [] for article_id in article_ids}
        for row in rows:
            article_id = row["additional_metadata"]["article_id"]
            if article_id not in expected_ids:
                raise ValueError(
                    f"Unknown article in {question_type} rows: {article_id!r}"
                )
            question = row["input"].strip()
            if not question:
                raise ValueError(f"Empty {question_type} question for {article_id}")
            if question_type == "keyword" and re.search(r"\[[^\]\n]+\]", question):
                raise ValueError(
                    f"Unfilled template in keyword question for {article_id}: "
                    f"{question!r}"
                )
            if question_type == "keyword" and len(question.split()) > 35:
                raise ValueError(
                    f"Keyword question is longer than 35 words for {article_id}: "
                    f"{question!r}"
                )
            grouped[article_id].append(row)

        wrong_counts = {
            article_id: len(grouped[article_id])
            for article_id in article_ids
            if len(grouped[article_id]) != expected_per_type
        }
        if wrong_counts:
            raise ValueError(
                f"Expected {expected_per_type} {question_type} rows per article, "
                f"got {wrong_counts}"
            )
        return grouped

    semantic_by_id = group_rows(semantic_rows, "semantic")
    keyword_by_id = group_rows(keyword_rows, "keyword")

    merged: list[dict[str, Any]] = []
    normalized_inputs: set[str] = set()
    for article_id in article_ids:
        for row in semantic_by_id[article_id] + keyword_by_id[article_id]:
            normalized_input = " ".join(row["input"].casefold().split())
            if normalized_input in normalized_inputs:
                raise ValueError(f"Duplicate generated question: {row['input']!r}")
            normalized_inputs.add(normalized_input)
            merged.append(row)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DeepEval retrieval goldens from the knowledge table."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is missing.")

    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0"))
    max_concurrent = int(os.getenv("DEEPSEEK_MAX_CONCURRENT", "5"))

    if max_concurrent < 1:
        raise ValueError("DEEPSEEK_MAX_CONCURRENT must be at least 1")
    if not args.db_path.is_file():
        raise FileNotFoundError(f"Knowledge database not found: {args.db_path}")

    records = load_knowledge_records(args.db_path)
    if not records:
        raise ValueError("The knowledge table is empty")

    snapshot_path = args.output_dir / "knowledge_snapshot.jsonl"
    write_jsonl(snapshot_path, records)

    from deepeval.synthesizer import Synthesizer

    model = CompatibleDeepSeekModel(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )

    semantic_rows = generate_retrieval_goldens(
        records=records,
        synthesizer=Synthesizer(
            model=model,
            max_concurrent=max_concurrent,
            **build_synthesizer_configs(model, "semantic"),
        ),
        max_goldens_per_context=2,
    )
    keyword_rows = generate_retrieval_goldens(
        records=records,
        synthesizer=Synthesizer(
            model=model,
            max_concurrent=max_concurrent,
            **build_synthesizer_configs(model, "keyword"),
        ),
        max_goldens_per_context=2,
    )
    golden_rows = merge_generation_passes(
        records=records,
        semantic_rows=semantic_rows,
        keyword_rows=keyword_rows,
    )
    goldens_path = args.output_dir / "retrieval_goldens.jsonl"
    write_jsonl(goldens_path, golden_rows)
    versioned_goldens_path = save_versioned_goldens(args.output_dir, golden_rows)

    print(f"Knowledge records: {len(records)} -> {snapshot_path}")
    print(f"Retrieval goldens: {len(golden_rows)} -> {goldens_path}")
    print(f"Versioned copy: {versioned_goldens_path}")


if __name__ == "__main__":
    main()
