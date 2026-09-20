from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models.udahub import Account, Base, Knowledge
from evals.generate_retrieval_goldens import (
    CompatibleDeepSeekModel,
    STYLING_CONFIG_VERSION,
    build_synthesizer_configs,
    generate_retrieval_goldens,
    load_knowledge_records,
    merge_generation_passes,
    save_versioned_goldens,
)


def test_compatible_deepseek_model_normalizes_unknown_cost():
    model = CompatibleDeepSeekModel(
        model="deepseek-flash",
        api_key="test-key",
    )

    assert model.calculate_cost(100, 50) == 0.0


def test_save_versioned_goldens_uses_config_version_and_never_overwrites(
    tmp_path: Path,
):
    rows = [{"input": "Question", "expected_output": "Answer"}]
    generated_at = datetime(2026, 9, 20, 5, 30, 45, tzinfo=timezone.utc)

    path = save_versioned_goldens(tmp_path, rows, generated_at=generated_at)

    assert path == (
        tmp_path
        / "history"
        / f"retrieval_goldens_{STYLING_CONFIG_VERSION}_20260920T053045Z.jsonl"
    )
    assert path.read_text(encoding="utf-8") == (
        '{"input": "Question", "expected_output": "Answer"}\n'
    )
    with pytest.raises(FileExistsError):
        save_versioned_goldens(tmp_path, rows, generated_at=generated_at)


def test_build_synthesizer_configs_uses_customer_support_settings():
    model = CompatibleDeepSeekModel(model="deepseek-flash", api_key="test-key")
    configs = build_synthesizer_configs(model)

    assert configs["filtration_config"].synthetic_input_quality_threshold == 0.3
    assert configs["filtration_config"].max_quality_retries == 3
    assert configs["filtration_config"].critic_model is model
    assert configs["evolution_config"].num_evolutions == 0

    styling = configs["styling_config"]
    assert styling.scenario == (
        "真实 CultPass 用户向客服咨询自己遇到的具体问题。"
        "用户使用自然、口语化表达，不使用客服内部术语。"
    )
    assert styling.task == (
        "根据知识文章生成客服检索测试数据。"
        "每条问题只描述一个具体用户场景，不包含解决措施。"
        "默认生成普通口语问题；仅当文章确实适合关键词检索时，"
        "才补充自然的关键词型问题。"
    )
    assert styling.input_format == (
        "一条简短的英文用户消息。"
        "使用第一人称或自然疑问句，不包含答案提示、排查步骤、"
        "授权条件或升级处理方案。"
    )
    assert styling.expected_output_format == (
        "一段客服直接回复用户的英文回答。"
        "只回答当前场景；优先采用上下文中的建议话术；"
        "明确用户下一步动作；不暴露内部授权或升级流程。"
    )


def test_build_keyword_synthesizer_configs_requires_natural_anchors():
    model = CompatibleDeepSeekModel(model="deepseek-flash", api_key="test-key")
    semantic_configs = build_synthesizer_configs(model, "semantic")
    configs = build_synthesizer_configs(model, "keyword")

    styling = configs["styling_config"]
    semantic_styling = semantic_configs["styling_config"]
    assert styling.scenario == semantic_styling.scenario
    assert styling.expected_output_format == semantic_styling.expected_output_format
    assert "关键词型" in styling.task
    assert "reservation_id" in styling.task
    assert "未填充模板" in styling.task
    assert "原文直接支持" in styling.task
    assert "编造新的故障场景" in styling.task
    assert "所需材料" in styling.task
    assert "不是关键词列表" in styling.input_format
    assert "不超过 35 个英文单词" in styling.input_format


def test_load_knowledge_records_preserves_content_and_selected_metadata(
    tmp_path: Path,
):
    db_path = tmp_path / "knowledge.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    with session_local() as session:
        session.add(Account(account_id="account-1", account_name="Test"))
        session.add(
            Knowledge(
                article_id="article-1",
                account_id="account-1",
                title="Original title",
                content="Original content only.",
                tags="tag-a,tag-b",
                created_at=datetime(2026, 9, 19, 8, 30, 0),
                updated_at=datetime(2026, 9, 20, 9, 0, 0),
            )
        )
        session.commit()

    records = load_knowledge_records(db_path)

    assert records == [
        {
            "content": "Original content only.",
            "metadata": {
                "article_id": "article-1",
                "account_id": "account-1",
                "title": "Original title",
                "tags": "tag-a,tag-b",
                "created_at": "2026-09-19T08:30:00",
            },
        }
    ]


def test_generate_retrieval_goldens_outputs_only_agreed_fields():
    records = [
        {
            "content": "Original content only.",
            "metadata": {
                "article_id": "article-1",
                "account_id": "account-1",
                "title": "Original title",
                "tags": "tag-a,tag-b",
                "created_at": "2026-09-19T08:30:00",
            },
        }
    ]

    class FakeSynthesizer:
        def generate_goldens_from_contexts(self, **kwargs):
            assert kwargs == {
                "contexts": [["Title: Original title\n\nOriginal content only."]],
                "source_files": ["article-1"],
                "include_expected_output": True,
                "max_goldens_per_context": 2,
            }
            return [
                SimpleNamespace(
                    input="Generated question?",
                    expected_output="Generated answer.",
                    context=["Title: Original title\n\nOriginal content only."],
                    source_file="article-1",
                )
            ]

    rows = generate_retrieval_goldens(records, FakeSynthesizer(), 2)

    assert rows == [
        {
            "input": "Generated question?",
            "expected_output": "Generated answer.",
            "context": ["Title: Original title\n\nOriginal content only."],
            "additional_metadata": records[0]["metadata"],
        }
    ]


def test_merge_generation_passes_keeps_schema_and_orders_types():
    metadata = {
        "article_id": "article-1",
        "account_id": "account-1",
        "title": "Original title",
        "tags": "tag-a,tag-b",
        "created_at": "2026-09-19T08:30:00",
    }
    records = [{"content": "Original content only.", "metadata": metadata}]
    semantic_row = {
        "input": "Why is this happening?",
        "expected_output": "Semantic answer.",
        "context": ["Title: Original title\n\nOriginal content only."],
        "additional_metadata": metadata,
    }
    keyword_row = {
        "input": "Why does Original title show this status?",
        "expected_output": "Keyword answer.",
        "context": ["Title: Original title\n\nOriginal content only."],
        "additional_metadata": metadata,
    }

    rows = merge_generation_passes(
        records,
        [semantic_row],
        [keyword_row],
        expected_per_type=1,
    )

    assert rows == [semantic_row, keyword_row]
    assert all("input_type" not in row for row in rows)


def test_merge_generation_passes_rejects_unfilled_keyword_template():
    metadata = {"article_id": "article-1", "title": "Original title"}
    records = [{"content": "Original content only.", "metadata": metadata}]
    semantic_row = {
        "input": "Why is this happening?",
        "expected_output": "Semantic answer.",
        "context": [],
        "additional_metadata": metadata,
    }
    keyword_row = {
        "input": "My [reservation_id] is not working.",
        "expected_output": "Keyword answer.",
        "context": [],
        "additional_metadata": metadata,
    }

    with pytest.raises(ValueError, match="Unfilled template"):
        merge_generation_passes(
            records,
            [semantic_row],
            [keyword_row],
            expected_per_type=1,
        )


def test_merge_generation_passes_rejects_long_keyword_question():
    metadata = {"article_id": "article-1", "title": "Original title"}
    records = [{"content": "Original content only.", "metadata": metadata}]
    semantic_row = {
        "input": "Why is this happening?",
        "expected_output": "Semantic answer.",
        "context": [],
        "additional_metadata": metadata,
    }
    keyword_row = {
        "input": " ".join(["keyword"] * 36),
        "expected_output": "Keyword answer.",
        "context": [],
        "additional_metadata": metadata,
    }

    with pytest.raises(ValueError, match="longer than 35 words"):
        merge_generation_passes(
            records,
            [semantic_row],
            [keyword_row],
            expected_per_type=1,
        )
