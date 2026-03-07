from agent_trace.config import JudgeConfig, _merge_judge_configs
from agent_trace.judge.prompt_builder import build_judge_prompt
from agent_trace.judge.scorer import _parse_judge_response


def test_build_judge_prompt_contains_fields():
    prompt = build_judge_prompt(
        name="researcher",
        description="Searches for information",
        goal="Must cite sources",
        input_data="What is Python?",
        output_data="Python is a programming language.",
    )
    assert "researcher" in prompt
    assert "Searches for information" in prompt
    assert "Must cite sources" in prompt
    assert "What is Python?" in prompt
    assert "Python is a programming language." in prompt


def test_build_judge_prompt_default_criteria():
    prompt = build_judge_prompt(
        name="x", description="", goal="", input_data="in", output_data="out",
    )
    assert "quality" in prompt.lower()


def test_parse_valid_json():
    score, explanation = _parse_judge_response(
        '{"score": 4, "explanation": "Good but missing sources"}'
    )
    assert score == 4.0
    assert "missing sources" in explanation


def test_parse_json_in_code_block():
    score, explanation = _parse_judge_response(
        '```json\n{"score": 3, "explanation": "Acceptable"}\n```'
    )
    assert score == 3.0
    assert "Acceptable" in explanation


def test_parse_fallback_score_slash():
    score, _ = _parse_judge_response(
        "I'd rate this 2/5 because it lacks detail."
    )
    assert score == 2.0


def test_parse_failure():
    score, explanation = _parse_judge_response(
        "This is gibberish with no score."
    )
    assert score == 0.0
    assert "Could not parse" in explanation


# ── JudgeConfig merge tests ──────────────────────────────────────────


def test_merge_configs_first_non_none_wins():
    step_cfg = JudgeConfig(model="step-model", temperature=0.9)
    call_cfg = JudgeConfig(model="call-model", api_key="call-key")
    global_cfg = JudgeConfig(model="global-model", api_key="global-key", provider="openai")

    merged = _merge_judge_configs(step_cfg, call_cfg, global_cfg)
    assert merged.model == "step-model"  # step wins
    assert merged.temperature == 0.9  # step wins
    assert merged.api_key == "call-key"  # call wins (step has None)
    assert merged.provider == "openai"  # global wins (step+call have None)


def test_merge_configs_skips_none_entries():
    merged = _merge_judge_configs(None, None, JudgeConfig(api_key="fallback"))
    assert merged.api_key == "fallback"


def test_merge_configs_all_none():
    merged = _merge_judge_configs(None, None, None)
    assert merged.model is None
    assert merged.api_key is None
