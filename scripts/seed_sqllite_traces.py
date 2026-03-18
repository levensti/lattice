"""Seed the local Lattice DB with realistic multi-agent demo traces."""

import sys
sys.path.insert(0, ".")

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from lattice.context import (
    ActionRecord,
    GroupRecord,
    JudgeResult,
    TraceSession,
    TransitionRecord,
)
from lattice.storage.store import save_session


def _assign_timestamps(session: TraceSession) -> None:
    """Assign synthetic started_at/ended_at from created_at + latency_ms.

    Parallel group members get the same start time; sequential actions
    flow one after another.
    """
    base = datetime.fromisoformat(session.created_at)
    parallel_groups = {
        g.group_id for g in session.groups if g.group_type == "parallel"
    }
    actions = sorted(session.actions, key=lambda a: a.action_index)

    cursor = base
    for action in actions:
        action.started_at = cursor.isoformat()
        action.ended_at = (cursor + timedelta(milliseconds=action.latency_ms)).isoformat()
        cursor += timedelta(milliseconds=action.latency_ms)

    by_group: dict[str, list[ActionRecord]] = defaultdict(list)
    for action in actions:
        if action.group_id in parallel_groups:
            by_group[action.group_id].append(action)

    for members in by_group.values():
        earliest = min(datetime.fromisoformat(a.started_at) for a in members)
        for a in members:
            a.started_at = earliest.isoformat()
            a.ended_at = (earliest + timedelta(milliseconds=a.latency_ms)).isoformat()


def _trace_deep_research():
    """Deep research agent: orchestrator -> parallel search -> extract -> synthesize -> critique loop -> fact-check -> format."""
    s = TraceSession(
        trace_id="deep-research-001",
        workflow_name="Deep Research Agent",
        goal="Produce a comprehensive, well-sourced report on the impact of large language models on software engineering productivity",
        session_score=4.2,
        session_score_explanation="Strong report with good source coverage. Minor gap in quantitative data for open-source LLM tooling adoption rates.",
        created_at="2026-03-16T09:15:00+00:00",
    )

    s.add_action(ActionRecord(
        span_id="orch-1",
        name="orchestrator",
        goal="Decompose the research question into sub-queries and assign to specialist agents",
        input_data='{"topic": "Impact of LLMs on software engineering productivity", "depth": "comprehensive", "min_sources": 15}',
        output_data='{"plan": ["search_academic", "search_industry", "search_oss_repos", "synthesize", "critique", "write_report"], "estimated_steps": 12}',
        action_index=0, latency_ms=1250.0,
        score=4.5,
        score_explanation="Well-structured decomposition covering academic, industry, and open-source dimensions.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate the orchestrator's planning quality. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Comprehensive decomposition with good coverage", name="planning_quality", model="claude-sonnet-4-5-20250514"),
    ))

    s.add_group(GroupRecord(group_id="search-fanout", group_type="parallel", name="Source Gathering", parent_span_id="orch-1"))

    s.add_action(ActionRecord(
        span_id="search-academic",
        name="search_academic_papers",
        goal="Find 5-8 high-quality academic papers on LLM-assisted coding productivity",
        input_data='{"queries": ["LLM software engineering productivity study", "copilot developer productivity RCT", "code generation evaluation benchmark"], "sources": ["arxiv", "semantic_scholar"]}',
        output_data='{"papers_found": 7, "top_papers": [{"title": "Productivity Assessment of Neural Code Completion", "year": 2024, "citations": 142}]}',
        action_index=1, latency_ms=3420.0,
        parent_span_id="orch-1",
        group_id="search-fanout",
        score=4.0,
        score_explanation="Good academic coverage, though missing some recent workshop papers.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate source quality and relevance. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Relevant papers with solid citation counts", name="source_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="search-industry",
        name="search_industry_reports",
        goal="Find 5-8 industry sources with real-world productivity metrics",
        input_data='{"queries": ["developer productivity AI tools survey 2025", "github copilot ROI enterprise"]}',
        output_data='{"sources_found": 6, "highlights": [{"source": "GitHub 2025 Octoverse", "finding": "55% of code in JS repos now AI-assisted"}]}',
        action_index=2, latency_ms=2870.0,
        parent_span_id="orch-1",
        group_id="search-fanout",
        score=4.5,
        score_explanation="Excellent industry coverage with concrete metrics.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate industry source quality and relevance of metrics cited. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Strong quantitative data from reputable sources", name="source_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="search-oss",
        name="search_oss_ecosystem",
        goal="Map the landscape of open-source LLM coding tools and their adoption",
        input_data='{"queries": ["open source code completion tools", "LLM IDE plugins github stars"]}',
        output_data='{"tools_found": 12, "adoption_trend": "3x YoY growth in combined downloads"}',
        action_index=3, latency_ms=4100.0,
        parent_span_id="orch-1",
        group_id="search-fanout",
        score=3.5,
        score_explanation="Good tool mapping but lacking quantitative adoption data for most tools.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate open-source ecosystem mapping: coverage breadth and adoption data depth. Score 1-5."},
        judge_result=JudgeResult(score=3.5, explanation="Comprehensive list but shallow on usage metrics", name="depth", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="reader-1",
        name="extract_key_findings",
        goal="Produce a unified set of key findings with citations",
        input_data='{"source_count": 25, "format": "structured_findings"}',
        output_data='{"total_findings": 11}',
        action_index=4, latency_ms=5600.0,
        parent_span_id="orch-1",
        score=4.0,
        score_explanation="Well-structured extraction with confidence levels.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate extraction quality: accuracy of key findings, proper citation, and confidence calibration. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Clean structured output with proper attribution", name="extraction_quality", model="claude-sonnet-4-5-20250514"),
    ))

    s.add_action(ActionRecord(
        span_id="synth-1",
        name="synthesize_narrative",
        goal="Create a structured draft report with introduction, 4-5 thematic sections, and conclusion",
        input_data='{"findings_count": 11, "target_sections": 5, "tone": "analytical"}',
        output_data='{"sections": 6, "word_count": 3200, "citations_used": 18}',
        action_index=5, latency_ms=8200.0,
        parent_span_id="orch-1",
        score=4.0,
        score_explanation="Strong narrative structure. Could use more quantitative anchoring in Section 5.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate narrative synthesis: logical flow, thematic coherence, and evidence integration. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Well-organized with clear thematic progression", name="structure", model="claude-sonnet-4-5-20250514"),
    ))

    s.add_group(GroupRecord(group_id="critique-loop", group_type="loop", name="Critique & Refine", parent_span_id="orch-1"))

    s.add_action(ActionRecord(
        span_id="critic-1",
        name="critique_draft",
        goal="Identify weaknesses and suggest specific improvements",
        input_data='{"draft_version": 1, "word_count": 3200}',
        output_data='{"issues": 3, "overall": "Strong draft with 3 issues to address before final"}',
        action_index=6, latency_ms=4500.0,
        parent_span_id="synth-1",
        group_id="critique-loop",
        iteration=0,
        score=4.5,
        score_explanation="Thorough critique with actionable suggestions.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate critique quality: specificity, actionability, and coverage of weaknesses. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Specific, actionable feedback with severity levels", name="critique_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="revise-1",
        name="revise_draft",
        goal="Resolve all major issues and most minor issues from the critique",
        input_data='{"issues_to_address": 3, "major": 1, "minor": 2}',
        output_data='{"changes_made": 3, "new_word_count": 3480}',
        action_index=7, latency_ms=6100.0,
        parent_span_id="synth-1",
        group_id="critique-loop",
        iteration=0,
        score=4.0,
        score_explanation="All issues addressed. Quantitative data added where suggested.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate revision quality: how well critique issues were addressed and overall improvement. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="critic-2",
        name="critique_draft",
        goal="Verify fixes and check for new issues",
        input_data='{"draft_version": 2, "word_count": 3480}',
        output_data='{"issues": 1, "overall": "Substantially improved. One minor suggestion remaining."}',
        action_index=8, latency_ms=3800.0,
        parent_span_id="synth-1",
        group_id="critique-loop",
        iteration=1,
        score=4.0,
        score_explanation="Good follow-up review.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate follow-up critique: verification of fixes and identification of remaining issues. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="revise-2",
        name="revise_draft",
        goal="Polish and finalize the report",
        input_data='{"issues_to_address": 1}',
        output_data='{"changes_made": 1, "new_word_count": 3630}',
        action_index=9, latency_ms=3200.0,
        parent_span_id="synth-1",
        group_id="critique-loop",
        iteration=1,
        score=4.5,
        score_explanation="Clean final revision.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate final revision polish: clarity, conciseness, and readiness for publication. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="factcheck-1",
        name="verify_claims",
        goal="Flag any unsupported or misrepresented statistics",
        input_data='{"claims_to_verify": 14, "sources_available": 25}',
        output_data='{"verified": 12, "flagged": 1, "unable_to_verify": 1}',
        action_index=10, latency_ms=7300.0,
        parent_span_id="orch-1",
        score=4.5,
        score_explanation="Caught an important overstatement. Thorough cross-referencing.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate fact-checking rigor: verification coverage, source cross-referencing, and flagging accuracy. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Rigorous verification with clear evidence trail", name="accuracy", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="format-1",
        name="format_final_report",
        goal="Produce publication-ready report in markdown",
        input_data='{"word_count": 3630, "sections": 6, "citations": 18}',
        output_data='{"format": "markdown", "word_count": 4100, "bibliography_entries": 18}',
        action_index=11, latency_ms=2900.0,
        parent_span_id="orch-1",
        score=4.5,
        score_explanation="Clean formatting with proper academic structure.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate report formatting: structure, readability, citation formatting, and presentation quality. Score 1-5."},
    ))

    _assign_timestamps(s)
    save_session(s)
    print(f"  Saved: {s.trace_id} ({len(s.actions)} actions)")


def _trace_multi_agent_coding_pipeline():
    """Complex multi-agent coding pipeline with deeply nested parallel/sequential operations.

    Architecture:
      planner -> parallel(architect, risk_assessor) -> parallel(
        team_backend(lead -> parallel(impl_api, impl_db, impl_auth) -> integrate),
        team_frontend(lead -> parallel(impl_ui, impl_state) -> integrate),
        team_infra(lead -> parallel(impl_ci, impl_deploy))
      ) -> parallel(test_unit, test_integration, test_security) -> reviewer -> deployer
    """
    s = TraceSession(
        trace_id="coding-pipeline-deep-001",
        workflow_name="Multi-Agent Coding Pipeline",
        goal="Implement a user authentication system with OAuth2, session management, and role-based access control across backend, frontend, and infrastructure",
        session_score=4.1,
        session_score_explanation="Feature implemented successfully across all layers. Minor gaps in RBAC edge cases for nested org hierarchies. Security tests caught a session fixation vulnerability that was fixed before deployment.",
        created_at="2026-03-16T10:00:00+00:00",
    )

    # ── planner (root) ────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="plan",
        name="project_planner",
        goal="Create implementation plan with task dependencies, team assignments, and acceptance criteria",
        input_data='{"feature": "OAuth2 authentication with RBAC", "teams": ["backend", "frontend", "infra"], "deadline": "sprint_14"}',
        output_data='{"tasks": 14, "teams_assigned": 3, "critical_path": ["architect", "backend_api", "frontend_auth", "integration_test", "deploy"], "estimated_hours": 32}',
        action_index=0, latency_ms=2100.0,
        score=4.5,
        score_explanation="Thorough task decomposition with correct dependency graph.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate planning quality: task decomposition, dependency accuracy, resource allocation. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Well-structured plan with appropriate parallelism", name="planning", model="claude-sonnet-4-5-20250514"),
    ))

    # ── parallel: architecture + risk assessment ──────────────
    s.add_group(GroupRecord(group_id="design-phase", group_type="parallel", name="Design Phase", parent_span_id="plan"))

    s.add_action(ActionRecord(
        span_id="architect",
        name="system_architect",
        goal="Produce a detailed architecture spec that all implementation teams can follow",
        input_data='{"scope": "OAuth2 + RBAC", "constraints": ["existing user table", "Redis for sessions", "PostgreSQL"]}',
        output_data='{"api_endpoints": 8, "data_models": 4, "sequence_diagrams": 3, "decisions": ["JWT for API auth", "Redis for session store", "RBAC via role+permission tables"]}',
        action_index=1, latency_ms=5400.0,
        parent_span_id="plan",
        group_id="design-phase",
        score=4.5,
        score_explanation="Clean architecture with good separation of concerns.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate architecture quality: completeness, scalability, security considerations. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Solid design with appropriate technology choices", name="design_quality", model="claude-sonnet-4-5-20250514"),
    ))

    s.add_action(ActionRecord(
        span_id="risk",
        name="risk_assessor",
        goal="Produce a risk matrix with mitigations for the auth implementation",
        input_data='{"feature": "OAuth2 + RBAC", "compliance": ["SOC2", "GDPR"]}',
        output_data='{"risks_identified": 7, "high_severity": 2, "mitigations": ["PKCE for OAuth flow", "session rotation on privilege escalation", "audit logging for all permission changes"]}',
        action_index=2, latency_ms=3200.0,
        parent_span_id="plan",
        group_id="design-phase",
        score=4.0,
        score_explanation="Good risk identification. Missed token replay attack vector.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate risk assessment: threat coverage, severity calibration, and mitigation quality. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Solid risk analysis with actionable mitigations", name="risk_coverage", model="gpt-4o"),
    ))

    # ── parallel: three implementation teams ──────────────────
    s.add_group(GroupRecord(group_id="impl-teams", group_type="parallel", name="Implementation Teams", parent_span_id="plan"))

    # ── BACKEND TEAM ──────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="be-lead",
        name="backend_lead",
        goal="Assign and coordinate backend sub-tasks",
        input_data='{"arch_spec": "8 endpoints, 4 models", "team_size": 3}',
        output_data='{"subtasks": ["api_endpoints", "db_migrations", "auth_service"], "parallelizable": true}',
        action_index=3, latency_ms=1100.0,
        parent_span_id="plan",
        group_id="impl-teams",
        score=4.5,
        score_explanation="Clean task breakdown with correct parallelism.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate team lead coordination: task decomposition, parallelism decisions, and delegation clarity. Score 1-5."},
    ))

    s.add_group(GroupRecord(group_id="be-impl", group_type="parallel", name="Backend Implementation", parent_span_id="be-lead"))

    s.add_action(ActionRecord(
        span_id="be-api",
        name="implement_api_endpoints",
        goal="Create all OAuth2 and RBAC API endpoints with proper validation",
        input_data='{"endpoints": 8, "framework": "FastAPI", "auth_spec": "OAuth2 + JWT"}',
        output_data='{"endpoints_created": 8, "lines_of_code": 420, "tests_written": 12, "openapi_spec_updated": true}',
        action_index=4, latency_ms=12500.0,
        parent_span_id="be-lead",
        group_id="be-impl",
        score=4.0,
        score_explanation="All endpoints implemented. Missing rate limiting on /auth/login.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate API implementation quality: correctness, security, error handling. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Clean implementation but missing rate limiting", name="code_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="be-db",
        name="implement_db_migrations",
        goal="Set up the data model with proper indexes and constraints",
        input_data='{"tables": ["roles", "permissions", "user_roles", "oauth_tokens", "audit_log"], "orm": "SQLAlchemy"}',
        output_data='{"migrations_created": 3, "tables_created": 5, "indexes": 8, "foreign_keys": 6}',
        action_index=5, latency_ms=6800.0,
        parent_span_id="be-lead",
        group_id="be-impl",
        score=4.5,
        score_explanation="Well-designed schema with proper normalization and indexes.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate database schema design: normalization, indexing strategy, and constraint correctness. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Proper schema design with cascading deletes", name="schema_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="be-auth",
        name="implement_auth_service",
        goal="Build the core authentication service with PKCE, token rotation, and session binding",
        input_data='{"oauth_providers": ["google", "github"], "token_type": "JWT", "session_store": "Redis"}',
        output_data='{"components": ["OAuthFlowHandler", "JWTService", "SessionManager", "PKCEVerifier"], "lines_of_code": 580}',
        action_index=6, latency_ms=15200.0,
        parent_span_id="be-lead",
        group_id="be-impl",
        score=3.8,
        score_explanation="Core auth works but session fixation vulnerability in session rotation logic.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate auth implementation for security best practices. Score 1-5."},
        judge_result=JudgeResult(score=3.5, explanation="Session fixation risk: old session not invalidated on role change", name="security", model="claude-sonnet-4-5-20250514"),
    ))

    s.add_action(ActionRecord(
        span_id="be-integrate",
        name="backend_integration",
        goal="Ensure all backend services work together end-to-end",
        input_data='{"components": ["api", "db", "auth_service"], "config": "dependency injection"}',
        output_data='{"integration_tests_passed": 18, "failed": 0, "coverage": "78%"}',
        action_index=7, latency_ms=8400.0,
        parent_span_id="be-lead",
        score=4.5,
        score_explanation="Clean integration with good test coverage.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate backend integration: component interoperability, test coverage, and error handling. Score 1-5."},
    ))

    # ── FRONTEND TEAM ─────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="fe-lead",
        name="frontend_lead",
        goal="Assign frontend tasks and establish component contracts",
        input_data='{"framework": "React", "design_system": "shadcn/ui", "pages": ["login", "callback", "settings/roles"]}',
        output_data='{"subtasks": ["ui_components", "auth_state_management"], "components_needed": 12}',
        action_index=8, latency_ms=900.0,
        parent_span_id="plan",
        group_id="impl-teams",
        score=4.0,
        score_explanation="Good component breakdown.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate frontend lead coordination: component planning and task assignment quality. Score 1-5."},
    ))

    s.add_group(GroupRecord(group_id="fe-impl", group_type="parallel", name="Frontend Implementation", parent_span_id="fe-lead"))

    s.add_action(ActionRecord(
        span_id="fe-ui",
        name="implement_auth_ui",
        goal="Create polished, accessible auth UI components",
        input_data='{"components": ["LoginPage", "OAuthCallback", "RoleManager", "PermissionGate", "UserMenu"], "design_system": "shadcn/ui"}',
        output_data='{"components_created": 8, "pages": 3, "accessibility_score": "AA", "storybook_stories": 6}',
        action_index=9, latency_ms=14200.0,
        parent_span_id="fe-lead",
        group_id="fe-impl",
        score=4.5,
        score_explanation="Beautiful, accessible components with good Storybook coverage.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate UI quality: accessibility, responsiveness, design consistency. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="WCAG AA compliant with consistent design language", name="ui_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="fe-state",
        name="implement_auth_state",
        goal="Create robust client-side auth state with automatic token refresh",
        input_data='{"state_lib": "zustand", "features": ["token_refresh", "permission_cache", "route_guards", "optimistic_logout"]}',
        output_data='{"stores_created": 2, "hooks": 5, "middleware": 1, "bundle_size_delta": "+4.2KB gzipped"}',
        action_index=10, latency_ms=9800.0,
        parent_span_id="fe-lead",
        group_id="fe-impl",
        score=4.0,
        score_explanation="Clean state management. Token refresh has a brief race condition window.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate client-side state management: token handling, race condition safety, and API consistency. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Good but token refresh race condition possible on slow networks", name="robustness", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="fe-integrate",
        name="frontend_integration",
        goal="Verify complete auth flow from login to protected route access",
        input_data='{"api_base": "/api/v1", "cors": true, "test_flows": ["login", "logout", "token_refresh", "role_check"]}',
        output_data='{"flows_tested": 4, "e2e_tests": 6, "all_passing": true}',
        action_index=11, latency_ms=7200.0,
        parent_span_id="fe-lead",
        score=4.0,
        score_explanation="Full flow works. E2E tests could cover more edge cases.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate frontend integration: end-to-end auth flow correctness and test coverage. Score 1-5."},
    ))

    # ── INFRA TEAM ────────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="infra-lead",
        name="infra_lead",
        goal="Ensure reliable deployment with proper secret management",
        input_data='{"cloud": "AWS", "ci": "GitHub Actions", "secrets": ["oauth_client_id", "oauth_client_secret", "jwt_signing_key"]}',
        output_data='{"subtasks": ["ci_pipeline", "deployment_config"], "environments": ["staging", "production"]}',
        action_index=12, latency_ms=800.0,
        parent_span_id="plan",
        group_id="impl-teams",
        score=4.5,
        score_explanation="Good infrastructure planning.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate infrastructure planning: secret management strategy and environment separation. Score 1-5."},
    ))

    s.add_group(GroupRecord(group_id="infra-impl", group_type="parallel", name="Infrastructure Setup", parent_span_id="infra-lead"))

    s.add_action(ActionRecord(
        span_id="infra-ci",
        name="configure_ci_pipeline",
        goal="Create a CI pipeline that catches issues before merge",
        input_data='{"stages": ["lint", "test", "security_scan", "build", "deploy_staging"]}',
        output_data='{"workflow_file": ".github/workflows/auth-service.yml", "stages": 5, "estimated_runtime": "4m30s"}',
        action_index=13, latency_ms=4500.0,
        parent_span_id="infra-lead",
        group_id="infra-impl",
        score=4.5,
        score_explanation="Comprehensive CI with security scanning.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate CI pipeline design: stage coverage, security scanning inclusion, and runtime efficiency. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="infra-deploy",
        name="configure_deployment",
        goal="Infrastructure-as-code for the auth service dependencies",
        input_data='{"resources": ["redis_cluster", "secrets_manager", "ecs_task_def", "alb_rules"]}',
        output_data='{"terraform_resources": 6, "secrets_configured": 3, "environments": ["staging", "prod"]}',
        action_index=14, latency_ms=6200.0,
        parent_span_id="infra-lead",
        group_id="infra-impl",
        score=4.0,
        score_explanation="Good IaC setup. Missing Redis failover configuration.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate IaC deployment config: resource completeness, HA setup, and secret management. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Solid but Redis HA config incomplete", name="infra_quality", model="gpt-4o"),
    ))

    # ── parallel: testing phase ───────────────────────────────
    s.add_group(GroupRecord(group_id="test-phase", group_type="parallel", name="Testing Phase", parent_span_id="plan"))

    s.add_action(ActionRecord(
        span_id="test-unit",
        name="run_unit_tests",
        goal="Verify all individual components work correctly in isolation",
        input_data='{"suites": ["backend_unit", "frontend_unit"], "total_tests": 86}',
        output_data='{"passed": 84, "failed": 2, "skipped": 0, "coverage": "82%", "failures": ["test_token_refresh_race", "test_session_rotation"]}',
        action_index=15, latency_ms=45000.0,
        parent_span_id="plan",
        group_id="test-phase",
        score=3.5,
        score_explanation="2 test failures pointing to real bugs in token refresh and session rotation.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate unit test results: pass rate, failure significance, and coverage adequacy. Score 1-5."},
        judge_result=JudgeResult(score=3.5, explanation="Test failures indicate genuine bugs that need fixing", name="test_results", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="test-integration",
        name="run_integration_tests",
        goal="Verify all services work together in a realistic environment",
        input_data='{"environment": "staging", "test_flows": ["full_oauth_flow", "rbac_enforcement", "token_lifecycle", "concurrent_sessions"]}',
        output_data='{"flows_tested": 4, "passed": 3, "failed": 1, "failure": "concurrent_sessions: second session not invalidated"}',
        action_index=16, latency_ms=62000.0,
        parent_span_id="plan",
        group_id="test-phase",
        score=3.5,
        score_explanation="Session management fails under concurrency.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate integration test results: scenario coverage, failure severity, and environment fidelity. Score 1-5."},
        judge_result=JudgeResult(score=3.5, explanation="Critical concurrency issue in session management", name="integration_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="test-security",
        name="run_security_scan",
        goal="Identify security vulnerabilities before production deployment",
        input_data='{"scanners": ["semgrep", "zap", "npm_audit", "custom_auth_checks"], "target": "staging"}',
        output_data='{"critical": 1, "high": 1, "medium": 3, "low": 5, "critical_detail": "Session fixation: old session token remains valid after re-authentication"}',
        action_index=17, latency_ms=38000.0,
        parent_span_id="plan",
        group_id="test-phase",
        score=4.5,
        score_explanation="Caught session fixation vulnerability. Comprehensive scan coverage.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate security test coverage and findings quality. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Found critical session fixation issue", name="scan_effectiveness", model="claude-sonnet-4-5-20250514"),
    ))

    # ── code reviewer ─────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="reviewer",
        name="code_reviewer",
        goal="Sign off on code quality and verify all issues are addressed",
        input_data='{"files_changed": 28, "security_findings": 1, "test_failures": 2, "focus": "session management fixes"}',
        output_data='{"verdict": "APPROVED_WITH_FIXES", "comments": 4, "blocking": 1, "blocking_detail": "Session fixation fix verified and correct"}',
        action_index=18, latency_ms=8500.0,
        parent_span_id="plan",
        score=4.0,
        score_explanation="Thorough review. Correctly prioritized the security fix.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate code review quality: finding prioritization, security focus, and actionability of feedback. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Good review prioritization but missed a minor edge case", name="review_quality", model="claude-sonnet-4-5-20250514"),
    ))

    # ── deployer ──────────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="deployer",
        name="deploy_to_production",
        goal="Deploy safely with the ability to roll back instantly",
        input_data='{"strategy": "blue_green", "canary_percent": 5, "canary_duration_min": 15}',
        output_data='{"deployment_id": "deploy-auth-v1.2.0-abc123", "canary_result": "healthy", "rollout": "100%", "latency_p99": "45ms"}',
        action_index=19, latency_ms=180000.0,
        parent_span_id="plan",
        score=4.5,
        score_explanation="Clean deployment with proper canary validation.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate deployment execution: rollout strategy, canary validation, and rollback readiness. Score 1-5."},
    ))

    _assign_timestamps(s)
    save_session(s)
    print(f"  Saved: {s.trace_id} ({len(s.actions)} actions, {len(s.groups)} groups)")


def _trace_rag_pipeline():
    """RAG pipeline: coordinator -> parallel expansion -> parallel retrieval -> rerank -> generate -> parallel quality checks -> refine."""
    s = TraceSession(
        trace_id="rag-pipeline-007",
        workflow_name="RAG Pipeline",
        goal="Answer complex technical question using retrieval-augmented generation with multi-source retrieval and quality validation",
        session_score=4.4,
        session_score_explanation="High-quality answer with good source coverage. Hallucination check passed. Minor relevance concern on one retrieved passage.",
        created_at="2026-03-16T16:30:00+00:00",
    )

    s.add_action(ActionRecord(
        span_id="rag-coord",
        name="query_coordinator",
        goal="Decompose query, plan retrieval strategy, set quality thresholds",
        input_data='{"query": "How does the transformer attention mechanism handle variable-length sequences, and what are the computational trade-offs compared to RNNs?", "mode": "comprehensive"}',
        output_data='{"complexity": "multi-part", "requires_expansion": true, "retrieval_sources": ["vector_store", "bm25_index", "knowledge_graph"]}',
        action_index=0, latency_ms=380.0,
        score=4.5,
        score_explanation="Good query decomposition with appropriate strategy selection.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate the coordinator's query analysis and strategy planning. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Correct identification of multi-part query requiring expansion", name="planning", model="claude-sonnet-4-5-20250514"),
    ))

    # ── query expansion (parallel) ─────────────────────────────
    s.add_group(GroupRecord(group_id="query-expand", group_type="parallel", name="Query Expansion", parent_span_id="rag-coord"))

    s.add_action(ActionRecord(
        span_id="expand-semantic",
        name="expand_semantic",
        goal="Create 3-5 semantic variants of the original query for broader recall",
        input_data='{"strategy": "semantic_similarity", "num_variants": 4}',
        output_data='{"variants": ["transformer self-attention mechanism sequence length handling", "attention score computation variable input lengths", "positional encoding role in sequence length variation", "multi-head attention computational complexity analysis"]}',
        action_index=1, latency_ms=420.0,
        parent_span_id="rag-coord",
        group_id="query-expand",
        score=4.5,
        score_explanation="Good semantic diversity in expanded queries.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate semantic query expansion: diversity, relevance, and recall potential of generated variants. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="expand-keyword",
        name="expand_keyword",
        goal="Generate keyword combinations for BM25 matching",
        input_data='{"strategy": "keyword_extraction", "include_synonyms": true}',
        output_data='{"keywords": ["transformer", "attention mechanism", "variable length", "sequence padding", "computational complexity", "RNN comparison", "self-attention", "quadratic scaling"]}',
        action_index=2, latency_ms=280.0,
        parent_span_id="rag-coord",
        group_id="query-expand",
        score=4.0,
        score_explanation="Good keyword coverage, could include more domain-specific terms.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate keyword extraction quality: term coverage, synonym inclusion, and BM25 matching potential. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="expand-hyde",
        name="expand_hypothetical",
        goal="Create a plausible answer passage to use as a dense retrieval query",
        input_data='{"strategy": "hypothetical_document_embedding"}',
        output_data='{"hypothetical_doc": "The transformer attention mechanism handles variable-length sequences through positional encoding and attention masking...", "confidence": 0.7}',
        action_index=3, latency_ms=890.0,
        parent_span_id="rag-coord",
        group_id="query-expand",
        score=4.0,
        score_explanation="Reasonable hypothetical document, captures key concepts.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate hypothetical document embedding: plausibility, concept coverage, and retrieval utility. Score 1-5."},
    ))

    # ── retrieval (parallel) ───────────────────────────────────
    s.add_group(GroupRecord(group_id="retrieval", group_type="parallel", name="Multi-Source Retrieval", parent_span_id="rag-coord"))

    s.add_action(ActionRecord(
        span_id="ret-vector",
        name="retrieve_vector_store",
        goal="Retrieve top-k passages by embedding similarity",
        input_data='{"index": "technical_docs_v3", "top_k": 10, "query_variants": 5}',
        output_data='{"passages_retrieved": 10, "avg_similarity": 0.84}',
        action_index=4, latency_ms=145.0,
        parent_span_id="rag-coord",
        group_id="retrieval",
        score=4.5,
        score_explanation="High-quality passages with good similarity scores.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate vector retrieval quality: relevance of returned documents and recall coverage. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="ret-bm25",
        name="retrieve_bm25",
        goal="Retrieve passages matching extracted keywords",
        input_data='{"index": "technical_docs_bm25", "top_k": 10, "keywords": 8}',
        output_data='{"passages_retrieved": 10, "avg_bm25_score": 12.3}',
        action_index=5, latency_ms=95.0,
        parent_span_id="rag-coord",
        group_id="retrieval",
        score=4.0,
        score_explanation="Good keyword matches. Some noise from overly broad terms.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate BM25 retrieval: precision of keyword matches and noise filtering. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="ret-kg",
        name="retrieve_knowledge_graph",
        goal="Find related concepts and relationships in the knowledge graph",
        input_data='{"entities": ["transformer", "attention", "RNN", "sequence_length"], "hops": 2}',
        output_data='{"subgraph_nodes": 24, "relationships": 3, "context_passages": 5}',
        action_index=6, latency_ms=320.0,
        parent_span_id="rag-coord",
        group_id="retrieval",
        score=4.5,
        score_explanation="Excellent graph traversal capturing key relationships.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate knowledge graph retrieval: relationship relevance and subgraph completeness. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Relevant subgraph with useful structural relationships", name="retrieval_quality", model="gpt-4o"),
    ))

    # ── rerank & fuse ──────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="rerank",
        name="rerank_and_fuse",
        goal="Select the top passages from all sources using learned relevance scoring",
        input_data='{"total_candidates": 25, "target_passages": 8, "model": "cross-encoder-ms-marco-v2"}',
        output_data='{"selected_passages": 8, "avg_rerank_score": 0.91, "dedup_removed": 3}',
        action_index=7, latency_ms=560.0,
        parent_span_id="rag-coord",
        score=4.5,
        score_explanation="Good fusion with appropriate source diversity.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate reranking and fusion: passage selection quality, deduplication, and source diversity. Score 1-5."},
    ))

    # ── generate answer ────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="generate",
        name="generate_answer",
        goal="Produce a detailed, well-cited answer addressing all parts of the query",
        input_data='{"context_passages": 8, "model": "claude-sonnet-4-5-20250514"}',
        output_data='{"answer_length": 450, "citations_used": 6, "sections": 4, "confidence": 0.88}',
        action_index=8, latency_ms=2400.0,
        parent_span_id="rag-coord",
        score=4.5,
        score_explanation="Comprehensive answer with clear structure and proper citations.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate answer quality: accuracy, completeness, citation usage, and clarity. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Accurate and well-structured with proper source attribution", name="answer_quality", model="claude-sonnet-4-5-20250514"),
    ))

    # ── quality checks (parallel) ──────────────────────────────
    s.add_group(GroupRecord(group_id="quality-checks", group_type="parallel", name="Quality Validation", parent_span_id="rag-coord"))

    s.add_action(ActionRecord(
        span_id="check-hallucination",
        name="check_hallucination",
        goal="Flag any statements not supported by the context",
        input_data='{"answer_sentences": 18, "context_passages": 8}',
        output_data='{"grounded": 17, "flagged": 1, "flagged_detail": "overgeneralized claim about relative positional encodings"}',
        action_index=9, latency_ms=1100.0,
        parent_span_id="rag-coord",
        group_id="quality-checks",
        score=4.5,
        score_explanation="Caught a subtle overgeneralization.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate hallucination detection: grounding accuracy and false-positive rate. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Thorough sentence-level verification", name="grounding_quality", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="check-relevance",
        name="check_relevance",
        goal="Ensure all parts of the multi-part query are answered",
        input_data='{"query_parts": ["attention mechanism variable-length handling", "computational trade-offs vs RNNs"]}',
        output_data='{"overall_relevance": 0.92, "suggestion": "Could strengthen the RNN comparison with specific benchmarks"}',
        action_index=10, latency_ms=780.0,
        parent_span_id="rag-coord",
        group_id="quality-checks",
        score=4.0,
        score_explanation="Good coverage assessment.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate relevance checking: query-answer alignment and multi-part coverage verification. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="check-complete",
        name="check_completeness",
        goal="Check if important subtopics or nuances are missing",
        input_data='{"expected_subtopics": ["padding_masking", "positional_encoding", "complexity_analysis", "memory_usage", "parallelization"]}',
        output_data='{"covered": 4, "missing": ["memory_usage"], "completeness_score": 0.85}',
        action_index=11, latency_ms=650.0,
        parent_span_id="rag-coord",
        group_id="quality-checks",
        score=4.0,
        score_explanation="Useful completeness check. Memory/KV cache is a valid gap.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate completeness checking: subtopic coverage assessment and gap identification. Score 1-5."},
    ))

    _assign_timestamps(s)
    save_session(s)
    print(f"  Saved: {s.trace_id} ({len(s.actions)} actions, {len(s.groups)} groups)")


def _trace_failed_data_pipeline():
    """A data pipeline run that partially fails."""
    s = TraceSession(
        trace_id="pipeline-etl-daily-0316",
        workflow_name="Daily ETL Pipeline",
        goal="Extract, transform, and load daily user analytics from 3 source systems into the data warehouse",
        session_score=2.0,
        session_score_explanation="Pipeline partially failed. 2 of 3 sources loaded successfully but the Salesforce extract failed due to an expired API token.",
        created_at="2026-03-16T06:00:00+00:00",
    )

    s.add_action(ActionRecord(
        span_id="etl-orch",
        name="pipeline_orchestrator",
        goal="Run all extraction, transformation, and loading steps",
        input_data='{"run_date": "2026-03-16", "sources": ["postgres_app_db", "salesforce_crm", "segment_events"]}',
        output_data='{"status": "partial_failure", "successful": 2, "failed": 1}',
        action_index=0, latency_ms=45000.0,
        score=2.0,
        score_explanation="Partial failure. Should have validated credentials before starting.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate pipeline orchestration: error handling, pre-flight validation, and graceful degradation. Score 1-5."},
    ))

    s.add_group(GroupRecord(group_id="extract-parallel", group_type="parallel", name="Source Extraction", parent_span_id="etl-orch"))

    s.add_action(ActionRecord(
        span_id="etl-pg",
        name="extract_postgres",
        goal="Pull all user events from the last 24 hours",
        input_data='{"source": "postgres_app_db", "table": "user_events"}',
        output_data='{"rows_extracted": 148293, "size_mb": 45.2}',
        action_index=1, latency_ms=8500.0,
        parent_span_id="etl-orch",
        group_id="extract-parallel",
        score=5.0,
        score_explanation="Clean extraction, expected row count.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate data extraction: completeness, row count validation, and performance. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="etl-sf",
        name="extract_salesforce",
        goal="Pull opportunity and account changes from the last 24 hours",
        input_data='{"source": "salesforce_crm", "objects": ["Opportunity", "Account"]}',
        output_data='{"error": "INVALID_SESSION_ID"}',
        action_index=2, latency_ms=1200.0,
        parent_span_id="etl-orch",
        group_id="extract-parallel",
        error="SalesforceAuthenticationError: INVALID_SESSION_ID — OAuth refresh token expired (last refreshed 2026-02-14).",
        score=1.0,
        score_explanation="Complete failure due to expired credentials.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate extraction failure handling: error messaging, credential validation, and recovery attempt. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="etl-seg",
        name="extract_segment",
        goal="Pull all tracked events from the last 24 hours",
        input_data='{"source": "segment_events"}',
        output_data='{"events_extracted": 523847, "size_mb": 189.3}',
        action_index=3, latency_ms=12000.0,
        parent_span_id="etl-orch",
        group_id="extract-parallel",
        score=5.0,
        score_explanation="Clean extraction at expected volume.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate event extraction: volume validation, data integrity, and throughput. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="etl-transform",
        name="transform_and_load",
        goal="Apply business logic transformations and load into warehouse tables",
        input_data='{"sources_available": ["postgres_app_db", "segment_events"], "sources_missing": ["salesforce_crm"]}',
        output_data='{"tables_loaded": 2, "tables_skipped": 1, "rows_loaded": 672140}',
        action_index=4, latency_ms=22000.0,
        parent_span_id="etl-orch",
        score=3.0,
        score_explanation="Correctly loaded available data but no alerting triggered.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate transform-and-load: partial failure handling, data quality, and alerting behavior. Score 1-5."},
    ))

    _assign_timestamps(s)
    save_session(s)
    print(f"  Saved: {s.trace_id} ({len(s.actions)} actions, {len(s.groups)} groups)")


def _trace_agentic_customer_support():
    """Complex customer support agent with nested tool calls and multi-step reasoning.

    Architecture:
      classifier -> parallel(sentiment_analysis, intent_extraction) ->
      router -> parallel(
        knowledge_search(parallel(internal_kb, external_docs, similar_tickets)),
        account_lookup(-> billing_api -> subscription_api)
      ) -> response_drafter -> parallel(tone_check, accuracy_check, compliance_check) ->
      response_polisher -> send_response
    """
    s = TraceSession(
        trace_id="support-complex-2847",
        workflow_name="Agentic Customer Support",
        goal="Resolve a complex customer escalation involving a billing discrepancy, feature request, and compliance question about data residency",
        session_score=4.3,
        session_score_explanation="All three customer concerns addressed. Billing issue resolved, feature request logged, and data residency question answered accurately with documentation link.",
        created_at="2026-03-16T14:05:00+00:00",
    )

    # ── classifier (root) ─────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="cs-classify",
        name="classify_ticket",
        goal="Route ticket to correct handling path",
        input_data='{"ticket_id": "ESC-2847", "subject": "Billing error + data residency question + feature request", "customer_tier": "enterprise"}',
        output_data='{"categories": ["billing_dispute", "compliance_inquiry", "feature_request"], "urgency": "high", "sentiment": "frustrated", "multi_intent": true}',
        action_index=0, latency_ms=650.0,
        score=5.0,
        score_explanation="Correctly identified all three intents in a multi-topic ticket.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate ticket classification accuracy. Score 1-5."},
        judge_result=JudgeResult(score=5.0, explanation="Perfect multi-intent classification with correct urgency", name="classification_accuracy", model="claude-sonnet-4-5-20250514"),
    ))

    # ── parallel: NLU analysis ────────────────────────────────
    s.add_group(GroupRecord(group_id="nlu-analysis", group_type="parallel", name="NLU Analysis", parent_span_id="cs-classify"))

    s.add_action(ActionRecord(
        span_id="cs-sentiment",
        name="analyze_sentiment",
        goal="Understand customer emotional state for appropriate response calibration",
        input_data='{"text": "ticket body...", "history": "3 previous interactions"}',
        output_data='{"sentiment": "frustrated_but_reasonable", "frustration_triggers": ["billing_error", "previous_unresolved_ticket"], "tone_recommendation": "empathetic_professional"}',
        action_index=1, latency_ms=320.0,
        parent_span_id="cs-classify",
        group_id="nlu-analysis",
        score=4.5,
        score_explanation="Accurate sentiment with good tone recommendation.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate sentiment analysis accuracy and tone recommendation appropriateness. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="cs-intent",
        name="extract_intents",
        goal="Parse all customer requests into actionable items",
        input_data='{"text": "ticket body..."}',
        output_data='{"intents": [{"type": "billing_dispute", "amount": "$340", "period": "February"}, {"type": "feature_request", "feature": "SSO SAML integration"}, {"type": "compliance_inquiry", "topic": "EU data residency", "regulation": "GDPR"}]}',
        action_index=2, latency_ms=450.0,
        parent_span_id="cs-classify",
        group_id="nlu-analysis",
        score=4.5,
        score_explanation="Clean intent extraction with correct entity linking.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate intent extraction: completeness, entity linking accuracy, and structural clarity. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="All intents correctly extracted with relevant entities", name="extraction_quality", model="gpt-4o"),
    ))

    # ── router ────────────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="cs-router",
        name="route_intents",
        goal="Create execution plan for resolving all customer intents",
        input_data='{"intents": 3, "customer_tier": "enterprise"}',
        output_data='{"plan": ["parallel_knowledge_search", "parallel_account_lookup", "sequential_resolution"], "estimated_steps": 8}',
        action_index=3, latency_ms=280.0,
        parent_span_id="cs-classify",
        score=4.5,
        score_explanation="Efficient routing with correct parallelization.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate intent routing: execution plan efficiency and parallelization decisions. Score 1-5."},
    ))

    # ── parallel: knowledge search + account lookup ───────────
    s.add_group(GroupRecord(group_id="data-gather", group_type="parallel", name="Data Gathering", parent_span_id="cs-router"))

    # Knowledge search sub-tree
    s.add_action(ActionRecord(
        span_id="cs-kb-search",
        name="knowledge_search",
        goal="Find documentation, precedents, and policies for all three customer intents",
        input_data='{"queries": ["billing adjustment policy", "SSO SAML roadmap", "EU data residency compliance"]}',
        output_data='{"sources_searched": 3, "results_found": 8}',
        action_index=4, latency_ms=180.0,
        parent_span_id="cs-router",
        group_id="data-gather",
        score=4.5,
        score_explanation="Good search coordination.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate knowledge search coordination: query formulation and source selection strategy. Score 1-5."},
    ))

    s.add_group(GroupRecord(group_id="kb-sources", group_type="parallel", name="Knowledge Sources", parent_span_id="cs-kb-search"))

    s.add_action(ActionRecord(
        span_id="cs-kb-internal",
        name="search_internal_kb",
        goal="Find billing adjustment policy, feature request process, and compliance docs",
        input_data='{"kb": "internal", "queries": 3}',
        output_data='{"articles_found": 4, "relevant": ["billing_adjustment_policy_v3", "feature_request_sla", "gdpr_data_residency_faq"]}',
        action_index=5, latency_ms=120.0,
        parent_span_id="cs-kb-search",
        group_id="kb-sources",
        score=4.5,
        score_explanation="Found all relevant internal docs.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate internal KB search: result relevance and coverage across all query intents. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="cs-kb-external",
        name="search_external_docs",
        goal="Find relevant help center articles and API docs",
        input_data='{"docs_site": "docs.example.com", "queries": 3}',
        output_data='{"articles_found": 3, "urls": ["docs.example.com/billing/adjustments", "docs.example.com/compliance/gdpr"]}',
        action_index=6, latency_ms=240.0,
        parent_span_id="cs-kb-search",
        group_id="kb-sources",
        score=4.0,
        score_explanation="Good external doc coverage.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate external documentation search: article relevance and URL accuracy. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="cs-kb-similar",
        name="search_similar_tickets",
        goal="Learn from how similar issues were previously resolved",
        input_data='{"similarity_search": true, "top_k": 5}',
        output_data='{"similar_tickets": 3, "best_match": {"ticket_id": "ESC-1923", "similarity": 0.87, "resolution": "billing_credit_applied"}}',
        action_index=7, latency_ms=350.0,
        parent_span_id="cs-kb-search",
        group_id="kb-sources",
        score=4.0,
        score_explanation="Found relevant precedent.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate similar ticket search: match quality and resolution applicability. Score 1-5."},
    ))

    # Account lookup sub-tree
    s.add_action(ActionRecord(
        span_id="cs-acct",
        name="lookup_account",
        goal="Gather full context on the billing situation and current subscription",
        input_data='{"account_id": "ENT-7392", "lookups": ["billing_history", "subscription_details"]}',
        output_data='{"account_found": true, "plan": "enterprise", "billing_discrepancy_confirmed": true}',
        action_index=8, latency_ms=180.0,
        parent_span_id="cs-router",
        group_id="data-gather",
        score=4.5,
        score_explanation="Efficient account lookup.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate account lookup: data retrieval completeness and response latency. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="cs-billing-api",
        name="query_billing_api",
        goal="Confirm the $340 overcharge and identify root cause",
        input_data='{"account_id": "ENT-7392", "period": "2026-02", "api": "billing_service"}',
        output_data='{"charges": [{"description": "Enterprise Plan", "amount": 2400}, {"description": "Add-on: Advanced Analytics (duplicate)", "amount": 340}], "root_cause": "Duplicate add-on charge from failed downgrade webhook"}',
        action_index=9, latency_ms=420.0,
        parent_span_id="cs-acct",
        score=4.5,
        score_explanation="Root cause correctly identified.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate billing API query: root cause identification accuracy and charge verification. Score 1-5."},
        judge_result=JudgeResult(score=4.5, explanation="Correct root cause analysis of the duplicate charge", name="accuracy", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="cs-sub-api",
        name="query_subscription_api",
        goal="Verify current plan features and SAML SSO availability",
        input_data='{"account_id": "ENT-7392", "check": ["current_features", "saml_sso_availability"]}',
        output_data='{"plan": "enterprise", "saml_sso": "available_on_enterprise_plus", "upgrade_path": "enterprise_plus", "price_delta": "+$600/mo"}',
        action_index=10, latency_ms=310.0,
        parent_span_id="cs-acct",
        score=4.5,
        score_explanation="Correct subscription info retrieved.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate subscription API query: feature availability accuracy and upgrade path correctness. Score 1-5."},
    ))

    # ── response drafter ──────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="cs-draft",
        name="draft_response",
        goal="Clear, empathetic response with actionable information for each concern",
        input_data='{"intents": 3, "tone": "empathetic_professional", "knowledge_sources": 8}',
        output_data='{"sections": ["billing_resolution", "feature_request_update", "compliance_answer"], "word_count": 320, "actions_taken": ["refund_initiated", "feature_request_logged", "compliance_doc_linked"]}',
        action_index=11, latency_ms=2800.0,
        parent_span_id="cs-classify",
        score=4.0,
        score_explanation="Good draft covering all concerns. Tone could be warmer.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate response draft: completeness of concern coverage, empathy, and actionability. Score 1-5."},
    ))

    # ── parallel: response quality checks ─────────────────────
    s.add_group(GroupRecord(group_id="response-checks", group_type="parallel", name="Response Quality Checks", parent_span_id="cs-draft"))

    s.add_action(ActionRecord(
        span_id="cs-tone-check",
        name="check_tone",
        goal="Ensure response is appropriately empathetic without being patronizing",
        input_data='{"response_draft": "...", "target_tone": "empathetic_professional"}',
        output_data='{"tone_score": 0.78, "suggestions": ["Add acknowledgment of frustration in opening", "Soften the upgrade upsell language"]}',
        action_index=12, latency_ms=380.0,
        parent_span_id="cs-draft",
        group_id="response-checks",
        score=3.5,
        score_explanation="Identified tone issues that needed fixing.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate tone check: detection of inappropriate tone and quality of suggestions. Score 1-5."},
        judge_result=JudgeResult(score=3.5, explanation="Draft was too transactional, needed more empathy", name="tone_quality", model="claude-sonnet-4-5-20250514"),
    ))

    s.add_action(ActionRecord(
        span_id="cs-accuracy-check",
        name="check_accuracy",
        goal="Ensure refund amount, feature availability, and compliance info are accurate",
        input_data='{"claims_to_verify": 6}',
        output_data='{"verified": 6, "flagged": 0, "all_accurate": true}',
        action_index=13, latency_ms=420.0,
        parent_span_id="cs-draft",
        group_id="response-checks",
        score=5.0,
        score_explanation="All claims verified against source data.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate accuracy verification: claim-to-source matching and factual correctness. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="cs-compliance-check",
        name="check_compliance",
        goal="Ensure response is compliant with internal policies and regulations",
        input_data='{"check": ["no_unauthorized_promises", "data_handling_compliant", "gdpr_accurate"]}',
        output_data='{"compliant": true, "notes": ["GDPR data residency answer correctly references EU region availability"]}',
        action_index=14, latency_ms=290.0,
        parent_span_id="cs-draft",
        group_id="response-checks",
        score=4.5,
        score_explanation="Compliance check passed.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate compliance check: policy adherence, regulatory accuracy, and promise validation. Score 1-5."},
    ))

    # ── response polisher ─────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="cs-polish",
        name="polish_response",
        goal="Produce the final, send-ready response",
        input_data='{"tone_feedback": 2, "accuracy_issues": 0, "compliance_issues": 0}',
        output_data='{"final_word_count": 350, "changes_made": 2, "tone_improvement": "Added empathetic opening, softened upsell"}',
        action_index=15, latency_ms=1200.0,
        parent_span_id="cs-classify",
        score=4.5,
        score_explanation="Well-polished final response.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate response polishing: tone improvements, clarity enhancements, and overall readability. Score 1-5."},
    ))

    # ── send response ─────────────────────────────────────────
    s.add_action(ActionRecord(
        span_id="cs-send",
        name="send_response",
        goal="Deliver response and update CRM",
        input_data='{"channel": "email", "ticket_id": "ESC-2847"}',
        output_data='{"sent": true, "ticket_status": "pending_customer", "refund_initiated": true, "feature_request_id": "FR-4521"}',
        action_index=16, latency_ms=450.0,
        parent_span_id="cs-classify",
        score=5.0,
        score_explanation="Response sent, all systems updated.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate response delivery: channel correctness, CRM update completeness, and follow-up actions. Score 1-5."},
    ))

    _assign_timestamps(s)
    save_session(s)
    print(f"  Saved: {s.trace_id} ({len(s.actions)} actions, {len(s.groups)} groups)")


def _trace_code_review():
    """Code review agent with parallel review tracks and nested analysis."""
    s = TraceSession(
        trace_id="code-review-042",
        workflow_name="Code Review Agent",
        goal="Review PR #287: Add rate limiting middleware to API gateway",
        session_score=3.8,
        session_score_explanation="Caught the Redis connection leak and missing tests, but missed the race condition in the token bucket implementation.",
        created_at="2026-03-16T11:30:00+00:00",
    )

    s.add_action(ActionRecord(
        span_id="cr-parse",
        name="parse_diff",
        goal="Build a structured representation of all changes",
        input_data='{"pr_number": 287, "files_changed": 5, "additions": 342, "deletions": 28}',
        output_data='{"files": 5, "languages": ["python"], "complexity": "medium"}',
        action_index=0, latency_ms=800.0,
        score=4.5,
        score_explanation="Accurate diff parsing with good context extraction.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate diff parsing: accuracy of change detection and context extraction quality. Score 1-5."},
    ))

    s.add_group(GroupRecord(group_id="review-parallel", group_type="parallel", name="Review Tracks", parent_span_id="cr-parse"))

    s.add_action(ActionRecord(
        span_id="cr-logic",
        name="review_logic",
        goal="Identify logical errors, edge cases, and correctness issues",
        input_data='{"focus_files": ["middleware/rate_limit.py"]}',
        output_data='{"findings": 2, "high_severity": 1}',
        action_index=1, latency_ms=3200.0,
        parent_span_id="cr-parse",
        group_id="review-parallel",
        score=3.5,
        score_explanation="Found the connection leak but missed the race condition.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate logic review thoroughness: bug detection rate and edge case coverage. Score 1-5."},
        judge_result=JudgeResult(score=3.5, explanation="Good catch on connection leak but missed TOCTOU race", name="thoroughness", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="cr-security",
        name="review_security",
        goal="Identify security issues: injection, auth bypass, data exposure",
        input_data='{"focus": "rate limiting bypass, Redis security"}',
        output_data='{"findings": 2, "vulnerabilities_found": 2}',
        action_index=2, latency_ms=2800.0,
        parent_span_id="cr-parse",
        group_id="review-parallel",
        score=4.0,
        score_explanation="Good catch on header spoofing.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate security review: vulnerability detection coverage and severity assessment accuracy. Score 1-5."},
        judge_result=JudgeResult(score=4.0, explanation="Identified the most critical security concern", name="security_coverage", model="gpt-4o"),
    ))

    s.add_action(ActionRecord(
        span_id="cr-style",
        name="review_style",
        goal="Ensure code follows project conventions",
        input_data='{"style_guide": "pep8 + project conventions", "files": 5}',
        output_data='{"findings": 2, "style_score": "good"}',
        action_index=3, latency_ms=1500.0,
        parent_span_id="cr-parse",
        group_id="review-parallel",
        score=4.0,
        score_explanation="Appropriate style feedback without being pedantic.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate style review: convention adherence checking and feedback appropriateness. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="cr-tests",
        name="review_test_coverage",
        goal="Identify missing test cases and assess existing test quality",
        input_data='{"test_file": "tests/test_rate_limit.py", "test_count": 8}',
        output_data='{"coverage_estimate": "65%", "missing_tests": 4}',
        action_index=4, latency_ms=2100.0,
        parent_span_id="cr-parse",
        group_id="review-parallel",
        score=4.0,
        score_explanation="Good identification of missing test scenarios.",
        judge_config={"provider": "AnthropicProvider", "model": "claude-sonnet-4-5-20250514", "system_prompt": "Evaluate test coverage review: gap identification accuracy and test quality assessment. Score 1-5."},
    ))

    s.add_action(ActionRecord(
        span_id="cr-summary",
        name="summarize_review",
        goal="Produce a clear, actionable review with priority-ordered findings",
        input_data='{"total_findings": 8, "high": 1, "medium": 2, "low": 3}',
        output_data='{"verdict": "REQUEST_CHANGES", "comment_count": 6}',
        action_index=5, latency_ms=1800.0,
        parent_span_id="cr-parse",
        score=4.5,
        score_explanation="Clear, well-prioritized summary.",
        judge_config={"provider": "OpenAIProvider", "model": "gpt-4o", "system_prompt": "Evaluate review summary: finding prioritization, verdict justification, and actionability. Score 1-5."},
    ))

    _assign_timestamps(s)
    save_session(s)
    print(f"  Saved: {s.trace_id} ({len(s.actions)} actions, {len(s.groups)} groups)")


if __name__ == "__main__":
    print("Seeding Lattice demo traces...")
    _trace_deep_research()
    _trace_multi_agent_coding_pipeline()
    _trace_rag_pipeline()
    _trace_failed_data_pipeline()
    _trace_agentic_customer_support()
    _trace_code_review()
    print("Done! Run: python -m lattice dashboard")
