"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from researchbot.agent.context import ContextBuilder
from researchbot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from researchbot.agent.memory import MemoryConsolidator
from researchbot.agent.runner import AgentRunSpec, AgentRunner
from researchbot.agent.subagent import SubagentManager
from researchbot.agent.tools.cron import CronTool
from researchbot.agent.skills import BUILTIN_SKILLS_DIR
from researchbot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from researchbot.agent.tools.message import MessageTool
from researchbot.agent.tools.registry import ToolRegistry
from researchbot.agent.tools.shell import ExecTool
from researchbot.agent.tools.spawn import SpawnTool
from researchbot.agent.tools.web import WebFetchTool, WebSearchTool
from researchbot.bus.events import InboundMessage, OutboundMessage
from researchbot.command import CommandContext, CommandRouter, register_builtin_commands
from researchbot.bus.queue import MessageBus
from researchbot.providers.base import LLMProvider
from researchbot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from researchbot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from researchbot.cron.service import CronService


class _LoopHook(AgentHook):
    """Core lifecycle hook for the main agent loop.

    Handles streaming delta relay, progress reporting, tool-call logging,
    and think-tag stripping for the built-in agent path.
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> None:
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from researchbot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean):]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        context.tool_calls = self._loop._rewrite_exec_innovation_calls(
            context.tool_calls,
            context.messages,
        )

        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class _LoopHookChain(AgentHook):
    """Run the core loop hook first, then best-effort extra hooks.

    This preserves the historical failure behavior of ``_LoopHook`` while still
    letting user-supplied hooks opt into ``CompositeHook`` isolation.
    """

    __slots__ = ("_primary", "_extras")

    def __init__(self, primary: AgentHook, extra_hooks: list[AgentHook]) -> None:
        self._primary = primary
        self._extras = CompositeHook(extra_hooks)

    def wants_streaming(self) -> bool:
        return self._primary.wants_streaming() or self._extras.wants_streaming()

    async def before_iteration(self, context: AgentHookContext) -> None:
        await self._primary.before_iteration(context)
        await self._extras.before_iteration(context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._primary.on_stream(context, delta)
        await self._extras.on_stream(context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._primary.on_stream_end(context, resuming=resuming)
        await self._extras.on_stream_end(context, resuming=resuming)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._primary.before_execute_tools(context)
        await self._extras.before_execute_tools(context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._primary.after_iteration(context)
        await self._extras.after_iteration(context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        content = self._primary.finalize_content(context, content)
        return self._extras.finalize_content(context, content)


# Pre-compiled regex for extracting workspace Path from exec commands
_RE_WS_FROM_PATH_CALL = re.compile(r"workspace\s*=\s*Path\(['\"]([^'\"]+)['\"]\)")


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        hooks: list[AgentHook] | None = None,
        literature_config: Any = None,
        innovation_config: Any = None,
        config: Any = None,
    ):
        from researchbot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.innovation_config = innovation_config
        self._config = config
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.literature_config = literature_config
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(workspace, timezone=timezone)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # RESEARCHBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        # Also reads NANOBOT_MAX_CONCURRENT_REQUESTS for backward compatibility.
        _max = int(os.environ.get("RESEARCHBOT_MAX_CONCURRENT_REQUESTS")
                   or os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        from researchbot.agent.tools.paper import (
            PaperCompareTool,
            PaperDownloadPdfTool,
            PaperExtractTextTool,
            PaperGetTool,
            PaperReviewTool,
            PaperSaveTool,
            PaperSearchTool,
            PaperSummarizeTool,
        )
        from researchbot.agent.tools.paper_enrich import (
            PaperEnrichTool,
            CrossrefSearchTool,
            OpenAlexSearchTool,
        )
        from researchbot.agent.tools.concept_explore import ConceptExploreTool
        from researchbot.agent.tools.paper_search_local import (
            PaperSearchLocalTool,
            PaperIndexTool,
            GraphQueryTool,
            KnowledgeGraphRebuildTool,
        )
        from researchbot.agent.tools.paper_cite import PaperCiteTool
        from researchbot.agent.tools.paper_recommend import PaperRecommendTool
        from researchbot.agent.tools.innovation import InnovationWorkflowTool
        from researchbot.agent.tools.research_gap_discovery import ResearchGapDiscoveryTool
        from researchbot.agent.tools.literature_survey import LiteratureSurveyTool
        semantic_config = getattr(self.literature_config, "semantic_search", None) if self.literature_config else None

        # Crossref/OpenAlex config for enrichment tools
        crossref_config = getattr(self.literature_config, "crossref", None) if self.literature_config else None
        openalex_config = getattr(self.literature_config, "openalex", None) if self.literature_config else None
        semantic_scholar_config = getattr(self.literature_config, "semantic_scholar", None) if self.literature_config else None

        crossref_mailto = getattr(crossref_config, "mailto", None) if crossref_config else None
        crossref_ua = getattr(crossref_config, "user_agent", None) if crossref_config else None
        openalex_api_key = getattr(openalex_config, "api_key", None) if openalex_config else None
        semantic_scholar_api_key = getattr(semantic_scholar_config, "api_key", None) if semantic_scholar_config else None

        self.tools.register(PaperSearchTool(proxy=self.web_proxy, mailto=crossref_mailto, semantic_scholar_api_key=semantic_scholar_api_key))
        self.tools.register(PaperGetTool(proxy=self.web_proxy))
        self.tools.register(PaperSaveTool(workspace=str(self.workspace), semantic_config=semantic_config, config=self._config, provider=self.provider))
        self.tools.register(PaperSummarizeTool(provider=self.provider, workspace=str(self.workspace), semantic_config=semantic_config, proxy=self.web_proxy))
        self.tools.register(PaperDownloadPdfTool(workspace=str(self.workspace), proxy=self.web_proxy))
        self.tools.register(PaperExtractTextTool(workspace=str(self.workspace), proxy=self.web_proxy))
        self.tools.register(PaperCompareTool(provider=self.provider, workspace=str(self.workspace), semantic_config=semantic_config, proxy=self.web_proxy))
        self.tools.register(PaperReviewTool(provider=self.provider, workspace=str(self.workspace), semantic_config=semantic_config, proxy=self.web_proxy))
        self.tools.register(InnovationWorkflowTool(provider=self.provider, workspace=str(self.workspace), semantic_config=semantic_config, innovation_config=self.innovation_config, config=self._config, proxy=self.web_proxy))
        gap_config = getattr(self.literature_config, "gap_discovery", None) if self.literature_config else None
        self.tools.register(ResearchGapDiscoveryTool(
            workspace=str(self.workspace),
            config=gap_config,
        ))
        self.tools.register(PaperRecommendTool(
            workspace=str(self.workspace),
            semantic_config=semantic_config,
            recommendation_config=getattr(self.literature_config, "recommendation", None) if self.literature_config else None,
        ))

        # Register Crossref/OpenAlex enrichment tools
        self.tools.register(CrossrefSearchTool(mailto=crossref_mailto, user_agent=crossref_ua, proxy=self.web_proxy))
        self.tools.register(OpenAlexSearchTool(api_key=openalex_api_key, proxy=self.web_proxy))
        self.tools.register(PaperEnrichTool(
            crossref_mailto=crossref_mailto,
            openalex_api_key=openalex_api_key,
            workspace=str(self.workspace),
            semantic_config=semantic_config,
            proxy=self.web_proxy,
        ))
        self.tools.register(ConceptExploreTool(
            workspace=str(self.workspace),
            semantic_config=semantic_config,
            openalex_api_key=openalex_api_key,
            proxy=self.web_proxy,
        ))

        # Register local semantic search tools
        self.tools.register(PaperSearchLocalTool(
            workspace=str(self.workspace),
            semantic_config=semantic_config,
        ))
        self.tools.register(PaperIndexTool(
            workspace=str(self.workspace),
            semantic_config=semantic_config,
        ))
        self.tools.register(GraphQueryTool(
            workspace=str(self.workspace),
            semantic_config=semantic_config,
        ))
        self.tools.register(KnowledgeGraphRebuildTool(
            workspace=str(self.workspace),
            semantic_config=semantic_config,
        ))

        # Register citation export tool
        self.tools.register(PaperCiteTool(workspace=str(self.workspace)))

        # Register literature survey tool
        survey_config = getattr(self.literature_config, "survey", None) if self.literature_config else None
        self.tools.register(LiteratureSurveyTool(
            workspace=str(self.workspace),
            config=survey_config,
        ))

        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from researchbot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from researchbot.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        """Flatten a message content payload into plain text for skill detection."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    def _infer_skill_names(self, history: list[dict[str, Any]], current_message: str) -> list[str]:
        """Infer which skill docs should be loaded for the current turn."""
        corpus_parts = [current_message]
        for msg in history[-6:]:
            corpus_parts.append(self._message_content_to_text(msg.get("content")))

        corpus = "\n".join(part for part in corpus_parts if part).lower()
        if not corpus:
            return []

        innovation_terms = (
            "innovation_workflow",
            "innovation workflow",
            "innovative",
            "创新点",
            "创新点生成",
            "创新点工作流",
            "创新方向",
            "查新",
            "novelty search",
        )
        if any(term in corpus for term in innovation_terms):
            return ["innovation"]

        survey_terms = (
            "调研", "文献调研", "literature survey", "survey",
            "研究综述", "领域调研", "survey paper", "做调研",
        )
        if any(term in corpus for term in survey_terms):
            return ["literature_survey"]
        return []

    @staticmethod
    def _extract_innovation_topic(text: str) -> str | None:
        """Extract topic from common innovation request phrasings."""
        if not text:
            return None

        text = text.strip()
        patterns = [
            r"以(.+?)的创新点",
            r"关于(.+?)的创新点",
            r"帮我想(.+?)的创新点",
            r"帮我想以(.+?)为baseline",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                topic = m.group(1).strip(" ，。,.!?！？")
                if topic:
                    return topic
        return None

    @staticmethod
    def _parse_exec_innovation_args(command: str) -> dict[str, Any]:
        """Parse innovation_workflow(...) args from an exec command string."""
        params: dict[str, Any] = {}

        topic_match = re.search(r'topic\s*=\s*["\']([^"\']+)["\']', command, re.IGNORECASE)
        if topic_match:
            params["topic"] = topic_match.group(1).strip()

        int_keys = (
            "num_candidates", "max_related", "top_k", "max_rounds",
            "min_proceed", "revise_top_k", "landscape_max_online",
        )
        for key in int_keys:
            m = re.search(rf"{key}\s*=\s*(\d+)", command, re.IGNORECASE)
            if m:
                params[key] = int(m.group(1))

        bool_keys = (
            "search_local", "search_online", "enable_review", "overwrite",
            "enable_iteration", "stop_if_no_change", "enable_landscape",
        )
        for key in bool_keys:
            m = re.search(rf"{key}\s*=\s*(true|false)", command, re.IGNORECASE)
            if m:
                params[key] = m.group(1).lower() == "true"

        return params

    def _rewrite_exec_innovation_calls(
        self,
        tool_calls: list[Any],
        messages: list[dict[str, Any]],
    ) -> list[Any]:
        """Rewrite exec-based innovation workflow calls into direct innovation_workflow tool calls.

        Detects when the Agent has both an exec innovation call AND a direct innovation_workflow
        tool call for the same topic, and deduplicates by dropping the exec rewrite when a
        direct call already exists.
        """
        from researchbot.providers.base import ToolCallRequest

        rewritten: list[Any] = []

        latest_user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_text = self._message_content_to_text(m.get("content"))
                if latest_user_text:
                    break

        # Extract direct innovation_workflow topics and check for existence in one pass
        direct_topics: set[str] = set()
        has_direct_call = False
        for tc in tool_calls:
            if tc.name == "innovation_workflow":
                has_direct_call = True
                args = tc.arguments if isinstance(tc.arguments, dict) else {}
                topic = str(args.get("topic", "")).lower().strip()
                if topic:
                    direct_topics.add(topic)

        for tc in tool_calls:
            if tc.name != "exec":
                rewritten.append(tc)
                continue

            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            command = str(args.get("command", ""))
            command_lower = command.lower()
            if "innovation_workflow" not in command_lower and "skills.innovation.workflow" not in command_lower:
                rewritten.append(tc)
                continue

            parsed = self._parse_exec_innovation_args(command)
            if not parsed.get("topic"):
                inferred_topic = self._extract_innovation_topic(latest_user_text)
                if inferred_topic:
                    parsed["topic"] = inferred_topic

            if not parsed.get("topic"):
                rewritten.append(tc)
                continue

            # If a direct innovation_workflow call exists for the same topic, skip rewrite
            # to avoid duplicate execution. Let the direct call (which Agent authored) win.
            topic_lower = parsed.get("topic", "").lower().strip()
            if has_direct_call and topic_lower in direct_topics:
                logger.info("Skipping exec rewrite: direct innovation_workflow call already exists for topic: {}", parsed.get("topic"))
                continue

            # Keep behavior close to exec wrapper: prefer fresh generation.
            parsed.setdefault("overwrite", True)

            rewritten.append(ToolCallRequest(
                id=tc.id,
                name="innovation_workflow",
                arguments=parsed,
                extra_content=tc.extra_content,
                provider_specific_fields=tc.provider_specific_fields,
                function_provider_specific_fields=tc.function_provider_specific_fields,
            ))
            logger.info("Rerouted exec innovation call to innovation_workflow with params: {}", parsed)

        # ------------------------------------------------------------
        # Knowledge graph rebuild rewrite
        # If the user is asking to rebuild the graph and the LLM generated
        # an exec call instead of the knowledge_graph_rebuild tool,
        # rewrite it to use the correct tool.
        # ------------------------------------------------------------
        kg_rebuild_keywords = [
            "重建图谱", "重建知识图谱", "rebuild graph", "rebuild the graph",
            "知识图谱重建", "kg rebuild", "knowledge graph rebuild",
        ]
        latest_user_text_lower = latest_user_text.lower()
        user_asked_kg_rebuild = any(kw in latest_user_text_lower for kw in kg_rebuild_keywords)

        has_direct_kg_call = any(tc.name == "knowledge_graph_rebuild" for tc in tool_calls)

        if user_asked_kg_rebuild and not has_direct_kg_call:
            for tc in tool_calls:
                if tc.name != "exec":
                    continue
                args = tc.arguments if isinstance(tc.arguments, dict) else {}
                command = str(args.get("command", "")).lower()
                kg_indicators = [
                    "知识图谱", "knowledge graph", "kg.", "kg_build",
                    "rebuild_from_papers", "literature/papers",
                    "重建图谱", "重建知识", "rebuild kg",
                ]
                if any(ind in command for ind in kg_indicators):
                    # Rewrite to knowledge_graph_rebuild tool
                    kg_args: dict[str, Any] = {}
                    # Try to extract workspace from the exec command
                    ws_match = _RE_WS_FROM_PATH_CALL.search(command)
                    if ws_match:
                        kg_args["workspace"] = ws_match.group(1)
                    rewritten.append(ToolCallRequest(
                        id=tc.id,
                        name="knowledge_graph_rebuild",
                        arguments=kg_args,
                        extra_content=tc.extra_content,
                        provider_specific_fields=tc.provider_specific_fields,
                        function_provider_specific_fields=tc.function_provider_specific_fields,
                    ))
                    logger.info("Rerouted exec knowledge-graph-rebuild call to knowledge_graph_rebuild tool")
                    continue

        return rewritten

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
        )
        hook: AgentHook = (
            _LoopHookChain(loop_hook, self._extra_hooks)
            if self._extra_hooks
            else loop_hook
        )

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        meta = dict(msg.metadata or {})
                        meta["_stream_delta"] = True
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta,
                            metadata=meta,
                        ))

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        meta = dict(msg.metadata or {})
                        meta["_stream_end"] = True
                        meta["_resuming"] = resuming
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="",
                            metadata=meta,
                        ))
                        stream_segment += 1

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            skill_names = self._infer_skill_names(history, msg.content)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                skill_names=skill_names,
                current_role=current_role,
            )
            final_content, _, all_msgs = await self._run_agent_loop(
                messages, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        skill_names = self._infer_skill_names(history, msg.content)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            skill_names=skill_names,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None:
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=meta,
        )

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        # Pre-process tool-call-like syntax into natural language
        content = self._preprocess_tool_syntax(content)
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
        )

    @staticmethod
    def _preprocess_tool_syntax(content: str) -> str:
        """Convert tool-call-like syntax into natural language for the LLM.

        Also handles "再换几个" / "别的" / "重新生成" style regeneration requests
        and converts them to explicit tool-call instructions with overwrite=True.
        """
        content_stripped = content.strip()

        # Regeneration trigger phrases → force overwrite=True for innovation_workflow
        innovation_regen_patterns = (
            "再换一个", "换一个", "再换几个", "别的", "来个别的", "换个方向",
            "再帮我想", "重新生成", "重来一版", "再来一版",
            "换一批", "再来几个", "再想几个", "再生成几个",
            "different ideas", "another batch", "more ideas",
            "regenerate", "generate more",
        )
        if any(pat in content_stripped for pat in innovation_regen_patterns):
            # Extract topic from conversation context if present
            # Look for recent topic keywords
            topic_hint = ""
            topic_keywords = (
                "弱监督视频异常检测", "vadclip", "VADClip",
                "weakly supervised", "video anomaly detection",
                "large language model", "LLM", "graph neural network",
                "GNN", "retrieval", "RAG", "multimodal",
            )
            for kw in topic_keywords:
                if kw.lower() in content_stripped.lower():
                    topic_hint = kw
                    break
            if topic_hint:
                return (
                    f"使用innovation_workflow topic=\"{topic_hint}\" overwrite=True "
                    f"重新生成不同于之前的创新点候选"
                )
            else:
                return (
                    "使用innovation_workflow overwrite=True 重新生成创新点候选，"
                    "确保生成与之前不同的角度和方向"
                )

        # Match patterns like: innovation_workflow topic="..." num_candidates=5
        # or: paper_search query="..." max_results=10
        tool_call_pattern = r'^(\w+)\s+(topic|query|message)=["\']([^"\']+)["\']?\s*(.*)?$'
        match = re.match(tool_call_pattern, content_stripped, re.IGNORECASE)
        if match:
            tool_name = match.group(1)
            primary_arg = match.group(2)
            primary_value = match.group(3)
            extra_args = match.group(4) or ""

            # Map tool names to natural language
            if tool_name == "innovation_workflow":
                # Parse extra arguments
                params = {}
                if extra_args:
                    for param in re.finditer(r'(\w+)=([^\s]+)', extra_args):
                        params[param.group(1)] = param.group(2).strip('"\'')
                num = params.get('num_candidates', '5')
                rounds = params.get('max_rounds', '')
                iteration = params.get('enable_iteration', '')

                parts = [f"使用innovation_workflow生成{num}个关于{primary_value}的创新点"]
                if iteration.lower() == 'true' if isinstance(iteration, str) else iteration:
                    parts.append(f"迭代{rounds or '多'}轮")
                return "".join(parts)

            elif tool_name == "paper_search":
                params = {}
                if extra_args:
                    for param in re.finditer(r'(\w+)=([^\s]+)', extra_args):
                        params[param.group(1)] = param.group(2).strip('"\'')
                max_results = params.get('max_results', '10')
                return f"搜索论文，关键词是{primary_value}，最多返回{max_results}个结果"

        return content
