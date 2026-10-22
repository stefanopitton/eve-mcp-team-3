"""Connect to MCP servers: a local script, or one from the registry clone."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastmcp import Client, FastMCP
from fastmcp.client.transports import PythonStdioTransport, StreamableHttpTransport

from .env import REPO_ROOT, settings

MY_SERVER = REPO_ROOT / "examples" / "geocode-server" / "server.py"


def local_server_client(script: str | Path = MY_SERVER) -> Client:
    """An MCP client that starts a server script as a subprocess (stdio transport)."""
    script = Path(script)
    return Client(PythonStdioTransport(script, args=["--transport", "stdio"], cwd=str(script.parent),
                                       python_cmd=sys.executable))


def registry_server_client(name: str) -> Client:
    """An MCP client for a server of the public registry, run locally from ~/mcp-tool-registry."""
    script = settings.registry_dir / "servers" / name / "server.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found. Is the registry cloned at {settings.registry_dir}?")
    return local_server_client(script)


def eve_mcp_client(name: str) -> Client:
    """An MCP client for a registry server hosted remotely, reached through EVE's MCP gateway."""
    if not settings.eve_configured:
        from .eve import EveUnavailable

        raise EveUnavailable("EVE is not configured: remote MCP servers are not available.")
    transport = StreamableHttpTransport(
        f"{settings.eve_mcp_base_url}/{name}",
        headers={"Authorization": f"Bearer {settings.eve_api_key}"},
    )
    return Client(transport, timeout=180)


def _schema_type(schema: Any) -> str:
    """A short type label for one JSON Schema node.

    Optional and union parameters are published as ``anyOf`` / ``oneOf`` rather than
    a top-level ``type``. See JSON Schema combining keywords:
    https://json-schema.org/understanding-json-schema/reference/combining
    """
    if not isinstance(schema, dict):
        return "unknown"
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            return " | ".join(_schema_type(option) for option in options)
    type_name = schema.get("type")
    if isinstance(type_name, list):
        type_name = " | ".join(str(part) for part in type_name)
    enum = schema.get("enum")
    enum_str = ""
    if isinstance(enum, list) and enum:
        enum_str = " {" + ", ".join(repr(value) for value in enum) + "}"
    if type_name == "array":
        items = schema.get("items")
        label = f"array[{_schema_type(items)}]" if isinstance(items, dict) else "array"
        return label + enum_str
    if isinstance(type_name, str):
        return type_name + enum_str
    if enum_str:
        return enum_str.strip()
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref:
        return ref.rsplit("/", 1)[-1]
    return "unknown"


def _tool_params_html(tool) -> str:
    schema = getattr(tool, "inputSchema", None) or {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict) or not props:
        return ""
    required = set(schema.get("required") or [])
    rows = []
    for name, info in props.items():
        if not isinstance(info, dict):
            info = {}
        marker = "required" if name in required else "optional"
        default = info.get("default")
        default_html = ""
        if default is not None:
            default_html = (
                f' <span style="color:#64748b">default {html.escape(repr(default))}</span>'
            )
        rows.append(
            '<div style="font-size:0.78rem;margin:0.15rem 0">'
            f'<code style="font-size:0.78rem">{html.escape(str(name))}</code> '
            f'<span style="color:#475569">{html.escape(_schema_type(info))}</span> '
            f'<span style="color:#94a3b8">{marker}</span>'
            f"{default_html}</div>"
        )
    return (
        '<div style="margin-top:0.45rem;font-size:0.68rem;letter-spacing:0.06em;'
        'color:#64748b">ARGUMENTS</div>' + "".join(rows)
    )


# One color per server, then the list repeats. Inline styles only: Jupyter's HTML
# sanitizer keeps these properties and drops <script> plus live <input>.
# https://github.com/jupyterlab/jupyterlab/blob/main/packages/apputils/src/sanitizer.ts
_SERVER_COLORS = (
    "#0f766e",
    "#1d4ed8",
    "#b45309",
    "#6d28d9",
    "#be123c",
    "#047857",
    "#0369a1",
    "#c2410c",
)


def _mix_white(hex_color: str, weight: float = 0.16) -> str:
    """Blend a hex color toward white. ``weight`` is how much of the color remains."""
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(channel * weight + 255 * (1 - weight)) for channel in channels]
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _tool_html(tool, color: str) -> str:
    description = (getattr(tool, "description", None) or "").strip()
    body = ""
    if description:
        body += (
            '<div style="white-space:pre-wrap;font-size:0.8rem;color:#334155;margin:0.3rem 0 0">'
            f"{html.escape(description)}</div>"
        )
    body += _tool_params_html(tool)
    return (
        f'<div style="margin:0.45rem 0;padding:0.55rem 0.7rem;border-radius:10px;'
        f'border:1px solid {_mix_white(color, 0.42)};background:#ffffff">'
        f'<div style="font-weight:700;font-size:0.88rem;color:{color}">{html.escape(tool.name)}</div>'
        f"{body}</div>"
    )


def _server_html(server: dict, batch, color: str) -> str:
    name = html.escape(str(server.get("name") or ""))
    description = (server.get("description") or "").strip()
    tint = _mix_white(color, 0.14)
    if isinstance(batch, Exception):
        count = "unavailable"
        inner = (
            f'<p style="margin:0.15rem 0 0.35rem;font-size:0.8rem;color:#b45309">{html.escape(str(batch))}</p>'
        )
    else:
        tools = list(batch or [])
        count = f"{len(tools)} tool" if len(tools) == 1 else f"{len(tools)} tools"
        inner = ""
        if description:
            inner += (
                f'<p style="margin:0.1rem 0 0.35rem;font-size:0.8rem;color:#475569">'
                f"{html.escape(description)}</p>"
            )
        inner += (
            f'<div style="font-size:0.68rem;letter-spacing:0.06em;color:{color};'
            'font-weight:700;margin:0.45rem 0 0.1rem">TOOLS</div>'
        )
        inner += "".join(_tool_html(tool, color) for tool in tools)
        if not tools:
            inner += '<p style="margin:0.2rem 0 0.35rem;font-size:0.8rem;color:#64748b">No tools.</p>'
    return (
        f'<details style="margin:0 0 0.5rem;border:1px solid {color};border-radius:12px;background:#ffffff">'
        '<summary style="cursor:pointer;padding:0.62rem 0.8rem;background:'
        f'{tint};border-radius:12px">'
        f'<span style="display:inline-block;background:{color};color:#ffffff;border-radius:999px;'
        'padding:0.12rem 0.48rem;font-size:0.68rem;font-weight:700;letter-spacing:0.04em">MCP</span>'
        f'<span style="font-weight:700;font-size:0.95rem;color:{color};margin-left:0.5rem">{name}</span>'
        f'<span style="font-weight:600;color:#475569;font-size:0.78rem;margin-left:0.45rem">{html.escape(count)}</span>'
        "</summary>"
        f'<div style="padding:0.55rem 0.8rem 0.35rem">{inner}</div>'
        "</details>"
    )


def _mcp_servers_html(servers, batches) -> str:
    """Colored MCP server rows. Open one row to read its tools.

    A ``<select>`` cannot swap panels here. Jupyter disables every ``<input>`` and
    strips ``<script>`` from untrusted HTML, so ``<details>`` is the control that
    still opens. See the JupyterLab sanitizer:
    https://github.com/jupyterlab/jupyterlab/blob/main/packages/apputils/src/sanitizer.ts
    """
    cards = []
    for index, (server, batch) in enumerate(zip(servers, batches)):
        color = _SERVER_COLORS[index % len(_SERVER_COLORS)]
        cards.append(_server_html(server, batch, color))
    body = "".join(cards) or '<p style="margin:0;color:#64748b">No registry servers are enabled.</p>'
    return (
        "<style>"
        ".mcp-registry summary::-webkit-details-marker{display:none}"
        ".mcp-registry summary{list-style:none}"
        ".mcp-registry details[open]>summary{border-radius:12px 12px 0 0}"
        "</style>"
        '<div class="mcp-registry" style="font-family:ui-sans-serif,system-ui,sans-serif;background:#ffffff;color:#1e293b;'
        'border-radius:12px;padding:0.9rem 1.1rem 1rem;margin:0.4rem 0 0.8rem;border:1px solid #e2e8f0">'
        '<div style="font-size:1.02rem;font-weight:700">Registry</div>'
        '<p style="margin:0.2rem 0 0.75rem;font-size:0.8rem;color:#64748b">'
        "Select an MCP server. Its tools open under that name. "
        "The server name is what you pass to <code>create_agent</code>.</p>"
        f"{body}</div>"
    )


def _print_mcp_index(servers, batches) -> None:
    print(f"{len(servers)} MCP servers")
    for server, batch in zip(servers, batches):
        name = server.get("name") or ""
        if isinstance(batch, Exception):
            print(f"- {name}: tools unavailable ({batch})")
            continue
        names = ", ".join(tool.name for tool in batch)
        print(f"- {name}: {names or 'no tools'}")


def _require_unique_tool_names(tools_by_server: dict[str, list]) -> None:
    """Raise when two servers publish the same tool name.

    LangChain keeps one tool per name, so the second copy is dropped with no error.
    """
    owners: dict[str, list[str]] = {}
    for server, tools in tools_by_server.items():
        for tool in tools:
            owners.setdefault(tool.name, []).append(server)
    clashes = {name: servers for name, servers in owners.items() if len(servers) > 1}
    if not clashes:
        return
    detail = "; ".join(f"{name} ({', '.join(servers)})" for name, servers in sorted(clashes.items()))
    raise ValueError(
        "Two MCP servers expose the same tool name, so one would be dropped: "
        f"{detail}. Rename one of them."
    )


async def pretty_print_mcp_servers() -> None:
    """Show EVE's registry MCP servers. In a notebook, select a server to read its tools.

    The server name is what you pass in the ``create_agent`` list (``\"effis\"``, ``\"serpapi\"``).
    """
    from .eve import list_eve_mcp_servers

    servers = [server for server in list_eve_mcp_servers() if server.get("enabled", True)]

    async def _tools(name: str):
        async with eve_mcp_client(name) as client:
            return await client.list_tools()

    batches = await asyncio.gather(*(_tools(server["name"]) for server in servers), return_exceptions=True)
    page = _mcp_servers_html(servers, batches)
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        _print_mcp_index(servers, batches)
        return
    if get_ipython() is None:
        _print_mcp_index(servers, batches)
        return
    display(HTML(page))


async def call_tool(client: Client, tool: str, **arguments):
    """Call one tool and return its result, decoded from JSON when possible."""
    async with client:
        result = await client.call_tool(tool, arguments)
    text = "".join(getattr(block, "text", "") for block in result.content)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _script_key(script: Path) -> str:
    """``some-server/server.py`` → ``some-server``; ``hello_server.py`` → ``hello_server``."""
    if script.stem == "server" and script.parent.name:
        return script.parent.name
    return script.stem


def _stdio_connection(script: Path) -> dict:
    """LangChain MCP adapter config: runs ``python server.py --transport stdio`` → ``mcp.run(transport=\"stdio\")``."""
    script = Path(script).resolve()
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(script), "--transport", "stdio"],
        "cwd": str(script.parent),
    }


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _http_connection(url: str, headers: dict[str, str] | None = None) -> dict:
    """Connect to an MCP server that is already listening (``mcp.run(transport=\"http\")``).

    Streamable HTTP is served at ``/mcp``. See
    https://gofastmcp.com/deployment/running-server
    """
    connection = {"transport": "http", "url": url}
    if headers:
        connection["headers"] = headers
    return connection


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


async def _serve_notebook_http(server: FastMCP, host: str, port: int, holder: dict[str, uvicorn.Server]) -> None:
    """Same HTTP app as ``FastMCP.run_http_async``, with the Uvicorn server kept so we can stop it.

    Streamable HTTP is served at ``/mcp``:
    https://gofastmcp.com/deployment/running-server
    """
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    async with server._lifespan_manager():
        app = server.http_app(transport="http", path="/mcp", stateless_http=True)
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            lifespan="on",
            ws="websockets-sansio",
        )
        uv_server = uvicorn.Server(config)
        holder["server"] = uv_server
        await uv_server.serve()


class _RunningNotebookServer:
    """One notebook FastMCP instance, listening until the last agent calls ``close``.

    Runs on a background thread so Uvicorn does not replace the notebook's SIGINT
    handler (``uvicorn.Server.capture_signals`` only installs handlers on the main
    thread). ``close`` sets ``Server.should_exit``, the flag Uvicorn uses for a stop
    signal. Same start-then-stop scope as ``fastmcp.utilities.tests.run_server_async``.
    """

    def __init__(self, server: FastMCP):
        self.server = server
        self.host = "127.0.0.1"
        self.port = _free_port(self.host)
        self.url = f"http://{self.host}:{self.port}/mcp"
        self.refs = 0
        self._holder: dict[str, uvicorn.Server] = {}
        self._errors: list[BaseException] = []
        self._thread: threading.Thread | None = None

    async def start(self) -> str:
        def _run() -> None:
            try:
                asyncio.run(_serve_notebook_http(self.server, self.host, self.port, self._holder))
            except Exception as exc:
                self._errors.append(exc)

        self._thread = threading.Thread(target=_run, name="notebook-mcp", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self._errors:
                raise RuntimeError(f"Notebook MCP server failed to start: {self._errors[0]}") from self._errors[0]
            if _port_is_open(self.port, self.host):
                await asyncio.sleep(0.1)
                return self.url
            if not self._thread.is_alive():
                raise RuntimeError("Notebook MCP server exited before it started listening")
            await asyncio.sleep(0.05)
        raise RuntimeError(f"Notebook MCP server did not start on {self.host}:{self.port}")

    async def stop(self) -> None:
        uv_server = self._holder.get("server")
        thread = self._thread
        if uv_server is not None:
            uv_server.should_exit = True
        if thread is not None:
            await asyncio.to_thread(thread.join, 5)
            if thread.is_alive() and uv_server is not None:
                uv_server.force_exit = True
                await asyncio.to_thread(thread.join, 2)


_live_notebook_servers: dict[int, _RunningNotebookServer] = {}


async def acquire_notebook_server(server: FastMCP) -> _RunningNotebookServer:
    """Start ``server`` once, and reuse it while an agent still has it open."""
    live = _live_notebook_servers.get(id(server))
    if live is not None and live._thread is not None and live._thread.is_alive():
        live.refs += 1
        return live
    started = _RunningNotebookServer(server)
    await started.start()
    started.refs = 1
    _live_notebook_servers[id(server)] = started
    return started


async def release_notebook_server(running: _RunningNotebookServer) -> None:
    """Drop one user of ``running``. The last one stops the process."""
    running.refs -= 1
    if running.refs > 0:
        return
    _live_notebook_servers.pop(id(running.server), None)
    await running.stop()


def _registry_connection(name: str) -> tuple[str, dict]:
    """A registry server hosted by EVE, at ``{EVE_API_BASE_URL}/mcp/{name}``."""
    if not settings.eve_configured:
        from .eve import EveUnavailable

        raise EveUnavailable("EVE is not configured: remote MCP servers are not available.")
    return name, _http_connection(
        f"{settings.eve_mcp_base_url}/{name}",
        headers={"Authorization": f"Bearer {settings.eve_api_key}"},
    )


def _mcp_server_connection(server: Any) -> tuple[str, dict]:
    """One LangChain connection for a URL, a ``server.py`` path, or an EVE registry name.

    A name such as ``\"effis\"`` is reached through EVE's MCP gateway. A FastMCP
    instance is started by :func:`agentic_eo.agent.create_agent` and passed here as
    its ``http://…/mcp`` URL.
    """
    if isinstance(server, (str, Path)):
        text = os.fspath(server)
        if _is_http_url(text):
            return "notebook", _http_connection(text)
        path = Path(text)
        if "/" in text or text.endswith(".py") or path.exists():
            return _script_key(path), _stdio_connection(path)
        return _registry_connection(text)

    raise TypeError(
        "Each server must be a registry name, a path to server.py, or an http(s) URL. "
        "Pass FastMCP instances to create_agent(); it starts them."
    )


async def mcp_tools_for_agent(servers: list | None = None):
    """LangChain tools from a list of MCP servers.

    Each entry is an ``http://…/mcp`` URL, a path to ``server.py``, or an EVE
    registry name such as ``"effis"``. FastMCP instances are started by
    :func:`agentic_eo.agent.create_agent`, which passes their URL here.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections: dict[str, dict] = {}
    for server in servers or []:
        key, connection = _mcp_server_connection(server)
        if key in connections:
            key = f"{key}-{len(connections)}"
        connections[key] = connection
    if not connections:
        return []
    client = MultiServerMCPClient(connections)
    names = list(connections)
    batches = await asyncio.gather(*(client.get_tools(server_name=name) for name in names))
    _require_unique_tool_names(dict(zip(names, batches)))
    return [tool for batch in batches for tool in batch]


async def ask(question: str, *,
              servers: list | None = None,
              prompt: str | None = None,
              verbose: bool = True):
    """Create an agent, ask one question, and close it.

    Prefer :func:`agentic_eo.agent.create_agent` when you want several ``chat`` turns
    and an explicit ``close``.
    """
    from .agent import create_agent

    agent = await create_agent(servers or [], prompt=prompt)
    try:
        return await agent.chat(question, echo=verbose)
    finally:
        await agent.close()
