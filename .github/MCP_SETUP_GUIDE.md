# MCP Setup Guide

MCP-Server erweitern GitHub Copilot um:
- **Context7**: aktuelle Dokumentation für Bibliotheken (bpy, NumPy, OpenCV etc.)
- **Serena**: semantische Codebase-Analyse (Call-Graphs, Abhängigkeiten, Struktursuche)

---

## VS Code (Workspace)

Die Datei `.vscode/mcp.json` im Repo-Root ist bereits vorkonfiguriert — sie wird automatisch geladen wenn du dieses Repo in VS Code öffnest. Nach dem Öffnen:

1. VS Code fragt, ob die MCP-Server gestartet werden sollen → **Allow**
2. Im Copilot Chat auf das **Tools-Icon** klicken und sicherstellen, dass `context7` und `serena` aktiviert sind

**Voraussetzungen:**
- Node.js 18+ (für `npx`)
- `uv` / `uvx` (für Serena) — `pip install uv` oder `winget install astral-sh.uv`

---

## Copilot Coding Agent (GitHub Settings)

Der autonome Coding Agent (der selbstständig PRs erstellt) benötigt eine separate Konfiguration in den Repository-Settings:

**Settings → Copilot → Coding agent → MCP configuration:**

```json
{
  "mcpServers": {
    "context7": {
      "type": "local",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
      "tools": ["*"]
    },
    "serena": {
      "type": "local",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server",
        "--context",
        "agent"
      ],
      "tools": ["*"]
    }
  }
}
```

> **Hinweis**: Für den Coding Agent `--context agent` verwenden. Für VS Code `--context ide-assistant` (bereits in `.vscode/mcp.json` gesetzt).

---

## Verifikation

Copilot Chat testen:

```
"What's the current bpy API for creating geometry nodes in Blender 4.5?"
```

Context7 sollte automatisch aktuelle Docs abrufen. Serena-Test:

```
"Find all classes that inherit from AssetFactory in this codebase"
```

---

## Ressourcen

- [Context7 GitHub](https://github.com/upstash/context7)
- [Serena GitHub](https://github.com/oraios/serena)
- [GitHub Copilot MCP Docs](https://docs.github.com/copilot/how-tos/agents/copilot-coding-agent/extending-copilot-coding-agent-with-mcp)
