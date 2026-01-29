# Model Context Protocol (MCP) Setup Guide for Infinigen

This guide explains how to configure Model Context Protocol (MCP) servers for use with GitHub Copilot in the Infinigen repository.

## Overview

MCP servers extend GitHub Copilot's capabilities by providing access to:
- **Context7**: Real-time documentation for frameworks and libraries
- **Serena**: Advanced codebase analysis and repository-specific tools
- **GitHub MCP**: GitHub-specific operations (already available in Copilot)

## Table of Contents

1. [Context7 MCP Server Setup](#context7-mcp-server-setup)
2. [Serena MCP Server Setup](#serena-mcp-server-setup)
3. [Using MCP with Copilot](#using-mcp-with-copilot)
4. [Troubleshooting](#troubleshooting)

---

## Context7 MCP Server Setup

Context7 provides up-to-date documentation for frameworks and libraries, which is especially useful for:
- Blender Python API (bpy) - frequently updated
- NumPy, SciPy, and scientific computing libraries
- OpenCV and computer vision libraries
- Python standard library updates

### Editor Configuration

MCP servers are configured **per editor/user**, not per repository. The configuration lives in your editor's settings.

#### Visual Studio Code Setup

**Step 1: Locate your MCP configuration file**

VS Code stores MCP configuration in `mcp.json` (not in `settings.json`):

- **Windows**: `C:\Users\<USERNAME>\AppData\Roaming\Code\User\mcp.json`
- **macOS**: `~/Library/Application Support/Code/User/mcp.json`
- **Linux**: `~/.config/Code/User/mcp.json`

**Step 2: Create or edit `mcp.json`**

If the file doesn't exist, create it. Add the Context7 server configuration:

```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

**Step 3: Add API key (optional but recommended)**

If you have a Context7 API key for enhanced features:

```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@upstash/context7-mcp@latest",
        "--api-key",
        "YOUR_CONTEXT7_API_KEY"
      ]
    }
  }
}
```

**Step 4: Reload VS Code**

After editing `mcp.json`:
1. Save the file
2. Reload VS Code: `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
3. Type "Reload Window" and press Enter

**Step 5: Enable Context7 in Copilot Chat**

1. Open GitHub Copilot Chat panel
2. Click the **Tools** icon (gear/settings icon in chat toolbar)
3. Ensure **Context7** is checked/enabled
4. The Context7 tools should now be available

#### JetBrains IDEs (IntelliJ, PyCharm, etc.)

MCP configuration for JetBrains IDEs follows a similar pattern. Consult the JetBrains documentation for the exact configuration file location and format.

### Verifying Context7 Setup

Test that Context7 is working:

1. Open Copilot Chat
2. Ask: "What's the latest way to create a mesh in Blender using bpy?"
3. Copilot should automatically invoke Context7 to get current documentation
4. The response should reference recent Blender API documentation

### Context7 Usage in Infinigen

Once configured, Context7 is automatically invoked by Copilot when:

- You ask about library APIs (Blender, NumPy, OpenCV, etc.)
- You request code examples using external frameworks
- You debug issues that might relate to API changes
- You work with libraries that have frequent updates

**Example prompts that benefit from Context7:**

- "How do I use bpy to create a procedural material with shader nodes?"
- "What's the current NumPy syntax for array broadcasting?"
- "Show me how to use OpenCV's latest feature detection methods"
- "How has the Blender Python API changed for modifiers?"

---

## Serena MCP Server Setup

Serena MCP provides advanced codebase analysis tools for structural queries and repo-specific operations.

### Installation Methods

Serena can be installed via:
- npm package (recommended)
- Binary installation
- From source

#### Option 1: npm Package (Recommended)

**Step 1: Edit `mcp.json`**

Add Serena to your MCP configuration:

```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    },
    "serena": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@serena/mcp-server"]
    }
  }
}
```

**Note**: Replace `@serena/mcp-server` with the actual npm package name if different.

**Step 2: Reload editor and enable**

1. Reload VS Code (Ctrl+Shift+P → "Reload Window")
2. Open Copilot Chat
3. Click **Tools** icon
4. Enable **Serena** in the tools list

#### Option 2: Binary Installation

If Serena is installed as a standalone binary:

```json
{
  "servers": {
    "serena": {
      "type": "stdio",
      "command": "/path/to/serena-mcp",
      "args": []
    }
  }
}
```

#### Option 3: With Configuration File

If Serena requires a configuration file or token:

```json
{
  "servers": {
    "serena": {
      "type": "stdio",
      "command": "serena-mcp",
      "args": ["--config", "/path/to/serena-config.json"],
      "env": {
        "SERENA_TOKEN": "your-token-here"
      }
    }
  }
}
```

### Serena Usage in Infinigen

Serena is useful for:

- **Code Search**: "Find all implementations of the Factory pattern"
- **Call Graph Analysis**: "Show all functions that call create_asset()"
- **Dependency Analysis**: "What modules depend on infinigen.core.nodes?"
- **Structural Queries**: "List all classes that inherit from AssetFactory"
- **Project Analysis**: "Analyze the module structure of infinigen.assets"

**Example prompts for Serena:**

- "Find all places where Blender materials are created"
- "Show the dependency graph for the terrain module"
- "List all procedural generators that use noise functions"
- "Find all test files that test asset generation"

---

## Using MCP with Copilot

### How Copilot Chooses MCP Servers

Copilot automatically selects the appropriate MCP server based on your query:

1. **Repository Questions**: Searches files first, may use Serena for structural analysis
2. **Documentation Questions**: Uses Context7 for current library/framework docs
3. **GitHub Operations**: Uses built-in GitHub MCP for issues, PRs, workflows

### Best Practices

**1. Be Specific About Libraries**

When asking about external libraries, mention names explicitly:

- ✅ "How do I use Blender's bpy.ops to add a mesh?"
- ❌ "How do I add a mesh?" (less clear which system)

**2. Specify Versions When Known**

- ✅ "Using Blender 4.5, how do I..."
- ✅ "With NumPy 1.x, what's the best way to..."

**3. Request Current Documentation**

You can explicitly ask for current docs:

- "Get the latest Blender documentation for material nodes"
- "Check current OpenCV docs for feature detection"

**4. Leverage Serena for Structure**

For architectural questions:

- "Analyze how assets are organized in this codebase"
- "Find all similar implementations of texture generation"

### Example Workflow

Here's a typical workflow using MCP servers:

```
You: "I need to add a new procedural tree generator to Infinigen"

Copilot (using Serena):
"Let me analyze existing tree generators..."
[Shows existing patterns in infinigen/assets/trees/]

You: "What's the current Blender API for creating geometry nodes?"

Copilot (using Context7):
[Retrieves latest bpy.types.GeometryNode documentation]
"Here's the current approach for Blender 4.5..."

You: "Show me how to test this"

Copilot (searches repo):
[Finds examples in tests/assets/]
"Here's the testing pattern used in similar assets..."
```

---

## Troubleshooting

### Context7 Not Working

**Symptom**: Copilot doesn't fetch current documentation

**Solutions**:

1. **Check MCP configuration**:
   ```bash
   # On Linux/macOS
   cat ~/.config/Code/User/mcp.json
   
   # On Windows (PowerShell)
   Get-Content "$env:APPDATA\Code\User\mcp.json"
   ```

2. **Verify Node.js is installed**:
   ```bash
   node --version
   npx --version
   ```
   If not installed, download from [nodejs.org](https://nodejs.org/)

3. **Test npx command directly**:
   ```bash
   npx -y @upstash/context7-mcp@latest --help
   ```

4. **Check Copilot Chat tools**:
   - Open Copilot Chat
   - Click Tools icon
   - Verify Context7 is listed and enabled

5. **Reload VS Code**:
   - Ctrl+Shift+P → "Reload Window"

### Serena Not Available

**Symptom**: Serena tools don't appear or don't work

**Solutions**:

1. **Verify installation**:
   ```bash
   # If using npm
   npx -y @serena/mcp-server --version
   
   # If using binary
   /path/to/serena-mcp --version
   ```

2. **Check mcp.json syntax**:
   - Ensure JSON is valid (no trailing commas, proper quotes)
   - Use a JSON validator if needed

3. **Check command path**:
   - Use absolute paths for binaries
   - Ensure executable permissions (Linux/macOS):
     ```bash
     chmod +x /path/to/serena-mcp
     ```

4. **Review Copilot Output**:
   - Open VS Code Output panel
   - Select "GitHub Copilot" from dropdown
   - Look for MCP-related errors

### MCP Configuration Not Loading

**Symptom**: Changes to `mcp.json` don't take effect

**Solutions**:

1. **Verify file location**:
   - Ensure editing the correct `mcp.json` (User-level, not workspace)
   - Check for typos in path

2. **Check JSON syntax**:
   - Use VS Code to edit `mcp.json` (shows syntax errors)
   - Validate with `jsonlint` or online validator

3. **Reload completely**:
   - Save `mcp.json`
   - Close VS Code completely
   - Reopen VS Code
   - Wait for Copilot to initialize

4. **Check VS Code version**:
   - Ensure you have a recent VS Code version
   - Update if necessary

### Permission Errors

**Symptom**: "Permission denied" or "Cannot execute" errors

**Solutions**:

1. **Linux/macOS - Check executable bit**:
   ```bash
   chmod +x /path/to/serena-mcp
   ```

2. **Windows - Run as administrator** (if needed)

3. **Check file ownership**:
   ```bash
   ls -l ~/.config/Code/User/mcp.json
   ```

### Network Issues

**Symptom**: "Cannot download" or "Registry error" when using npx

**Solutions**:

1. **Check internet connection**

2. **Configure npm registry** (if behind proxy):
   ```bash
   npm config set proxy http://proxy.company.com:8080
   npm config set https-proxy http://proxy.company.com:8080
   ```

3. **Try with explicit version**:
   ```bash
   npx @upstash/context7-mcp@1.0.0
   ```

---

## Advanced Configuration

### Multiple MCP Servers

You can configure multiple MCP servers simultaneously:

```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    },
    "serena": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@serena/mcp-server"]
    },
    "custom-tools": {
      "type": "stdio",
      "command": "/path/to/custom-mcp",
      "args": ["--config", "config.json"]
    }
  }
}
```

### Environment Variables

Pass environment variables to MCP servers:

```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
      "env": {
        "CONTEXT7_API_KEY": "${env:CONTEXT7_API_KEY}",
        "DEBUG": "true"
      }
    }
  }
}
```

### Logging and Debugging

Enable debug logging for MCP servers:

```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest", "--debug"]
    }
  }
}
```

Then check VS Code Output panel (GitHub Copilot channel) for logs.

---

## Additional Resources

- [GitHub Copilot MCP Documentation](https://docs.github.com/en/copilot/customizing-copilot/using-model-context-protocol)
- [Context7 MCP Documentation](https://github.com/upstash/context7-mcp)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

## Getting Help

If you encounter issues with MCP setup:

1. Check this troubleshooting guide first
2. Review VS Code Output panel for errors
3. Test MCP commands in terminal directly
4. Open an issue with:
   - Your OS and VS Code version
   - Your `mcp.json` configuration (redact sensitive info)
   - Error messages from Output panel
   - Steps to reproduce

---

## Summary

After completing this setup:

✅ Context7 provides real-time documentation for libraries and frameworks  
✅ Serena enables advanced codebase analysis  
✅ Copilot automatically chooses the right tool for your queries  
✅ You can explicitly request specific MCP capabilities

Remember:
- MCP configuration is **editor-level**, not repo-level
- Always reload your editor after changing `mcp.json`
- Enable MCP tools in Copilot Chat's Tools menu
- Be specific in your queries to get the best results

Happy coding with enhanced Copilot capabilities!
