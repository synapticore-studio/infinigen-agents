# GitHub Configuration for Infinigen

This directory contains GitHub-specific configuration files and documentation for the Infinigen project.

## Files Overview

### Copilot Configuration

- **[copilot-instructions.md](copilot-instructions.md)** - Repository-level instructions for GitHub Copilot
  - Defines tech stack, architecture, and coding standards
  - Provides guidance on using MCP servers (Context7, Serena)
  - Contains development workflow and common patterns
  - **This file is automatically read by GitHub Copilot when working in this repository**

- **[MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md)** - Guide for setting up Model Context Protocol servers
  - Step-by-step setup for Context7 (real-time documentation)
  - Step-by-step setup for Serena (advanced codebase analysis)
  - Troubleshooting and platform-specific instructions
  - Configuration examples for VS Code and other editors

### Copilot Prompts

- **[prompts/onboarding-plan.prompt.md](prompts/onboarding-plan.prompt.md)** - Onboarding plan generator
  - Invoke with `/onboarding-plan` in GitHub Copilot Chat
  - Generates personalized onboarding plans for new contributors
  - Adapts to contributor's background and interests

### Workflows

- **[workflows/](workflows/)** - GitHub Actions CI/CD workflows
  - `checks.yml` - Linting, testing, and code quality checks
  - `release.yml` - Release automation

### Issue Templates

- **[ISSUE_TEMPLATE/](ISSUE_TEMPLATE/)** - Templates for GitHub issues
  - `ask-for-help.md` - Template for getting help
  - `bug-report.md` - Template for reporting bugs
  - `suggestion.md` - Template for feature requests
  - `other.md` - General issue template

## Using GitHub Copilot with Infinigen

### 1. Repository-Level Instructions

When you open this repository in an editor with GitHub Copilot:

1. Copilot automatically reads `.github/copilot-instructions.md`
2. This provides context about the project structure, coding standards, and best practices
3. Copilot uses this information to give more accurate suggestions

### 2. MCP Servers (Optional but Recommended)

Model Context Protocol servers extend Copilot's capabilities:

- **Context7**: Provides up-to-date documentation for Blender, NumPy, OpenCV, etc.
- **Serena**: Enables advanced codebase analysis and structural queries

**Setup**: Follow [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md) for detailed instructions.

**Note**: MCP servers are configured per-editor (in VS Code's `mcp.json`), not per-repository.

### 3. Copilot Prompt Files

Use prompt files for common workflows:

```
# In GitHub Copilot Chat
/onboarding-plan

# Or with workspace context
@workspace /onboarding-plan
```

This invokes the onboarding plan generator with full repository context.

## For Contributors

If you're contributing to Infinigen, please also read:

- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- **[../README.md](../README.md)** - Project overview
- **[../docs/](../docs/)** - Detailed documentation

## Adding New Copilot Features

### Adding a New Prompt File

To add a new Copilot prompt file:

1. Create a new file in `prompts/` with the `.prompt.md` extension
2. Add YAML front matter with `mode` and `description`:
   ```yaml
   ---
   mode: agent
   description: Brief description of what this prompt does
   ---
   ```
3. Write the prompt instructions in Markdown
4. Users can invoke it with `/<filename-prefix>` in Copilot Chat

### Updating Copilot Instructions

To update how Copilot behaves in this repository:

1. Edit `copilot-instructions.md`
2. Add or update sections as needed
3. Copilot will automatically use the updated instructions
4. No need to reload or restart (Copilot reads the file as needed)

### Testing Copilot Prompts

To test a new prompt:

1. Open GitHub Copilot Chat in your editor
2. Type `/<your-prompt-name>` (based on filename)
3. Verify the prompt appears and works as expected
4. Iterate on the prompt content based on results

## Resources

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [Customizing GitHub Copilot](https://docs.github.com/en/copilot/customizing-copilot)
- [Prompt Files Guide](https://docs.github.com/en/copilot/tutorials/customization-library/prompt-files)
- [Model Context Protocol](https://docs.github.com/en/copilot/customizing-copilot/using-model-context-protocol)

## Questions?

For questions about GitHub Copilot setup or usage:

1. Check the [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md) for MCP-related issues
2. Review [copilot-instructions.md](copilot-instructions.md) for repository guidelines
3. Open an issue using the "ask-for-help" template
