# TigerBx Skills

This directory contains skills for AI coding assistants that support the
[Claude Code skills standard](https://code.claude.com/docs/en/skills.md).

Once installed, your AI assistant will automatically know when and how to use
TigerBx for brain MRI analysis tasks — brain extraction, segmentation,
registration, EPI distortion correction, and hippocampus embedding.

---

## Install (Claude Code)

Copy the `tigerbx` skill directory to your skills folder:

```bash
# Project-level (this project only)
cp -r skills/tigerbx .claude/skills/

# User-level (all your projects)
cp -r skills/tigerbx ~/.claude/skills/
```

Then reload Claude Code. The `/tigerbx` skill will be available, and Claude
will automatically use it whenever you ask about brain MRI processing tasks.

---

## Install (Codex CLI)

In Codex CLI interactive mode, run:

```
$skill-installer https://github.com/htylab/tigerbx/tree/main/skills/tigerbx
```

---

## Available skills

| Skill | Description |
|-------|-------------|
| `tigerbx` | Brain MRI analysis: extraction, segmentation, registration, EPI correction, hippocampus embedding, quantitative evaluation |
