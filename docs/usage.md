[← Back to README](../README.md#documentation)

# Usage Guide

## Command Line

```bash
# Review GitHub Pull Request
codespy review https://github.com/owner/repo/pull/123

# Review GitLab Merge Request
codespy review https://gitlab.com/group/project/-/merge_requests/123

# GitLab with nested groups
codespy review https://gitlab.com/group/subgroup/project/-/merge_requests/123

# Self-hosted GitLab
codespy review https://gitlab.mycompany.com/team/project/-/merge_requests/123

# Output as JSON
codespy review https://github.com/owner/repo/pull/123 --output json

# Use a specific model
codespy review https://github.com/owner/repo/pull/123 --model anthropic/claude-opus-4-6

# Use a custom config file
codespy review https://github.com/owner/repo/pull/123 --config path/to/config.yaml
codespy review https://github.com/owner/repo/pull/123 -f staging.yaml

# Disable stdout output (useful with --git-comment)
codespy review https://github.com/owner/repo/pull/123 --no-stdout

# Post review as GitHub/GitLab comment
codespy review https://github.com/owner/repo/pull/123 --git-comment

# Combine: only post to Git platform, no stdout
codespy review https://github.com/owner/repo/pull/123 --no-stdout --git-comment

# Show current configuration
codespy config

# Show configuration from a specific file
codespy config --config path/to/config.yaml

# Show version
codespy --version

# Review local git changes (no GitHub/GitLab needed)
codespy review-local                    # Review current dir vs main
codespy review-local /path/to/repo      # Review specific repo
codespy review-local --base develop     # Compare against develop
codespy review-local --base origin/main # Compare against origin/main
codespy review-local --base HEAD~5      # Compare against 5 commits back

# Review uncommitted changes (staged + unstaged)
codespy review-uncommitted              # Review current dir
codespy review-uncommitted /path/to/repo
codespy review-uncommitted --output json
```

## IDE Integration (MCP Server)

CodeSpy can run as an MCP (Model Context Protocol) server for integration with AI coding assistants like Cline, enabling code reviews directly from your editor without leaving your workflow.

```bash
# Start the MCP server
codespy serve

# Use a custom config file
codespy serve --config path/to/config.yaml
```

**Configure your IDE** (example for Cline in VS Code):

Add to `cline_mcp_settings.json`:
```json
{
  "mcpServers": {
    "codespy-reviewer": {
      "command": "codespy",
      "args": ["serve"],
      "env": {
        "DEFAULT_MODEL": "anthropic/claude-opus-4-6",
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

Or for AWS Bedrock:
```json
{
  "mcpServers": {
    "codespy-reviewer": {
      "command": "codespy",
      "args": ["serve"],
      "env": {
        "DEFAULT_MODEL": "bedrock/us.anthropic.claude-opus-4-6-v1",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "your-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret-key"
      }
    }
  }
}
```

**Available MCP Tools:**
- `review_local_changes(repo_path, base_ref)` — Review branch changes vs base (e.g., vs `main`)
- `review_uncommitted(repo_path)` — Review staged + unstaged working tree changes
- `review_pr(mr_url)` — Review a GitHub PR or GitLab MR by URL

Then ask your AI assistant: *"Review my local changes"* or *"Review uncommitted work in /path/to/repo"*

## Docker

```bash
# With docker run (using GHCR image)
docker run --rm \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e DEFAULT_MODEL=anthropic/claude-opus-4-6 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/khezen/codespy:latest review https://github.com/owner/repo/pull/123

# Or use a specific version
docker run --rm \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e DEFAULT_MODEL=anthropic/claude-opus-4-6 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/khezen/codespy:0.2.1 review https://github.com/owner/repo/pull/123
```

## GitHub Action

Add CodeSpy to your repository for automatic PR reviews:

**Trigger on `/codespy review` comment:**

```yaml
# .github/workflows/codespy-review.yml
name: CodeSpy Code Review

on:
  issue_comment:
    types: [created]

jobs:
  review:
    # Only run on PR comments containing '/codespy review'
    if: |
      github.event.issue.pull_request &&
      contains(github.event.comment.body, '/codespy review')
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - name: Run CodeSpy Review
        uses: khezen/codespy@v1
        with:
          model: 'anthropic/claude-opus-4-6'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Trigger automatically on every PR:**

```yaml
# .github/workflows/codespy-review.yml
name: CodeSpy Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - name: Run CodeSpy Review
        uses: khezen/codespy@v1
        with:
          model: 'anthropic/claude-opus-4-6'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

See [`../.github/workflows/codespy-review.yml.example`](../.github/workflows/codespy-review.yml.example) for more examples.

## Output

### Markdown (default)

```markdown
# Code Review: Add user authentication

**PR:** [owner/repo#123](https://github.com/owner/repo/pull/123)
**Reviewed at:** 2024-01-15 10:30 UTC
**Model:** anthropic/claude-opus-4-6

## Summary

This PR implements user authentication with JWT tokens...

## Statistics

- **Total Issues:** 3
- **Critical:** 1
- **Security:** 1
- **Bugs:** 1
- **Documentation:** 1

## Issues

### 🔴 Critical (1)

#### SQL Injection Vulnerability

**Location:** `src/auth/login.py:45`
**Category:** security

The user input is directly interpolated into the SQL query...

**Code:**
query = f"SELECT * FROM users WHERE username = '{username}'"

**Suggestion:**
Use parameterized queries instead...

**Reference:** [CWE-89](https://cwe.mitre.org/data/definitions/89.html)
```

### GitHub/GitLab Review Comments

CodeSpy can post reviews directly to GitHub PRs or GitLab MRs as native review comments with inline annotations.

**Enable via CLI:**
```bash
# GitHub
codespy review https://github.com/owner/repo/pull/123 --git-comment

# GitLab
codespy review https://gitlab.com/group/project/-/merge_requests/123 --git-comment

# Combine: only post to platform, no stdout
codespy review https://github.com/owner/repo/pull/123 --no-stdout --git-comment
```

**Enable via configuration:**
```bash
# Environment variable
export OUTPUT_GIT=true

# Or in codespy.yaml
output_git: true
```

**Features:**

- 🎯 **Inline Comments** - Issues are posted as review comments on the exact lines where they occur
- 📏 **Multi-line Support** - Issues spanning multiple lines are annotated with start/end line ranges
- 🔴🟠🟡🔵 **Severity Indicators** - Visual emoji markers for Critical, High, Medium, Low severity
- 📦 **Collapsible Sections** - Organized review body with expandable details:
  - 📋 Summary of changes
  - 🎯 Quality Assessment
  - 📊 Statistics table
  - 💰 Cost breakdown per signature
  - 💡 Recommendation
- 🔗 **CWE References** - Security issues link directly to MITRE CWE database

---

See [Configuration](configuration.md) for all available settings.

---

[← Back to README](../README.md#documentation)
