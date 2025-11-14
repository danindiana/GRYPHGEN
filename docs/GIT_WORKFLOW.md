# GRYPHGEN Git Workflow Documentation

This document provides comprehensive mermaid diagrams explaining the git workflow, branching strategy, and collaboration model for the GRYPHGEN project.

## Table of Contents
1. [Branch Strategy](#branch-strategy)
2. [Claude Code Development Workflow](#claude-code-development-workflow)
3. [Feature Development Lifecycle](#feature-development-lifecycle)
4. [Commit and Push Workflow](#commit-and-push-workflow)
5. [Pull Request Process](#pull-request-process)

---

## Branch Strategy

The GRYPHGEN project uses a feature branch workflow where all development happens on dedicated branches prefixed with `claude/`.

```mermaid
gitGraph
    commit id: "Initial commit"
    commit id: "Add SYMORQ component"
    commit id: "Add SYMORG component"
    branch claude/feature-mcp-server
    checkout claude/feature-mcp-server
    commit id: "Setup MCP server structure"
    commit id: "Add API endpoints"
    commit id: "Add documentation"
    checkout main
    merge claude/feature-mcp-server
    branch claude/add-git-diagrams
    checkout claude/add-git-diagrams
    commit id: "Create git workflow docs"
    commit id: "Add mermaid diagrams"
    checkout main
    merge claude/add-git-diagrams
    commit id: "Release v1.0"
```

### Branch Naming Convention

```mermaid
graph LR
    A[Branch Name] --> B[claude/]
    B --> C[descriptive-name]
    C --> D[-sessionID]

    style A fill:#e1f5ff
    style B fill:#ffe1e1
    style C fill:#e1ffe1
    style D fill:#fff5e1
```

**Format:** `claude/{descriptive-name}-{sessionID}`

**Examples:**
- `claude/add-git-mermaid-diagrams-01A7FntuzYu1pDpWAiqwbqSv`
- `claude/feature-rag-system-AbC123XyZ456`
- `claude/fix-zeromq-connection-DeF789UvW012`

---

## Claude Code Development Workflow

This diagram shows the complete workflow when using Claude Code to develop features.

```mermaid
sequenceDiagram
    participant User
    participant Claude as Claude Code
    participant Git as Git Repository
    participant Remote as GitHub Remote

    User->>Claude: Request feature/fix
    Claude->>Claude: Analyze requirements
    Claude->>Git: Check current branch

    alt Branch doesn't exist
        Claude->>Git: Create feature branch
        Note over Claude,Git: git checkout -b claude/feature-name-sessionID
    else Branch exists
        Claude->>Git: Checkout existing branch
    end

    loop Development Cycle
        Claude->>Claude: Make code changes
        Claude->>Git: Stage changes
        Claude->>Git: Commit with message
        Note over Claude,Git: git commit -m "Descriptive message"
    end

    Claude->>Git: Push to remote
    Note over Claude,Git: git push -u origin branch-name

    alt Push fails (network)
        Claude->>Claude: Wait 2s
        Claude->>Git: Retry push
        alt Still fails
            Claude->>Claude: Wait 4s (exponential backoff)
            Claude->>Git: Retry push (up to 4 attempts)
        end
    end

    Claude->>User: Confirm push successful
    Claude->>User: Provide branch name for PR
```

---

## Feature Development Lifecycle

Complete lifecycle from feature request to merge.

```mermaid
stateDiagram-v2
    [*] --> FeatureRequest
    FeatureRequest --> Analysis
    Analysis --> Planning
    Planning --> BranchCreation

    BranchCreation --> Development

    state Development {
        [*] --> CodeWriting
        CodeWriting --> Testing
        Testing --> CodeReview
        CodeReview --> Fixes: Issues found
        Fixes --> Testing
        Testing --> Commit: Tests pass
        Commit --> [*]
    }

    Development --> Staging
    Staging --> Commit
    Commit --> Push

    Push --> NetworkRetry: Failure
    NetworkRetry --> Push: Retry (max 4x)
    Push --> PullRequest: Success

    PullRequest --> CodeReview2: Create PR
    CodeReview2 --> Approved: LGTM
    CodeReview2 --> Development: Changes requested

    Approved --> Merge
    Merge --> [*]
```

---

## Commit and Push Workflow

Detailed workflow for committing and pushing changes with retry logic.

```mermaid
flowchart TD
    A[Start: Changes Made] --> B{Stage Changes}
    B --> C[git add .]
    C --> D[Create Commit Message]
    D --> E[Analyze changes]
    E --> F[Write descriptive commit]
    F --> G[git commit -m message]

    G --> H{Push to Remote}
    H --> I[git push -u origin branch]

    I --> J{Push Successful?}
    J -->|Yes| K[Complete]
    J -->|No| L{Error Type?}

    L -->|Network Error| M{Retry Count < 4?}
    L -->|Auth/403 Error| N[Check Branch Name]
    L -->|Other Error| O[Report to User]

    M -->|Yes| P[Wait - Exponential Backoff]
    M -->|No| Q[Max Retries: Report Failure]

    P --> R{Attempt #}
    R -->|1| S[Wait 2s]
    R -->|2| T[Wait 4s]
    R -->|3| U[Wait 8s]
    R -->|4| V[Wait 16s]

    S --> I
    T --> I
    U --> I
    V --> I

    N --> W{Correct Format?}
    W -->|No| X[Fix: Must start with claude/ and end with sessionID]
    W -->|Yes| O
    X --> H

    K --> Y[Notify User]
    O --> Y
    Q --> Y
    Y --> Z[End]

    style K fill:#90EE90
    style Q fill:#FFB6C1
    style O fill:#FFB6C1
    style N fill:#FFD700
```

---

## Pull Request Process

Workflow for creating and managing pull requests.

```mermaid
sequenceDiagram
    participant Dev as Developer/Claude
    participant Branch as Feature Branch
    participant Remote as GitHub Remote
    participant CI as CI/CD Pipeline
    participant Rev as Code Reviewer
    participant Main as Main Branch

    Dev->>Branch: Complete feature development
    Dev->>Remote: Push feature branch
    Remote->>CI: Trigger automated tests

    CI->>CI: Run test suite
    CI->>CI: Run linters
    CI->>CI: Security scan
    CI->>CI: Build verification

    alt Tests Pass
        CI->>Remote: ✅ All checks passed
        Dev->>Remote: Create Pull Request
        Remote->>Rev: Notify reviewer

        Rev->>Remote: Review code

        alt Approved
            Rev->>Remote: Approve PR
            Rev->>Main: Merge to main
            Main->>Remote: Update main branch
            Remote->>Branch: Delete feature branch
        else Changes Requested
            Rev->>Dev: Request changes
            Dev->>Branch: Make updates
            Dev->>Remote: Push updates
            Remote->>CI: Re-run tests
        end
    else Tests Fail
        CI->>Remote: ❌ Tests failed
        CI->>Dev: Notify of failures
        Dev->>Branch: Fix issues
        Dev->>Remote: Push fixes
        Remote->>CI: Re-trigger tests
    end
```

---

## Repository Structure Workflow

How different components interact in the repository structure.

```mermaid
graph TB
    ROOT[GRYPHGEN Repository Root]

    ROOT --> DOCS[📁 docs/]
    ROOT --> MCP[📁 MCP_SERVER/]
    ROOT --> MERMAID[📁 mermaid/]
    ROOT --> COMPONENTS[📁 Component Dirs/]
    ROOT --> DATED[📁 Dated Releases/]
    ROOT --> README[📄 README.md]
    ROOT --> LICENSE[📄 LICENSE]

    DOCS --> WORKFLOW[📄 GIT_WORKFLOW.md]
    DOCS --> STRUCTURE[📄 STRUCTURE.md]

    MCP --> MCP_DOCS[📄 Documentation]
    MCP --> MCP_CODE[💻 Server Code]

    MERMAID --> DIAGRAMS[📄 Diagram Files]
    MERMAID --> MERMAID_README[📄 readme.md]

    COMPONENTS --> SYMORQ[📁 SYMORQ/]
    COMPONENTS --> SYMORG[📁 SYMORG/]
    COMPONENTS --> SYMAUG[📁 SYMAUG/]
    COMPONENTS --> UTILS[📁 Utilities/]

    DATED --> JAN[📁 jan13, jan14/]
    DATED --> MARCH[📁 march13-25/]
    DATED --> AUG[📁 Aug_20_2025/]

    README --> MAIN_DOCS[Main Documentation]
    README --> ARCH_DIAGRAMS[Architecture Diagrams]

    style ROOT fill:#e1f5ff
    style DOCS fill:#e8f5e9
    style MCP fill:#fff3e0
    style MERMAID fill:#f3e5f5
    style COMPONENTS fill:#e0f2f1
    style README fill:#fce4ec
```

---

## Collaboration Model

Multi-developer collaboration workflow with conflict resolution.

```mermaid
sequenceDiagram
    participant Dev1 as Developer 1
    participant Dev2 as Developer 2
    participant Remote as Remote Repository
    participant Main as Main Branch

    Main->>Dev1: Clone/Pull latest
    Main->>Dev2: Clone/Pull latest

    Dev1->>Dev1: Create feature branch A
    Dev2->>Dev2: Create feature branch B

    par Parallel Development
        Dev1->>Dev1: Develop feature A
        Dev1->>Remote: Push branch A
    and
        Dev2->>Dev2: Develop feature B
        Dev2->>Remote: Push branch B
    end

    Dev1->>Remote: Create PR for feature A
    Remote->>Main: Merge feature A

    Dev2->>Remote: Create PR for feature B
    Remote->>Dev2: Conflict detected

    Dev2->>Remote: Fetch latest main
    Dev2->>Dev2: Rebase on main
    Dev2->>Dev2: Resolve conflicts
    Dev2->>Remote: Push updated branch B
    Remote->>Main: Merge feature B
```

---

## Best Practices

### Branch Management
1. **Always** work on feature branches prefixed with `claude/`
2. **Never** commit directly to main branch
3. **Include** session ID in branch name for traceability
4. **Delete** branches after successful merge

### Commit Messages
1. **Use** descriptive, imperative mood messages
2. **Reference** issue numbers when applicable
3. **Group** related changes in single commits
4. **Avoid** generic messages like "fix" or "update"

### Push Strategy
1. **Push** regularly to backup work
2. **Use** `-u` flag for first push to set upstream
3. **Implement** retry logic with exponential backoff
4. **Verify** branch name format before pushing

### Code Review
1. **Request** review before merging
2. **Test** thoroughly before creating PR
3. **Document** complex changes
4. **Respond** promptly to review feedback

---

## Troubleshooting

### Common Issues and Solutions

```mermaid
flowchart TD
    A[Git Issue] --> B{Issue Type?}

    B -->|Push Failed 403| C[Check Branch Name Format]
    C --> C1{Starts with claude/?}
    C1 -->|No| C2[Rename branch]
    C1 -->|Yes| C3{Ends with sessionID?}
    C3 -->|No| C4[Add sessionID suffix]
    C3 -->|Yes| C5[Check network]

    B -->|Merge Conflict| D[Fetch latest main]
    D --> D1[Rebase on main]
    D1 --> D2[Resolve conflicts]
    D2 --> D3[Continue rebase]
    D3 --> D4[Force push to branch]

    B -->|Network Error| E[Check connectivity]
    E --> E1[Retry with backoff]
    E1 --> E2{Success?}
    E2 -->|Yes| E3[Continue]
    E2 -->|No| E4[Report error]

    B -->|Detached HEAD| F[Checkout branch]
    F --> F1[git checkout branch-name]

    style C2 fill:#90EE90
    style C4 fill:#90EE90
    style D4 fill:#90EE90
    style E3 fill:#90EE90
    style C5 fill:#FFD700
    style E4 fill:#FFB6C1
```

---

## Integration with GRYPHGEN System

How git workflow integrates with the overall GRYPHGEN architecture.

```mermaid
graph TB
    subgraph "Development Layer"
        GIT[Git Repository]
        CLAUDE[Claude Code Agent]
        DEV[Developer]
    end

    subgraph "GRYPHGEN Core"
        SYMORQ[SYMORQ<br/>Orchestration]
        SYMORG[SYMORG<br/>RAG System]
        SYMAUG[SYMAUG<br/>Microservices]
    end

    subgraph "Deployment Layer"
        CI[CI/CD Pipeline]
        TARGET[Target Server]
        DOCKER[Docker Containers]
    end

    DEV -->|Requests| CLAUDE
    CLAUDE -->|Commits| GIT
    GIT -->|Triggers| CI

    CI -->|Deploys| SYMORQ
    CI -->|Deploys| SYMORG
    CI -->|Deploys| SYMAUG

    SYMORQ -->|Orchestrates| TARGET
    SYMORG -->|Retrieves| TARGET
    SYMAUG -->|Executes| DOCKER

    DOCKER -->|Runs on| TARGET

    style GIT fill:#e1f5ff
    style CLAUDE fill:#ffe1e1
    style SYMORQ fill:#e1ffe1
    style SYMORG fill:#fff5e1
    style SYMAUG fill:#f5e1ff
```

---

## Conclusion

This git workflow documentation provides a comprehensive guide to:
- Branch management and naming conventions
- Development workflow with Claude Code
- Commit and push strategies with retry logic
- Pull request process and code review
- Collaboration and conflict resolution
- Integration with GRYPHGEN architecture

Following these workflows ensures consistent, maintainable, and collaborative development practices for the GRYPHGEN project.
