# GRYPHGEN

**Grid Resource Prioritization in Heterogeneous Environments**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub stars](https://img.shields.io/github/stars/danindiana/GRYPHGEN.svg?style=social&label=Star)](https://github.com/danindiana/GRYPHGEN)
[![GitHub forks](https://img.shields.io/github/forks/danindiana/GRYPHGEN.svg?style=social&label=Fork)](https://github.com/danindiana/GRYPHGEN/fork)
[![GitHub issues](https://img.shields.io/github/issues/danindiana/GRYPHGEN.svg)](https://github.com/danindiana/GRYPHGEN/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/danindiana/GRYPHGEN.svg)](https://github.com/danindiana/GRYPHGEN/pulls)
[![Last commit](https://img.shields.io/github/last-commit/danindiana/GRYPHGEN.svg)](https://github.com/danindiana/GRYPHGEN/commits)
[![Repo size](https://img.shields.io/github/repo-size/danindiana/GRYPHGEN.svg)](https://github.com/danindiana/GRYPHGEN)
[![Contributors](https://img.shields.io/github/contributors/danindiana/GRYPHGEN.svg)](https://github.com/danindiana/GRYPHGEN/graphs/contributors)

GRYPHGEN (pronounced 'Griffin') is a framework that leverages large language models (LLMs) to automate software production at scale. The framework consists of three main components: SYMORQ, SYMORG, and SYMAUG.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [System Architecture](#system-architecture)
4. [Git Workflow and Development Process](#git-workflow-and-development-process)
5. [Implementation Details](#implementation-details)
6. [Documentation](#documentation)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

## Core Components

### 1. SYMORQ (Systems Orchestration for Resource Quality)

An LLM-based orchestration service that uses the ZeroMQ message passing API to manage and coordinate resources in the grid. It ensures that resources are utilized efficiently and effectively to meet the demands of the software production process.

### 2. SYMORG (Systems Orchestration Retrieval Generator)

An LLM-based automated RAG (Retrieval Augmented Generation) constructor. RAG is a technique used to enhance the accuracy and reliability of generative AI models by incorporating facts fetched from external sources. SYMORG automates the retrieval and incorporation of relevant information into the software production process.

### 3. SYMAUG (Smart Yielding Microservices for Agile and Ultra-Portable Grids)

A dockerized or virtual machine implementation of the CCDE-SIOS ensemble. It provides a lightweight and portable solution for GRYPHGEN deployment across various platforms and hardware architectures. This ensures that GRYPHGEN can be easily integrated into different environments and used to produce software at scale.

---

## System Architecture

**Motivation:** GRYPHGEN creates a self-deploying LLM cooperative programming environment capable of producing any type of software at any level of complexity and scale. To illustrate the workflow of GRYPHGEN, a series of interconnected sequence diagrams are used to describe the stages of operation.

### Basic Workflow Sequence

GRYPHGEN is designed to download/install/run local language models as code generators, analyzers, task monitors, and workflow optimizers. Local models are run using llama.cpp (https://github.com/ggerganov/llama.cpp) Inference of Meta's LLaMA model (and others) in pure C/C++. GRYPHGEN uses llama.cpp web server which is a lightweight OpenAI API (https://github.com/openai/openai-openapi) compatible HTTP server that can be used to serve local models and easily connect them to existing clients. GRYPHGEN integrates with Jan.ai and lm-studio running as local servers. 

Implementation Details:
llama.cpp Web Server: A fast, lightweight, pure C/C++ HTTP server used for LLM inference.
Features:
LLM inference of F16 and quantum models on GPU and CPU
OpenAI API compatible chat completions and embeddings routes
Parallel decoding with multi-user support
Continuous batching
Multimodal (work in progress)
Monitoring endpoints
Schema-constrained JSON response format
By leveraging the GRYPHGEN framework, developers can create a self-deploying LLM cooperative programming environment capable of producing any type of software at any level of complexity and scale.

```mermaid
sequenceDiagram
    participant A as Code Generator
    participant B as Code Analyzer
    participant C as Task Monitor
    participant D as Workflow Optimizer
    participant Target_Server as Ubuntu Linux

    A->>+B: Generate code output
    B->>+A: Analyze output for errors
    B->>+C: Check alignment with project parameters
    C->>+B: Monitor output for proper function
    B->>+D: Prevent roadblocks for A, B, and C
    D->>+B: Restart processes as needed
    D->>+C: Revert to previous checkpoints
    A->>+Target_Server: Write code and execute tasks
    B->>+Target_Server: Analyze code for errors and suggestions
    C->>+Target_Server: Ensure alignment with assigned tasks
    D->>+Target_Server: Optimize workflow
```
- Task submission and initialization: The code generator (A) generates code output, which is analyzed by the code analyzer (B) for errors and alignment with project parameters. The task monitor (C) ensures that the output functions properly and aligns with assigned tasks.

- Code generation and analysis: The code generator writes the code and executes tasks on the target server (Target_Server), while the code analyzer analyzes the code for errors and suggestions.

- Task monitoring and workflow optimization: The task monitor ensures alignment with assigned tasks and reverts to previous checkpoints if necessary. The workflow optimizer (D) optimizes the process by restarting processes as needed and preventing roadblocks for components A, B, and C.

- Continuous deployment and monitoring: The target server executes tasks and writes code, while the code analyzer and task monitor continuously monitor the process for errors and proper function.

- Adaptive learning and system evolution: The system learns from previous iterations and evolves to improve efficiency and effectiveness over time.

- By breaking down the workflow into these distinct stages, the sequence diagrams provide a comprehensive understanding of how Gryphgen functions and how it can be used to automate software production at scale.


```mermaid
sequenceDiagram
    participant A as Code Generator (LLM A)
    participant B as Code Analyzer (LLM B)
    participant C as Task Monitor (LLM C)
    participant D as Workflow Optimizer (LLM D)
    participant TS as Target_Server


    A->>+B: Generates code and sends output
    B-->>-A: Analyzes output, returns error analysis or feedback


    A->>+C: Sends output for task alignment check
    C-->>-A: Confirms if outputs align with project parameters


    A->>+D: Requests process management
    D-->>-A: Restarts process or reverts to last known good checkpoint


    A->>+TS: Connects to server, executes development tasks
    B->>+TS: Analyzes and reasons about output from A
    C->>+TS: Ensures A's outputs align with project parameters
    D->>+TS: Manages workflow, avoids roadblocks, maintains efficiency


    loop Health Monitoring
        D->>D: Monitors system health and performance
    end


    loop Dynamic Output Adjustment
        C->>C: Reviews outputs of A and B
        D->>C: Adjusts processes based on C's feedback
    end


    loop Continuous Deployment
        A->>TS: Deploys code
        B->>TS: Deploys feedback mechanisms
        C->>TS: Ensures continuous alignment
        D->>TS: Ensures smooth deployment
    end
```

---

## Git Workflow and Development Process

GRYPHGEN follows a structured git workflow using feature branches and Claude Code for development automation. All development happens on dedicated branches prefixed with `claude/` and includes session IDs for traceability.

### Quick Start for Contributors

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GRYPHGEN
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b claude/your-feature-name-sessionID
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

4. **Push to remote (with retry logic)**
   ```bash
   git push -u origin claude/your-feature-name-sessionID
   ```

### Branch Strategy Diagram

```mermaid
gitGraph
    commit id: "Initial commit"
    commit id: "Core: SYMORQ"
    commit id: "Core: SYMORG"
    branch claude/feature-mcp-server
    checkout claude/feature-mcp-server
    commit id: "Add MCP server"
    commit id: "Add API endpoints"
    checkout main
    merge claude/feature-mcp-server
    branch claude/add-documentation
    checkout claude/add-documentation
    commit id: "Add git workflow docs"
    commit id: "Add mermaid diagrams"
    checkout main
    merge claude/add-documentation
    commit id: "Release v1.0"
```

### Development Workflow with Claude Code

```mermaid
sequenceDiagram
    participant User
    participant Claude as Claude Code
    participant Git as Local Git
    participant Remote as GitHub

    User->>Claude: Request feature implementation
    Claude->>Git: Create/checkout feature branch

    loop Development
        Claude->>Claude: Implement changes
        Claude->>Git: Stage and commit
    end

    Claude->>Remote: Push with retry logic

    alt Push successful
        Remote-->>Claude: ✅ Push confirmed
        Claude-->>User: Branch ready for PR
    else Network error
        Claude->>Claude: Exponential backoff (2s, 4s, 8s, 16s)
        Claude->>Remote: Retry push (max 4 attempts)
    end
```

### Branch Naming Convention

All branches must follow this format for successful pushes:

```
claude/{descriptive-name}-{sessionID}
```

**Examples:**
- `claude/add-git-mermaid-diagrams-01A7FntuzYu1pDpWAiqwbqSv`
- `claude/implement-rag-system-AbC123XyZ456`
- `claude/fix-zeromq-bug-DeF789UvW012`

### Commit and Push Workflow

```mermaid
flowchart TD
    A[Make Code Changes] --> B[Stage Changes]
    B --> C[Write Commit Message]
    C --> D[Commit Locally]
    D --> E[Push to Remote]

    E --> F{Push Success?}
    F -->|Yes| G[✅ Complete]
    F -->|Network Error| H{Retry < 4?}
    F -->|Auth Error| I[Check Branch Name]

    H -->|Yes| J[Exponential Backoff]
    J --> K[Wait 2s/4s/8s/16s]
    K --> E
    H -->|No| L[❌ Report Failure]

    I --> M{Correct Format?}
    M -->|No| N[Fix Branch Name]
    M -->|Yes| O[Check Credentials]

    N --> E

    style G fill:#90EE90
    style L fill:#FFB6C1
    style I fill:#FFD700
```

For comprehensive git workflow documentation, see **[docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)**.

For repository structure improvements and organization, see **[docs/STRUCTURE_IMPROVEMENTS.md](docs/STRUCTURE_IMPROVEMENTS.md)**.

---

### Integrated System RoadMap Overview and Sequence Diagram

```mermaid
graph TD
    UI[User Interface]
    TS[Task Submitter]
    SYMORQ[SYMORQ - Systems Orchestration for Resource Quality]
    SYMORG[SYMORG - Systems Orchestration Retrieval Generator]
    SYMAUG[SYMAUG - Smart Yielding Microservices for Agile and Ultra-Portable Grids]
    TServer[Target Server]
    DS[Data Storage]
    ML[Monitoring and Logging]
    SC[Security and Compliance]
    CF[Community Feedback Loop]
    CI[CI/CD Integration]
    TI[Third-party Integrations]
    LLAMA[LLAMA Data Structure Handling]
    LM[Local Language Model]
    BL[Blockchain Layer]
    QC[Quantum Computing Module]

    UI --> TS
    TS --> SYMORQ
    SYMORQ --> SYMORG
    SYMORQ --> SYMAUG
    SYMORG --> SYMORQ
    SYMAUG --> TServer
    TServer --> DS
    TServer --> ML
    ML --> SC
    SC --> TServer
    CF --> UI
    CI --> TServer
    TI --> UI
    TI --> TS
    TI --> SYMORQ
    TI --> SYMORG
    TI --> SYMAUG
    TI --> TServer
    BL --> SC
    QC --> SYMORQ
    QC --> SYMORG
    QC --> SYMAUG
    LLAMA --> LM
    LM --> UI
    LM --> SYMORQ
    LM --> SYMORG
    LM --> SYMAUG

    subgraph Core Components
        SYMORQ
        SYMORG
        SYMAUG
    end

    subgraph Supporting Infrastructure
        TServer
        DS
        ML
        SC
        CI
    end

    subgraph User and Community Engagement
        UI
        CF
        TI
    end

    subgraph Advanced Integration Layers
        BL
        QC
        LLAMA
        LM
    end
```

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Core Documentation
- **[Git Workflow](docs/GIT_WORKFLOW.md)** - Complete git workflow, branching strategy, and development process
- **[Structure Improvements](docs/STRUCTURE_IMPROVEMENTS.md)** - Repository organization and improvement proposals

### Component Documentation
- **[MCP Server](MCP_SERVER/readme.md)** - Model Context Protocol server documentation
- **[Mermaid Diagrams](mermaid/readme.md)** - System architecture diagrams and visualizations
- **[System Overview](System_overview.md)** - High-level system architecture overview

### Additional Resources
- **[Perl Pipes](perl_pipes/readme.md)** - Inter-process communication utilities
- **[gstruct](gstruct/readme.md)** - Data structure handling documentation
- **[Swarm OpenAI](swarm_openai/readme.md)** - OpenAI integration documentation

---

## Contributing

We welcome contributions to GRYPHGEN! To contribute:

1. **Fork the repository** and clone it locally
2. **Create a feature branch** following the naming convention:
   ```bash
   git checkout -b claude/your-feature-name-sessionID
   ```
3. **Make your changes** with clear, descriptive commits
4. **Test thoroughly** to ensure your changes work as expected
5. **Push to your fork** with retry logic:
   ```bash
   git push -u origin claude/your-feature-name-sessionID
   ```
6. **Create a Pull Request** with a clear description of your changes

### Development Guidelines

- Follow the git workflow outlined in [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)
- Write clear, descriptive commit messages
- Include tests for new functionality
- Update documentation as needed
- Ensure code follows project conventions

### Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

## Quick Links

- **Repository Structure**: [docs/STRUCTURE_IMPROVEMENTS.md](docs/STRUCTURE_IMPROVEMENTS.md)
- **Git Workflow**: [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)
- **System Overview**: [System_overview.md](System_overview.md)
- **Issues**: Report bugs and request features via GitHub Issues
- **Pull Requests**: Submit improvements via Pull Requests

---

**GRYPHGEN** - Automating software production at scale with LLM-powered orchestration.
