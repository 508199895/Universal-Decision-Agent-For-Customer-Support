# Universal Decision Agent: Multi-Agent Customer Support Orchestrator

Universal Decision Agent is a fully functional multi-agent support automation system built with LangGraph, LangChain, FastMCP, and SQLite.
It processes customer support tickets end-to-end through intelligent agents that ingest, classify, resolve, and escalate issues while integrating data from:

Knowledge Base (KB)

Account & Reservation System (CultPass)

Long-Term Memory

The system chooses actions dynamically using a Supervisor agent and tools, following a hierarchical multi-agent pattern that supports real-time decision making and human escalation.

This project meets and exceeds the Udacity rubric for the Multi-Agent Editing & Decisioning track.

## Table of Contents

1. Overview
2. Architecture & Design
    * Multi-Agent Pattern
    * Architecture Diagram
    * State Schema
3. Agents & Responsibilities
4. Knowledge, Tools & MCP Services
5. Ticket Processing Flow
6. Short-Term & Long-Term Memory
7. Repository Structure
8. Setup Instructions
9. Running the App
10. Running Tests

## Overview

Universal Decision Agent simulates a production-grade customer-support agent system, capable of:

* Understanding incoming tickets

* Classifying issue type & urgency

* Searching KB for solutions

* Looking up user profiles & reservations

* Consulting long-term memory for prior issues

* Attempting resolution with confidence scoring

* Escalating to human agents when needed

* Producing structured explanations for support teams

It runs on a hybrid of rule-based routing and agentic reasoning, and integrates multiple data sources via FastMCP servers that behave like external microservices.

## Architecture & Design

### Multi-Agent Pattern

The system uses a Supervisor / Hierarchical agent design:
    * Supervisor Agent acts as the central router
    * Intake Agent standardizes the ticket
    * Classifier Agent determines issue characteristics
    * Resolver Agent uses tools & memory to attempt solutions
    * Escalation Agent handles human-handoff summaries
Each agent writes into a shared LangGraph state, and the supervisor decides the next step based on the evolving state.

### Architecture Diagram

![Graph](https://github.com/shilpamadini/Universal-Decision-Agent-For-Customer-Support/blob/8765e484244cbeb24af5deb5dff2d88e7cbcf8fd/docs/design/Graph.jpeg)


### State Management

The LangGraph state schema:

```
class TicketState(TypedDict, total=False):
    ticket: Dict[str, Any]
    intake: Dict[str, Any]
    classification: Dict[str, Any]
    resolution: Dict[str, Any]
    escalation: Dict[str, Any]
    supervisor: Dict[str, Any]
```

Stored per thread_id using MemorySaver.


## Ticket Events and Tracing

Workflow messages, lifecycle events, tool results, errors, and token usage are persisted with the customer-support ticket in the ticket event store. LangSmith provides Agent and tool-call tracing. The active workflow does not use the root `logger.py` utility.
## Agents & Responsibilities

1. Intake Agent

Normalizes incoming ticket content.

Outputs:
* summary
* normalized_issue
* sentiment

2. Classifier Agent

Predicts:

* issue_type (login / billing / reservation / account / other)
* urgency (low / medium / high)
* complexity (low / medium / high)
* escaltion
* rationale

3. Resolver Agent

Uses MCP tools:

* kb_search
* account_get_user
* account_get_user_reservations
* memory_search
* memory_write

Produces:

* status: resolved / needs_escalation
* answer
* confidence
* used_kb_articles
* notes_for_human

4. Escalation Agent

Generates human-ready summary including:

* issue summary
* recommended department
* next steps
* optional inclusion of resolver notes

5. Supervisor Agent

Routes between:

* resolver
* escalation
* done

Rules include escalation checks, resolver confidence threshold, and classification hints.

## Knowledge, Tools & MCP Services

Universal Decision Agent  connects to three FastMCP microservices:

### KB Server (kb/)

* Searches the knowledge base table
* Tool: kb_search(query, limit)

### Account Server (account/)

* Fetches user profile + user reservations
* Tools:
    account_get_user(external_user_id)
    account_get_user_reservations(external_user_id)

### Memory Server (memory/)

* Long-term memory write & search
* Tools:
    memory_write(...)
    memory_search(...)
    memory_get_all(external_user_id)
These tools are wrapped and used inside the Resolver Agent.

## Ticket Processing Flow
```
Incoming ticket
    ↓
Intake Agent (normalize)
    ↓
Classifier Agent (problem type, urgency)
    ↓
Supervisor Agent
    ↳ Resolver Agent → try to solve via tools
        ↳ If confident: DONE
        ↳ If not: back to Supervisor
    ↳ Escalation Agent → human summary
    ↳ DONE

```
Everything is stored in the LangGraph state and returned as the final output.

## Memory Design

### Short-Term (Per-Ticket)

* Backed by LangGraph MemorySaver
* Stores intermediate agent outputs

### Long-Term (Cross-Ticket)

* SQLite backed
* Accessed via MCP
* Stores historical issue content, metadata, timestamps
* Used by Resolver to personalize answers

## Repository Structure

```text
uda-hub/
├─ agentic/
│  ├─ app/cli.py                    # L6: local CLI
│  ├─ orchestration/                # L5: workflow, State, pause and recovery
│  ├─ agents/                       # L4: current single Agent and agent modules
│  ├─ capabilities/tools/           # L3: Agent-facing tools
│  ├─ data/tickets/                 # L2: ticket persistence and event records
│  ├─ capabilities/mcp/             # L2: MCP client
│  └─ evals/                        # Evaluation scripts, datasets, and results
├─ mcp_services/                    # MCP service implementations
├─ tests/
├─ notebooks/                       # Database setup notebooks and notebook helpers
├─ scripts/                         # Migration, development, and diagram utilities
├─ data/                            # Existing database models, source data, and local databases
├─ docs/                            # Agent rules, design, and research
├─ .scratch/<feature>/              # Fixed location for generated Specs and task tickets
├─ CONTEXT.md
├─ AGENTS.md
├─ requirements.txt
└─ README.md
```

Existing support-workflow Specs and architecture documents remain in `docs/design/`. New feature Specs and task tickets follow `docs/agents/issue-tracker.md` and remain under `.scratch/<feature>/`.
## Setup Instructions
1. Use the project-local environment. The checked-out `.venv` already contains the cloned environment. If it is missing, create it and install dependencies:

```
conda create --prefix .venv python=3.11 -y
.\.venv\python.exe -m pip install -r requirements.txt

```

2. Configure environment variable

```
touch .env

```
Edit the .env to add key for openai 

```
OPENAI_API_KEY=your_openai_key_here

```

3. Build databases

Run:

* 01_external_db_setup.ipynb

* 02_core_db_setup.ipynb

These create:

* data/external/cultpass.db

* data/core/udahub.db

## Running the App

Run a demo ticket:

```
.\.venv\python.exe -m agentic.app.cli --mode demo

```

Run interactive mode:

```
.\.venv\python.exe -m agentic.app.cli --mode chat

```

Run the FastAPI customer-support API and Feishu bridge:

```
.\.venv\python.exe -m uvicorn agentic.app.fastapi_server:app --host 127.0.0.1 --port 8000
```

Submit a customer message to `POST /tickets` with only the database `user_id` and message text. The API reads `external_user_id` and `user_name` from `data/core/udahub.db`, then returns `202 Accepted` with the new ticket ID and `pending` status. The local MCP subprocesses start with the application; the Qdrant Docker service must already be available.

```json
{
  "user_id": "<existing user ID from the users table>",
  "message": "我想查询我的账户状态"
}
```

The service requires `DEEPSEEK_API_KEY` and the existing Feishu settings (`LARK_APP_ID`, `LARK_APP_SECRET`, `FEISHU_CHAT_ID`). `UDAHUB_DB_PATH` can override the default user database path.

## Running Tests

```
.\.venv\python.exe -m pytest --basetemp=.pytest-tmp-current -q

```

Pytest clears `.pytest-tmp-current` before each run; reserve this directory for test output.

You should see the test output as:

```
8 passed in XXs

```
