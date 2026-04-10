# Multi-Agent Workflow

A two-agent LLM pipeline that maps requirements to test cases — automatically, deterministically, and without hand-coded rules.

Designed for any domain where requirements traceability is mandatory (functional safety, regulated software, systems engineering). The matching logic lives entirely in a structured system prompt, not in code. Agent 1 makes that prompt data-aware. Agent 2 executes it.

---

## Pipeline

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Requirements.xlsx  │  │  TestCases.xlsx      │  │  prompt_file.docx   │
│  ID · Summary       │  │  ID · Name · Type    │  │                     │
│  Description · Type │  │  Pre-Cond · Desc     │  │  The algorithm      │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           └─────────────────────────┼─────────────────────────┘
                                     ▼
           ┌─────────────────────────────────────────────────────────────┐
           │                  AGENT 1 — Prompt Builder                   │
           │                                                             │
           │  Step A  Read both Excel files · cap corpus at 6 000 chars  │
           │  Step B  LLM call → classify vocab into 3 groups            │
           │            activationKeywords  (action verbs, trigger states)│
           │            genericNoise        (folder / process boilerplate)│
           │            commonDomainWords   (component / system nouns)   │
           │  Step C  Inject token lists + column names into template     │
           │                                                             │
           │  AzureChatOpenAI · temperature=0 · max_tokens 3 000         │
           └─────────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────┐
                          │     Final system prompt   │
                          │  (baked in via .partial())│
                          └──────────────┬────────────┘
                                         │
                                         ▼
           ┌─────────────────────────────────────────────────────────────┐
           │                  AGENT 2 — REQ-TEST Mapping                 │
           │                                                             │
           │  Batch loop   10 requirements per LLM call                  │
           │  LLM output   markdown pipe-table per batch                 │
           │  parse_table  pipe-table → 5-tuple rows                     │
           │  Accumulate   all_rows → pandas DataFrame                   │
           │                                                             │
           │  Chain: ChatPromptTemplate | AzureChatOpenAI | StrOutputParser│
           │  temperature=0 · max_tokens 4 000                           │
           └─────────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │           req_test_report.xlsx      │
                    │                                    │
                    │  Sheet 1: REQ-TEST Mapping          │
                    │    green rows = matched             │
                    │    red rows   = NO MATCH            │
                    │  Sheet 2: Summary                   │
                    │    coverage % · matched count       │
                    └────────────────────────────────────┘
```

---

## The prompt is the algorithm

Most LLM pipelines treat the system prompt as a wrapper around real logic that lives in code. This pipeline inverts that.

The matching algorithm — folder-scoped search, vocabulary stratification, activation-gated scoring, tie-breaking rules, global fallback — is defined entirely inside `prompt_file.docx`. It reads like pseudocode:

```
F1. SCORING
    if ReqActivationTokens != empty:
        score_act = |intersect(ReqActivationTokens, TCActivationTokens)|
                    / |ReqActivationTokens|
    else score_act = 1

    score_other = |intersect(ReqOtherCore, TCOtherCore)|
                  / max(1, |ReqOtherCore|)

    score_total = 0  if score_act == 0
                else  0.7 * score_act + 0.3 * score_other
```

Agent 1's sole job is to make this prompt **data-aware** — substituting the actual column names, type values, and LLM-classified token lists from the current files before Agent 2 ever sees it. The matching logic is tuned by editing the prompt, not the code.

Key decisions embedded in the prompt:

- **Three-tier vocabulary stratification.** Tokens are split into activation keywords (action verbs, control states, trigger conditions), domain nouns (component names that provide context but are too broad to discriminate), and process noise (folder labels, test-management boilerplate with zero signal). A flat keyword list conflates all three — this approach does not.

- **Activation-gating with hard zero.** If a requirement contains activation keywords but the candidate test case shares none, `score_total` is zeroed regardless of domain overlap. This prevents requirements that share common component vocabulary from matching the wrong test case.

- **Folder-scoped search before global fallback.** Requirements and test cases are organised under folder hierarchies. The prompt maps the requirement's folder to the nearest test-case folder by domain-token intersection, then restricts candidate search to that folder. Global search only runs when the local candidate set is empty — scored as `0.6 * score_total + 0.4 * score_dom`.

- **Determinism at every level.** `temperature=0` on the LLM. Explicit tie-breaking rules (token overlap → name length → smallest ID). "No reasoning in output" instruction to suppress chain-of-thought preamble that would break the pipe-table parser.

---

## Agent 1 — why LLM for tokenisation

Agent 1 calls the LLM once before any mapping begins, asking it to classify the raw corpus vocabulary into the three tiers above.

A frequency counter surfaces common tokens but cannot distinguish an action verb from a component noun from a process label — all three appear frequently in any large requirements export. The LLM understands the difference from context, produces the correct classification in one pass, and returns a Python dict literal that Agent 1 parses with a sandboxed `eval(clean, {'__builtins__': {}})`.

The result — `activationKeywords`, `genericNoise`, `commonDomainWords` — is substituted into the base prompt alongside auto-detected column names and type values. Agent 2 never sees generic placeholders; it receives a prompt built from the actual files.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Azure OpenAI GPT-4 |
| Chain | LangChain Core — `ChatPromptTemplate \| AzureChatOpenAI \| StrOutputParser` |
| Data | pandas · openpyxl · python-docx |
| Environment | python-dotenv · httpx |
| UI | ipywidgets · Jupyter / VS Code |

No orchestration framework (LangGraph, AutoGen, CrewAI). The agent interaction is strictly sequential — Agent 1 produces a string, Agent 2 consumes it — so a framework adds abstraction cost with no benefit. Every step is visible in a Jupyter cell.

---

## Project structure

```
.
├── req_test_mapping.ipynb   # Main notebook — run cells top to bottom
├── .env                     # API credentials (never commit)
└── InputFiles/
    ├── Requirements.xlsx    # Requirements export  (Export sheet, header row 1)
    ├── TestCases.xlsx       # Test cases export    (Export sheet, header row 1)
    ├── prompt_file.docx     # System prompt — the matching algorithm
    └── req_test_report.xlsx # Output — generated on first run
```

---

## Setup

**1.** Run Cell 0 — installs any missing packages.

**2.** Create `.env` next to the notebook:

```
AZURE_API_KEY=your_key_here
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-01
AZURE_DEPLOYMENT=your_deployment_name
PROXY_URL=http://your.proxy:port
```

**3.** Place your Excel exports and `prompt_file.docx` in `InputFiles/`.

**4.** Run all cells top to bottom. Each cell prints what was loaded, classified, and mapped.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BATCH_SIZE` | `10` | Requirements per LLM call. Raise to 20 for 8k+ token deployments. |
| `REQ_EXCEL_PATH` | `InputFiles/Requirements.xlsx` | Requirements file |
| `VV_EXCEL_PATH` | `InputFiles/TestCases.xlsx` | Test cases file |
| `PROMPT_DOC_PATH` | `InputFiles/prompt_file.docx` | System prompt / algorithm |
| `OUTPUT_PATH` | `InputFiles/req_test_report.xlsx` | Report output path |
