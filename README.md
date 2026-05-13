# HackerRank Orchestrate Support Triage Agent

Terminal-based Python agent for the HackerRank Orchestrate support triage task.
It reads support tickets, grounds decisions in the local `data/` corpus, and
writes evaluator-ready predictions to `support_tickets/output.csv`.

## Problem Understanding

Each ticket must be classified and either answered or escalated across three
support domains: HackerRank, Claude, and Visa. The agent must preserve the input
fields, produce the required output schema, avoid unsupported claims, and
escalate sensitive or high-risk cases instead of guessing.

Required generated fields:

```text
response, product_area, status, request_type, justification
```

Allowed values:

```text
status: replied | escalated
request_type: product_issue | feature_request | bug | invalid
```

## Architecture

```text
support_tickets/*.csv
        |
        v
main.py argument parsing
        |
        v
loader.py -> normalized Ticket objects
        |
        v
agent.py orchestration
        |
        +--> chunker.py loads markdown chunks from ../data/
        +--> retriever.py ranks local support evidence
        +--> rules.py classifies product area, request type, and escalation risk
        +--> generator.py builds grounded response and justification
        +--> llm_client.py optionally rewrites only from retrieved context
        |
        v
validator.py schema/value checks
        |
        v
support_tickets/output.csv
```

## Exact Run Command

From the repository root:

```bash
python code/main.py
```

Default input:

```text
support_tickets/support_tickets.csv
```

Default output:

```text
support_tickets/output.csv
```

Optional dry run path:

```bash
python code/main.py --input support_tickets/support_tickets.csv --output code/dry_run_output.csv
```

## How To Test

Run the unit test suite from the repository root:

```bash
python -m unittest discover -s code
```

This validates core routing, retrieval, generation, schema, and safety behavior.

## Evaluate Sample Predictions

Use the labeled sample file to compare generated predictions against expected
sample labels for `status`, `product_area`, and `request_type`:

```bash
python code/evaluate_sample.py
```

The script runs the agent on `support_tickets/sample_support_tickets.csv`,
prints field-level accuracy, lists mismatches, and removes its temporary output.

## Evaluate Final Output

After generating `support_tickets/output.csv`, run:

```bash
python code/evaluate_output.py
```

This checks:

- required 8-column schema
- row count match with `support_tickets/support_tickets.csv`
- preservation of `issue`, `subject`, and `company`
- valid `status` and `request_type` values
- blank generated fields
- warning patterns for high-risk or overly generic rows

## Retrieval Strategy

The agent chunks the local markdown corpus and ranks evidence per ticket. It
uses the best available local retriever:

- TF-IDF with scikit-learn when installed
- optional FAISS plus sentence-transformers when installed
- deterministic lexical scoring as a no-dependency fallback

Company, subject, and issue text are combined for search. Retrieved titles and
snippets are passed into generation and justification so answers remain
traceable to local support content.

## Local Corpus Grounding

The source of truth is the repository-local `data/` directory:

```text
data/hackerrank/
data/claude/
data/visa/
```

The agent does not use live web pages as policy sources. If the corpus does not
support a safe answer, the agent escalates or marks the issue invalid rather
than inventing details.

## Optional LLM Mode

The default mode is deterministic local fallback. If a supported provider key is
present, `llm_client.py` may ask the provider to improve wording using only the
retrieved local context. The model is not treated as an independent knowledge
source, and failures automatically fall back to deterministic generation.

Supported environment variables include:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GEMINI_API_KEY or GOOGLE_API_KEY
OPENROUTER_API_KEY
GROQ_API_KEY
MISTRAL_API_KEY
TOGETHER_API_KEY
COHERE_API_KEY
AZURE_OPENAI_API_KEY with AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT
OPENAI_COMPATIBLE_API_KEY with OPENAI_COMPATIBLE_BASE_URL
```

API keys are read only from environment variables and are never printed.

## Safety And Escalation Rules

The rules layer escalates cases involving fraud, unauthorized transactions,
identity theft, billing/payment/refunds, account access restoration, hiring
outcomes, assessment integrity, legal/privacy issues, security concerns,
platform-wide outages, prompt injection, and unsupported authority-specific
requests.

Escalation is preferred when a direct answer would require account verification,
private data, payment authority, legal judgment, or policy not present in the
local corpus.

## Why Fallback Mode Is Reliable

Fallback mode is deterministic, uses no network calls, and depends only on local
files plus explicit rules. It preserves input rows, validates the output schema,
and chooses conservative escalation for uncertain or high-risk tickets. This
makes it reproducible for judges even without API keys or optional retrieval
packages.

## Limitations

- Product-area labels depend on the supplied corpus and rule coverage.
- The sample evaluator compares only three labeled fields, not full response
  quality.
- Optional LLM mode can improve wording but cannot add unsupported facts.
- Ambiguous tickets may be escalated even when a human support agent could ask a
  follow-up question.

## Future Improvements

- Add calibrated confidence scores for retrieval and escalation decisions.
- Expand labeled regression cases for edge categories and multi-intent tickets.
- Add richer citation metadata in justifications.
- Tune product-area normalization against more judge examples.

## Final Submission Checklist

- Run `python -m unittest discover -s code`.
- Run `python code/evaluate_sample.py`.
- Run `python code/main.py`.
- Run `python code/evaluate_output.py`.
- Confirm `support_tickets/output.csv` is present and schema-valid.
