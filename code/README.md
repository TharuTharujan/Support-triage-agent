# HackerRank Orchestrate Support Triage Agent

This is a terminal-based Python hybrid RAG agent for the HackerRank Orchestrate
support triage task.

## Approach

- Loads the provided markdown support corpus from `../data/`.
- Uses local retrieval only: scikit-learn TF-IDF when installed, optional FAISS
  if FAISS and sentence-transformers are installed, and a deterministic lexical
  fallback otherwise.
- Uses `pandas` for CSV input/output when installed, with a standard-library CSV
  fallback so the agent still runs in minimal environments.
- Applies explicit safety rules for fraud, billing, payment, refunds, account
  access, security, legal, privacy, assessment integrity, outages, hiring
  outcomes, prompt injection, and unsupported requests.
- Optionally uses an LLM only to improve responses and justifications from
  retrieved local context; it never uses a model as a source of support policy.
- Always validates the required output schema before writing predictions.

## Run

From the repository root:

```bash
python code/main.py
```

By default this reads:

```text
support_tickets/support_tickets.csv
```

and writes:

```text
support_tickets/output.csv
```

Output columns are written in this exact order:

```text
issue,subject,company,response,product_area,status,request_type,justification
```

You can override paths:

```bash
python code/main.py --input support_tickets/support_tickets.csv --output code/dry_run_output.csv
```

## Test

```bash
python -m unittest discover -s code
```

## Optional Dependencies

The agent runs without these packages, but uses them when available:

```bash
pip install pandas scikit-learn
```

Optional vector retrieval:

```bash
pip install faiss-cpu sentence-transformers
```

## Generation Modes

### Default deterministic mode

If no supported API key is configured, the agent uses deterministic template
responses and justifications based on retrieved local corpus chunks. This mode
does not make network calls.

### Optional LLM mode

If a supported API key is configured, the agent asks that provider to improve
the response and justification using only the retrieved local corpus chunks. If
the API call fails, times out, or returns an unusable response, the agent falls
back to deterministic generation.

Supported providers are detected through environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `MISTRAL_API_KEY`
- `TOGETHER_API_KEY`
- `COHERE_API_KEY`
- `AZURE_OPENAI_API_KEY` with `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT`
- `OPENAI_COMPATIBLE_API_KEY` with `OPENAI_COMPATIBLE_BASE_URL`

The terminal run report prints the detected provider name, such as
`LLM provider: gemini`, and whether the generated CSV used `llm-assisted` or
`local fallback` generation. It never prints API key values.

Examples:

```powershell
$env:GEMINI_API_KEY="your_key_here"
python code\main.py
```

```powershell
$env:OPENROUTER_API_KEY="your_key_here"
python code\main.py
```

Never commit API keys. Use environment variables only.

## Design Notes

The implementation favors conservative escalation over unsupported guessing.
Responses are grounded in retrieved document titles and snippets from the local
corpus. Cases involving verified authority, payments, fraud, stolen cards,
security vulnerabilities, platform-wide outages, or hiring decisions are routed
to support instead of being resolved directly.
