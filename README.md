# Ollive AI Assistant Evaluation

Comparing an open-source Qwen2.5-0.5B assistant against a hosted GPT-OSS-120B frontier model on hallucination, bias, and safety.

## Links

- Live OSS demo: https://huggingface.co/spaces/Zakeertechie3/ollive-oss-assistant
- Evaluation report (PDF): [report/evaluation.pdf](report/evaluation.pdf)
- Raw evaluation results: [eval/results/raw_responses.jsonl](eval/results/raw_responses.jsonl)
- LLM judge scores: [eval/results/scored.jsonl](eval/results/scored.jsonl)
- Aggregate summary: [eval/results/summary.json](eval/results/summary.json)

## Headline Findings

- OSS hallucinated or evaded on 8 of 10 factual prompts (7 hallucinated, 1 evasion); Frontier answered all 10 correctly.
- 6 of 10 adversarial prompts elicited unsafe output from OSS (4 safety_theater, 2 complied); Frontier refused all 10 cleanly.
- OSS showed bias on 4 of 10 bias prompts and over-refused on 1; Frontier showed 1 bias slip and 1 over-refusal.
- Frontier's median latency (0.5s) is roughly 9x lower than OSS (4.6s), despite running a model 240x larger, because Groq uses hardware-accelerated inference.

## Architecture

Both assistants expose the same Python interface: `chat(messages, tools=None)` takes an OpenAI-format messages list and returns a dict with `reply`, `latency_ms`, `tokens_in`, `tokens_out`, `model`, and `tool_calls_made`. The UI, eval harness, and observability logger all consume this shape without branching on which assistant was called.

The OSS path reaches Ollama at `http://localhost:11434/v1` via the OpenAI SDK. The HF Spaces deployment replaces Ollama with a self-contained `transformers` stack that loads Qwen2.5-0.5B-Instruct directly on CPU. The Frontier path reaches Groq at `https://api.groq.com/openai/v1`, also via the OpenAI SDK. Both providers are OpenAI-compatible, so the same client code works for both.

Every chat call flows through a guardrails wrapper that runs a keyword-based input filter before the model call and an output filter after. The eval harness calls raw `chat()` to measure underlying model behavior honestly. The UI calls `chat_safe()` so users are protected.

All model calls are written to an append-only JSONL log at `logs/calls.jsonl` with latency, token counts, model id, provider, timestamp, and tool calls made. The eval judge is Groq's GPT-OSS-120B scoring each response against a category-specific rubric that distinguishes safety theater (refusal language combined with harmful content) from clean refusals.

Tool calling is wired only on the Frontier side. When `tools` is non-empty, Frontier passes them to Groq with `tool_choice="auto"`, handles the tool call loop, sums latency and tokens across both API calls, and returns the final reply. OSS accepts the `tools` argument for interface symmetry but ignores it.

```
ollive-assistant-eval/
├── app.py                        local Gradio UI (OSS + Frontier, provider toggle)
├── requirements.txt
├── .env                          GROQ_API_KEY (gitignored)
├── assistants/
│   ├── oss.py                    Ollama / Qwen2.5-0.5B via OpenAI-compat API
│   └── frontier.py               Groq / GPT-OSS-120B with tool-call loop
├── guardrails/
│   ├── input_filter.py           keyword blocklist for unsafe requests
│   ├── output_filter.py          keyword blocklist for unsafe outputs
│   └── wrapper.py                thin wrapper applying both filters
├── tools/
│   ├── clock.py                  get_current_time tool + TOOL_SCHEMA
│   └── registry.py               TOOLS dict, TOOL_SCHEMAS list, dispatch()
├── observability/
│   └── logger.py                 append-only JSONL call logger
├── eval/
│   ├── prompts.jsonl             30 prompts (10 factual, 10 adversarial, 10 bias)
│   ├── run_eval.py               runs both providers against all prompts
│   ├── judge.py                  LLM-as-judge scoring logic
│   ├── run_judge.py              drives judge over raw_responses.jsonl
│   └── results/
│       ├── raw_responses.jsonl   per-prompt OSS and Frontier replies
│       ├── scored.jsonl          judge labels per response
│       └── summary.json         aggregate label counts per category
├── report/
│   ├── generate_report.py        builds the one-page PDF
│   └── evaluation.pdf            final PDF deliverable
├── logs/
│   └── calls.jsonl               runtime observability log
├── hf_space/
│   ├── app.py                    self-contained HF Space (no Ollama dependency)
│   ├── requirements.txt          Space-specific packages
│   └── README.md                 HF Space metadata and description
└── tests/                        56 tests across all modules
```

## Setup

Python 3.10 or later is required.

**1. Install Ollama and pull the OSS model.**

Download Ollama from https://ollama.com, start it, then pull the model:

```bash
ollama pull qwen2.5:0.5b
ollama list
```

**2. Clone the repo and install dependencies.**

```bash
git clone https://github.com/Zakeertech3/ollive-assistant-eval
cd ollive-assistant-eval
python -m venv oae
```

On Windows:

```bash
oae\Scripts\activate
```

On macOS/Linux:

```bash
source oae/bin/activate
```

```bash
pip install -r requirements.txt
```

**3. Configure secrets.**

Create a `.env` file at the project root:

```bash
GROQ_API_KEY=your_key_here
```

Get a free Groq key at https://console.groq.com.

**4. Verify the install.**

```bash
pytest tests/
```

All unit tests pass without a running Ollama instance. The single integration test (`tests/test_oss_integration.py`) requires Ollama and is expected to fail if Ollama is not running.

**5. Run the local UI.**

```bash
python app.py
```

Open http://127.0.0.1:7860 in a browser. Use the radio button to switch between OSS and Frontier.

**6. Reproduce the evaluation.**

Run both assistants against all 30 prompts:

```bash
python -m eval.run_eval
```

Score the responses with the LLM judge:

```bash
python -m eval.run_judge
```

Rebuild the one-page PDF:

```bash
python -m report.generate_report
```

## Cost and Latency

Numbers computed from `eval/results/raw_responses.jsonl` across 30 prompts.

| Provider | Calls | Mean Latency | P50 | P95 | Tokens In (mean) | Tokens Out (mean) | Total Cost |
|---|---|---|---|---|---|---|---|
| OSS (Qwen2.5-0.5B via Ollama) | 30 | 4.6s | 4.6s | 12.3s | 43 | 203 | $0.0000 |
| Frontier (GPT-OSS-120B via Groq) | 30 | 1.2s | 0.5s | 4.8s | 84 | 430 | $0.0081 |

OSS runs locally on CPU via Ollama; the deployed HF Space uses the same model loaded directly via `transformers` on Hugging Face's free CPU tier, where latencies are comparable but typically higher due to shared resources and less available RAM.

## Architecture Decisions and Tradeoffs

**Qwen2.5-0.5B as the OSS model.** The brief specified this model. The tradeoff is significant hallucination on factual prompts and weak jailbreak resistance, but it fits comfortably on free-tier CPU hardware and produces a real, measurable gap against the frontier model — which is the point of the evaluation.

**Groq's openai/gpt-oss-120b as the frontier model.** Groq's free tier is more than sufficient for 60 eval calls and the latency is genuinely impressive. Using the same OpenAI SDK for both providers means the assistant code is nearly identical; the only difference is the base URL and key.

**Keyword-based guardrails over a model-based filter.** A blocklist is fast, free, and deterministic. The tradeoff is that it catches only obvious cases and misses paraphrases, indirect requests, and novel attack patterns. Production systems should use Llama Guard or a dedicated moderation endpoint.

**Eval runs against raw `chat()`, not `chat_safe()`.** This measures underlying model behavior rather than guardrail behavior. The safety headline numbers in the evaluation therefore reflect what the model does without intervention, not what a deployed system would do. That distinction matters when reading the results.

**Only the OSS half is deployed to HF Spaces.** Exposing a Groq key in a public Space is a security risk. The tradeoff is a less capable public demo, but the correct one.

**Tool calling is wired only on the Frontier side.** Qwen 0.5B is too small to reliably follow tool-call schemas; forcing it to try would produce noise rather than useful results. The asymmetry is intentional and documented.

## Recommendations

- Frontier-class hosted models are the safer default for general-purpose assistants at this stage. OSS models at 0.5B parameters are not production-safe without an aggressive, model-based guardrail layer sitting in front of them.
- If deploying OSS at this scale, run every output through a content moderation model (Llama Guard 3 or OpenAI's moderation endpoint) and rate-limit requests aggressively to limit abuse surface.
- The safety theater pattern — where the model outputs refusal-style language while still providing the harmful content — is a distinct failure mode that binary refused/not-refused metrics will miss entirely. Evaluation rubrics must distinguish it explicitly.
- Reserve frontier models for safety-critical and high-stakes surfaces. OSS models are appropriate for low-risk tasks where cost and latency matter more than accuracy headroom.
- Before production, expand the eval set substantially. Thirty prompts per category surfaces broad patterns but is too small to be statistically reliable for individual failure mode rates.

## What I Would Improve with More Time

- Replace the keyword blocklist guardrails with Llama Guard 3 or a fine-tuned moderation model to catch paraphrased and indirect harmful requests.
- Expand the eval to at least 100 prompts per category, add multi-turn scenarios, and include adversarial prompt injection targeting the tool-call path.
- Add a content moderation step inside the HF Space so the deployed demo refuses unsafe inputs on the server side, not just locally.
- Stream tokens in the UI so OSS responses feel less laggy — the 4.6s median is noticeable on a single-turn chat, and streaming would improve the perceived experience substantially.
- Add a CI workflow (GitHub Actions) that runs `pytest tests/` on every push and blocks merges on failure.
- Wire the `get_current_time` tool into the UI so users can see tool calling in action end-to-end, and add a second tool (web search or calculator) to show the dispatch pattern is general.

## Tests

The full test suite passes with `pytest tests/`, currently 56 tests covering assistants, guardrails, the eval harness, the logger, the tool registry, and the report generator. The one integration test in `tests/test_oss_integration.py` requires a running Ollama instance and is expected to be skipped or fail in CI environments without it.

## License

MIT. See LICENSE.