# Bake-off

Put three language models head to head on your own hardware and see which one
wins - accuracy, speed, and cost - from a simple web dashboard anyone can drive.
Think of it as a speed test for local AI: no accounts, no cloud, your questions
never leave your network.

**New here? Open [INSTALL.html](INSTALL.html) in a browser - it walks you from
zero to a running comparison with copy-paste commands.**

## Install (one command)

On the machine that will run the dashboard (any Ubuntu-ish Linux with systemd):

```bash
curl -fsSL https://raw.githubusercontent.com/jdbarzy/bake-off/HEAD/get.sh | bash
```

That downloads Bake-off to `~/bake-off`, installs it as an always-on app
(auto-starts on boot), and prints the address to open from any device on your
network. Remove it any time with `~/bake-off/uninstall.sh`.

Want a login or https? In `~/bake-off/run.config.sh` set `DASH_AUTH="user:pass"`
and/or `DASH_TLS="self"` (auto-created certificate; browsers ask to trust it
once per device), then re-run `~/bake-off/install.sh` - it restarts the app and
updates the launcher and printed addresses to match.

Prefer not to install services? Clone this repo and run `./run.sh` instead.

### Windows (lab jump hosts)

Often you don't need to install anything on Windows: put Bake-off on any Linux
box in the lab (a GPU machine works - the app is tiny) and open its address
from the jump host's browser. When the jump host must run it, paste this into
PowerShell (needs Python 3 on PATH, no admin rights):

```powershell
irm https://raw.githubusercontent.com/jdbarzy/bake-off/HEAD/get.ps1 | iex
```

It installs to `%USERPROFILE%\bake-off`, starts on sign-in, and updates the
same way (or via Settings > General). Settings live in `start-bakeoff.cmd`;
remove with `uninstall-bakeoff.cmd`. WSL2 with the Linux one-liner also works
where it's allowed.

## Get some models

Bake-off does not ship models - it measures the ones you serve over the
OpenAI `/v1` API. The quickest path is the provisioner, run on each GPU
machine:

```bash
bash mwboot.sh
```

It installs vLLM, serves a VRAM-sized model per GPU with a swap agent beside
it, and prints the exact settings to paste into the app. From then on the
dashboard's model catalog can change what each GPU serves with one tap.

## Topology (optional, for bigger setups)

Point the three slots at any OpenAI-compatible endpoints (vLLM, TensorRT-LLM,
SGLang, NVIDIA NIM, llama.cpp's `llama-server`, LM Studio). A typical layout
with dedicated model hosts:

| Role            | Serves                              | Endpoint                  |
|-----------------|-------------------------------------|---------------------------|
| Model host A    | a large model via vLLM              | http://host-a:8000/v1     |
| Model host B    | 2x 7-8B via vLLM (one per GPU)      | http://host-b:8000/v1     |
|                 |                                     | http://host-b:8001/v1     |
| Control machine | dashboard (+ optional SSH tunnels)  | http://control:15600      |

- `mwboot.sh` provisions a fresh GPU box into a ready endpoint and prints the
  exact settings to paste into the app.
- `vllm-serve.sh` is an example launch script to run ON a model host.
- If a host is only reachable over SSH, list tunnels in `run.config.sh`
  (template: `run.config.example.sh`); the installer keeps them up as services.
- `./verify.sh` checks that every configured endpoint answers.

## Using the dashboard

Three steps, guided on screen: **Power on** (warms the models), **Load task**
(pick a question card or bring your own), **Press Play**. Everything else lives
behind the gear on the control bar - comparison cards, question sets,
export/import of your whole setup, and preferences.

What it shows per model: **Accuracy**, **Response time**, **Output speed**,
**Time to first answer**, and **Tokens used**. A **Compare** panel shows how
alike the answers are; a **Recommendation** panel picks a winner with a
confidence score and a cost-avoided estimate, plus a downloadable report.

### Document ingestion (vision models)

Beyond Q&A, Bake-off ships **Document ingestion** task cards - page images
(invoices, tables, charts, filled-in forms) with exact ground-truth fields.
Models with the **Vision badge** (Gemma 3, Qwen2.5-VL, Llama 3.2 Vision,
Phi-3.5 Vision, ...) read each page directly; grading is per field and fully
deterministic. The Compare panel also reruns every page from **OCR text
alone**, so the report answers the real procurement question: what does a
vision model buy over an OCR pipeline on your documents? Play-view metrics
switch to **Fields read correctly**, **Response time per page**, and **Pages
per minute**.

### Bring your own tests

Load tasks accepts any of:

- **A GitHub/GitLab link** - paste the normal file (or repo) URL. Private
  repos: add an access token.
- **Upload** a `.csv`, `.txt`, or `.json` file.
- **Build questions & answers** - a guided form, the foolproof way to score
  accuracy.
- **Type / paste** questions - one per line, or `Q:`/`A:` pairs.

Graded sets (a `question` column plus an `answer` column, contains-checked)
unlock the Accuracy score; question-only sets measure speed. Strict-rule
grading and document sets are plain files too - see
[docs/task-format.md](docs/task-format.md). Example:

```csv
question,answer
What is the capital of Australia?,Canberra
Who wrote the novel 1984?,Orwell
```

## Moving your setup

Settings > Move setup exports everything (endpoints, comparison cards, saved
lineups, question sets, preferences) as one file; import it on any other
Bake-off. No hand-edited config required.

## Notes

- Grading is contains-based and runs locally.
- Endpoint config lives in `dashboard/models.config.json` (gitignored;
  template: `models.config.example.json`) - but the app edits it for you.
- Verify GPU use on each host during a generation (nvidia-smi / nvtop) before
  trusting numbers - a CPU fallback silently ruins the benchmark.
