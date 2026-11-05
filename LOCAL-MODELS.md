# Running Kyklos on a local model

Kyklos talks to a local [Ollama](https://ollama.com) the same way it talks to any other provider:
through the OpenAI-compatible endpoint, with `BASE_URL` pointed at the machine instead of a vendor.
Everything below is the part that is not obvious from `.env.example`.

## Start here: the context window will not take care of itself

Ollama serves **4096 tokens by default**. Kyklos assumes `DEFAULT_CONTEXT_WINDOW = 128_000` and
compacts at 75% of whatever it believes, so the two together produce a conversation that grows
towards 96,000 tokens against a server that stops reading at 4,096. Nothing errors. The model simply
appears to forget everything but the last few thousand tokens.

4096 is also too small for Kyklos to work at all: the system prompt and the tool schemas take a large
bite before your first message. Setting the window is step two below, and it is not optional.

Kyklos will not detect this for you, deliberately. Ollama's `/api/show` reports the model's *trained*
maximum, not the `num_ctx` the server was started with, and the API does not expose the latter. A
number read from there would arrive labelled `detected` and quietly outrank the default while being
just as wrong — see the comment above `MIN_DETECTED_CONTEXT` in `kyklos.py`.

## 1. Pull a model that can call tools

Kyklos drives everything through tool calls, and the catalog filters to models that advertise
support. A model without it will not appear in `/model`.

```bash
ollama pull qwen3:8b
```

Run it from wherever you like. `ollama pull` is a client command that asks the daemon to fetch the
model into its own store (`~/.ollama/models` unless `OLLAMA_MODELS` says otherwise); the working
directory plays no part. The same goes for `ollama list`, `ollama ps` and `ollama run`.

As a sizing guide on Apple silicon with unified memory: 8B–14B at q4 is comfortable on 16–24 GB,
and a 30B-class model will fit but leave little room for a large KV cache.

## 2. Bake the context window into the model

The reliable way is a `Modelfile`, because it travels with the model and survives everything else.
These are the two commands in this guide that *do* care where you are — the first writes a file into
the current directory and the second reads it back. Run them somewhere disposable rather than inside
a repository, or you will leave a stray `Modelfile` in your working tree. It is not needed once
`ollama create` has baked the parameter into the model.

```bash
printf 'FROM qwen3:8b\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create qwen3-32k -f ./Modelfile
```

The environment-variable routes are worse here than they look:

- `launchctl setenv OLLAMA_CONTEXT_LENGTH ...`, which Ollama's FAQ gives for macOS, applies to the
  Ollama **app**. It is not how a Homebrew service gets its environment.
- Editing `~/Library/LaunchAgents/sh.brew.ollama.plist` does not stick. Homebrew generates that file
  from the formula's `service do` block — which is where `OLLAMA_FLASH_ATTENTION` and
  `OLLAMA_KV_CACHE_TYPE` come from — and regenerates it on the next `brew services restart`.

If you run the server yourself rather than as a service, `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`
works and applies to every model at once.

Ollama's own documentation disagrees with itself about the default — the FAQ says 4096, the Modelfile
page says 2048 — which is reason enough to state the number explicitly rather than inherit one.

## 3. Point Kyklos at it

```
BASE_URL=http://localhost:11434/v1
API_KEY=ollama
MODEL=qwen3-32k
CONTEXT_WINDOW=32768
```

`CONTEXT_WINDOW` must match the `num_ctx` you baked in. That is the entire point of step 2.

If you also use a cloud provider, put the local one in a **profile** instead of swapping these four
lines back and forth — a suffixed block that `kyklos --profile local` selects, leaving the
unsuffixed block as your default. Each profile keeps its own remembered model, which is what makes
the next paragraph survivable:

```
BASE_URL_LOCAL=http://localhost:11434/v1
API_KEY_LOCAL=ollama
MODEL_LOCAL=lfm3-omni-32k:latest
CONTEXT_WINDOW_LOCAL=32768
```

**`MODEL` is the weakest of the three ways Kyklos decides which model to run.** The order is
`--model` > the model last picked with `/model` > `MODEL` in `.env`. A `/model` pick is remembered in
`~/.kyklos/settings.json`, so if you have ever used it, editing `.env` changes nothing and Kyklos
asks the local server for a model it has never heard of. Either pick the local model with `/model`
once you are running, or clear `model` and `subagent_model` from `settings.json` first.
`SUBAGENT_MODEL` works the same way and is just as easy to leave pointing at a cloud model.

`API_KEY` is a placeholder, and `ollama` is the literal value Ollama's own documentation uses. A local
server does not authenticate: it answers with no `Authorization` header, a garbage one, or an empty
one. Kyklos still requires the variable to be non-empty at startup, and the OpenAI client rejects an
empty `api_key` before a request is built, so something has to be there. Put the placeholder in
rather than leaving a real cloud key behind — an unused credential can still leak, and if `BASE_URL`
ever moves back you would be spending without noticing.

## 4. Verify

`ollama ps` has a `CONTEXT` column, which reports what the model actually loaded with:

```bash
ollama ps
```

Then start Kyklos and run `/model`. A local server is read through Ollama's own API, so models arrive
with their real names, tool support and effort levels rather than as bare `/v1/models` ids.

## What changes once you are local

The API key stops being used for anything. Command output stops being flagged as leaving the machine:
`remote_provider_host` returns `''` for localhost, so the `output will be sent to remote provider`
warnings on the approval prompts no longer appear, because they would no longer be true.

One caveat worth keeping in mind. Ollama can serve **cloud-hosted** models through the same local
daemon, authenticated by `ollama signin` rather than by anything in `.env`. In that case Kyklos's own
traffic still goes only to localhost, but the compute and your data do not stay on the machine — and
since the warning is computed from `BASE_URL`, it would call that local. The model tag is what tells
you where the work happens, not the URL.

---

Checked against Ollama 0.40.14 and the docs as of November 2026:
[OpenAI compatibility](https://docs.ollama.com/openai) ·
[FAQ](https://docs.ollama.com/faq) ·
[Modelfile](https://docs.ollama.com/modelfile)
