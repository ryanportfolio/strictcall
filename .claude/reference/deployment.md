# Deployment

> Deploy target, build output, asset paths, publish flow.

## Railway is not wired to GitHub

**Merging to `main` does not deploy.** There is no GitHub integration, no
Railway webhook, and no deploy job in `.github/workflows/ci.yml` — CI only runs
tests, lint, and a Docker smoke test. Confirmed 2026-08-16: a merge to `main`
produced no Railway deployment, and the live site kept serving the previous
build until it was pushed by hand.

Publishing is an explicit CLI step, every time.

## Publish

Deploy from a clean checkout of `main`, never from a worktree — `railway up`
uploads the working directory, so local scratch state would ship with it.

```bash
git clone --depth 1 -b main https://github.com/ryanportfolio/strictcall.git /tmp/deploy
```

```bash
cd /tmp/deploy && railway link --project strictcall --environment production --service strictcall && railway up --service strictcall
```

A build takes roughly 60-90 seconds to go live after `railway up` returns.

## Verify

Do not trust the CLI's success message alone — it reports the upload, not the
cutover. Poll the site for something only the new build contains:

```bash
curl -s https://strictcall-production.up.railway.app/ping
```

```bash
railway deployment list
```

The top row's timestamp should be the deploy you just made. `SUCCESS` on an
older row means the new one has not replaced it yet.

## Facts

| Thing | Value |
|---|---|
| Project / service | `strictcall` / `strictcall` (environment `production`) |
| Public URL | https://strictcall-production.up.railway.app |
| Region | US East |
| Build | The repo `Dockerfile` (python:3.13-slim) |
| Warehouse | Generated at image build time by `python -m strictcall.dataset generate`; `data/` is in both `.gitignore` and `.dockerignore`, so nothing is uploaded |
| Model + key | Railway service variables, set outside the repo. A redeploy keeps them; they are not in any committed file |

## Gotcha: the free model tier moves

`STRICTCALL_MODEL` pointed at an OpenRouter `:free` slug that was withdrawn and
started returning `404 ... This model is unavailable for free`. Free slugs are
not stable — when the live site or `tests/test_live.py` starts 404ing on the
model, check whether the `:free` variant still exists before debugging anything
else.
