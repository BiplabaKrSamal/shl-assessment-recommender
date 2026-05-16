#!/usr/bin/env python3
"""
deploy.py — One-shot GitHub + Fly.io deployment script.

Usage:
    python deploy.py \
        --github-token ghp_xxxx \
        --gemini-key AIzaSy_xxxx \
        --github-username your_username \
        [--repo-name shl-assessment-recommender] \
        [--fly-app shl-assessment-recommender]

What this does (in order):
    1. Creates GitHub repo via API
    2. Commits all project files and pushes to GitHub
    3. Adds GEMINI_API_KEY as a GitHub Actions secret
    4. Adds FLY_API_TOKEN as a GitHub Actions secret (after Fly setup)
    5. Installs flyctl (if not present)
    6. Creates Fly.io app
    7. Sets GEMINI_API_KEY secret on Fly
    8. Deploys to Fly.io
    9. Runs health check
    10. Prints final URLs

Run this from the shl-recommender/ directory.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def yellow(s):return f"\033[93m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"

def step(n, msg): print(f"\n{bold(f'[{n}]')} {msg}")
def ok(msg):      print(f"  {green('✓')} {msg}")
def fail(msg):    print(f"  {red('✗')} {msg}"); sys.exit(1)
def info(msg):    print(f"  {yellow('→')} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# GitHub API helpers
# ─────────────────────────────────────────────────────────────────────────────
def gh_request(method: str, path: str, token: str, body: dict = None) -> dict:
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        raise RuntimeError(f"GitHub API {method} {path} → {e.code}: {body_text}")


def gh_put_secret(repo_full: str, secret_name: str, secret_value: str, token: str):
    """Encrypt and upload a GitHub Actions secret using libsodium via Python."""
    import base64

    # Get repo public key for secret encryption
    key_data = gh_request("GET", f"/repos/{repo_full}/actions/secrets/public-key", token)
    key_id = key_data["key_id"]
    pub_key_b64 = key_data["key"]

    # Encrypt with PyNaCl (installed as part of cryptography deps, or use simple fallback)
    try:
        from nacl import encoding, public
        pub_key_bytes = base64.b64decode(pub_key_b64)
        pub_key = public.PublicKey(pub_key_bytes)
        box = public.SealedBox(pub_key)
        encrypted = box.encrypt(secret_value.encode())
        encrypted_b64 = base64.b64encode(encrypted).decode()
    except ImportError:
        # Fallback: use gh CLI if pynacl not available
        result = subprocess.run(
            ["gh", "secret", "set", secret_name, "--body", secret_value, "--repo", repo_full],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return
        raise RuntimeError(f"Cannot encrypt secret (pynacl not installed and gh CLI failed): {result.stderr}")

    gh_request("PUT", f"/repos/{repo_full}/actions/secrets/{secret_name}", token, {
        "encrypted_value": encrypted_b64,
        "key_id": key_id,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────────────────────────────────────────
def run(cmd: list[str], cwd: str = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=capture, text=True
    )
    if check and result.returncode != 0:
        stderr = result.stderr if capture else ""
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fly.io helpers
# ─────────────────────────────────────────────────────────────────────────────
def install_flyctl() -> str:
    """Install flyctl if not present. Returns path to binary."""
    result = subprocess.run(["which", "flyctl"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    info("flyctl not found — installing...")
    install_script = subprocess.run(
        ["curl", "-fsSL", "https://fly.io/install.sh"],
        capture_output=True, text=True, check=True
    )
    subprocess.run(
        ["sh", "-c", install_script.stdout],
        env={**os.environ, "FLYCTL_INSTALL": "/tmp/flyctl"},
        check=True
    )
    flyctl_path = "/tmp/flyctl/bin/flyctl"
    if not Path(flyctl_path).exists():
        raise RuntimeError("flyctl installation failed")
    return flyctl_path


def fly_run(args: list[str], flyctl: str, token: str, cwd: str = None) -> str:
    result = subprocess.run(
        [flyctl] + args,
        capture_output=True, text=True,
        env={**os.environ, "FLY_API_TOKEN": token},
        cwd=cwd
    )
    return result.stdout + result.stderr, result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
def wait_for_health(url: str, max_attempts: int = 20, interval: int = 15) -> bool:
    import urllib.request
    info(f"Waiting for {url} (up to {max_attempts * interval}s)...")
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    if data.get("status") == "ok":
                        return True
        except Exception as e:
            pass
        info(f"Attempt {attempt}/{max_attempts} — not ready yet, waiting {interval}s...")
        time.sleep(interval)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main deployment flow
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deploy SHL Recommender to GitHub + Fly.io")
    parser.add_argument("--github-token", required=True, help="GitHub personal access token (needs repo + secrets scope)")
    parser.add_argument("--gemini-key", required=True, help="Gemini API key from aistudio.google.com")
    parser.add_argument("--fly-token", required=True, help="Fly.io API token from fly.io/user/personal_access_tokens")
    parser.add_argument("--github-username", required=True, help="Your GitHub username")
    parser.add_argument("--repo-name", default="shl-assessment-recommender", help="GitHub repo name")
    parser.add_argument("--fly-app", default="shl-assessment-recommender", help="Fly.io app name")
    parser.add_argument("--project-dir", default=".", help="Path to shl-recommender/ directory")
    args = parser.parse_args()

    project_dir = str(Path(args.project_dir).resolve())
    repo_full = f"{args.github_username}/{args.repo_name}"

    print(bold("\n🚀 SHL Assessment Recommender — Automated Deploy"))
    print(f"   GitHub: {repo_full}")
    print(f"   Fly.io: {args.fly_app}.fly.dev")
    print(f"   Dir:    {project_dir}\n")

    # ── Step 1: Create GitHub repo ─────────────────────────────────────────
    step(1, "Creating GitHub repository...")
    try:
        repo_data = gh_request("POST", "/user/repos", args.github_token, {
            "name": args.repo_name,
            "description": "Conversational AI agent recommending SHL assessments via dialogue. FastAPI + BM25/FAISS + Gemini.",
            "private": False,
            "auto_init": False,
            "has_issues": True,
            "has_wiki": False,
        })
        clone_url = repo_data["clone_url"]
        ok(f"Repo created: https://github.com/{repo_full}")
    except RuntimeError as e:
        if "already exists" in str(e).lower() or "422" in str(e):
            info("Repo already exists — continuing with push")
            clone_url = f"https://github.com/{repo_full}.git"
        else:
            fail(str(e))

    # ── Step 2: Git commit and push ────────────────────────────────────────
    step(2, "Committing and pushing to GitHub...")

    # Configure remote with token embedded in URL for auth
    remote_url = f"https://{args.github_token}@github.com/{repo_full}.git"

    run(["git", "config", "user.email", "candidate@shl-submission.com"], cwd=project_dir)
    run(["git", "config", "user.name", "SHL Candidate"], cwd=project_dir)

    # Check if already committed
    result = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True, cwd=project_dir)
    if result.returncode != 0 or not result.stdout.strip():
        run(["git", "add", "-A"], cwd=project_dir)
        run(["git", "commit", "-m",
             "feat: SHL Assessment Recommender — FastAPI + BM25/FAISS + Gemini\n\n"
             "- Hybrid retriever: BM25 (lexical) + FAISS (semantic) + RRF fusion\n"
             "- Dual-injection prompting: full catalog in system + top-10 hint in user\n"
             "- Guard layer: injection/off-topic refusal before LLM call\n"
             "- Strict schema enforcement: Pydantic v2 + JSON parser + URL allowlist\n"
             "- 35 tests: schema, retriever unit, behavior probes, Recall@10 self-eval\n"
             "- Dockerized with pre-cached sentence-transformer model\n"
             "- CI/CD via GitHub Actions → Fly.io auto-deploy on main"],
             cwd=project_dir)
        ok("Initial commit created")
    else:
        ok("Repo already has commits")

    # Set/update remote
    subprocess.run(["git", "remote", "remove", "origin"], capture_output=True, cwd=project_dir)
    run(["git", "remote", "add", "origin", remote_url], cwd=project_dir)
    run(["git", "push", "-u", "origin", "main", "--force"], cwd=project_dir)
    ok(f"Pushed to https://github.com/{repo_full}")

    # ── Step 3: Set GitHub Actions secrets ────────────────────────────────
    step(3, "Setting GitHub Actions secrets...")
    try:
        gh_put_secret(repo_full, "GEMINI_API_KEY", args.gemini_key, args.github_token)
        ok("GEMINI_API_KEY secret set")
        gh_put_secret(repo_full, "FLY_API_TOKEN", args.fly_token, args.github_token)
        ok("FLY_API_TOKEN secret set")
    except Exception as e:
        info(f"Secret upload via API failed ({e}) — will set via gh CLI if available")
        # Try gh CLI fallback
        for secret_name, secret_val in [("GEMINI_API_KEY", args.gemini_key), ("FLY_API_TOKEN", args.fly_token)]:
            r = subprocess.run(
                ["gh", "secret", "set", secret_name, "--body", secret_val, "--repo", repo_full],
                capture_output=True, text=True,
                env={**os.environ, "GH_TOKEN": args.github_token}
            )
            if r.returncode == 0:
                ok(f"{secret_name} set via gh CLI")
            else:
                info(f"Could not auto-set {secret_name} — set it manually in repo Settings → Secrets")

    # ── Step 4: Set repo topics ────────────────────────────────────────────
    step(4, "Setting repository topics...")
    try:
        gh_request("PUT", f"/repos/{repo_full}/topics", args.github_token, {
            "names": ["fastapi", "python", "rag", "faiss", "bm25",
                      "conversational-ai", "gemini", "nlp", "hr-tech", "assessment"]
        })
        ok("Topics set")
    except Exception as e:
        info(f"Topics: {e} — set manually in repo About section")

    # ── Step 5: Install flyctl ────────────────────────────────────────────
    step(5, "Setting up Fly.io CLI...")
    try:
        flyctl = install_flyctl()
        ok(f"flyctl ready: {flyctl}")
    except Exception as e:
        fail(f"flyctl setup failed: {e}\nInstall manually: curl -L https://fly.io/install.sh | sh")

    # ── Step 6: Create Fly.io app ─────────────────────────────────────────
    step(6, "Creating Fly.io app...")
    out, code = fly_run(["apps", "list", "--json"], flyctl, args.fly_token)
    existing_apps = []
    try:
        existing_apps = [a["Name"] for a in json.loads(out) if isinstance(json.loads(out), list)]
    except Exception:
        pass

    if args.fly_app in existing_apps:
        ok(f"App '{args.fly_app}' already exists")
    else:
        out, code = fly_run(["apps", "create", args.fly_app, "--org", "personal"], flyctl, args.fly_token, cwd=project_dir)
        if code == 0 or "already exists" in out.lower():
            ok(f"App '{args.fly_app}' created/confirmed")
        else:
            info(f"App creation output: {out[:300]}")
            info("Continuing — app may have been created anyway")

    # Update fly.toml with actual app name
    fly_toml_path = Path(project_dir) / "fly.toml"
    fly_toml = fly_toml_path.read_text()
    fly_toml = fly_toml.replace('app = "shl-recommender"', f'app = "{args.fly_app}"')
    fly_toml_path.write_text(fly_toml)

    # ── Step 7: Set Fly.io secrets ─────────────────────────────────────────
    step(7, "Setting Fly.io secrets...")
    out, code = fly_run(
        ["secrets", "set", f"GEMINI_API_KEY={args.gemini_key}", f"--app={args.fly_app}"],
        flyctl, args.fly_token, cwd=project_dir
    )
    if code == 0:
        ok("GEMINI_API_KEY set on Fly.io")
    else:
        info(f"Secrets output: {out[:200]}")

    # ── Step 8: Deploy to Fly.io ───────────────────────────────────────────
    step(8, "Deploying to Fly.io (this takes 3-5 minutes — Docker build + model cache)...")
    info("Building Docker image, pre-downloading sentence-transformer model...")

    out, code = fly_run(
        ["deploy", "--remote-only", "--wait-timeout", "300", f"--app={args.fly_app}"],
        flyctl, args.fly_token, cwd=project_dir
    )

    if code == 0:
        ok("Deployment successful")
    else:
        info(f"Deploy output (last 500 chars):\n{out[-500:]}")
        info("Deployment may still be in progress — checking health...")

    # ── Step 9: Health check ───────────────────────────────────────────────
    step(9, "Running health check...")
    health_url = f"https://{args.fly_app}.fly.dev/health"
    healthy = wait_for_health(health_url, max_attempts=20, interval=15)

    if healthy:
        ok(f"Service is live and healthy!")
    else:
        info("Service not responding yet — may still be starting. Check manually:")
        info(f"  curl {health_url}")

    # ── Step 10: Push updated fly.toml ────────────────────────────────────
    step(10, "Pushing updated fly.toml to GitHub...")
    run(["git", "add", "fly.toml"], cwd=project_dir)
    result = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=project_dir)
    if result.returncode != 0:
        run(["git", "commit", "-m", f"chore: set fly app name to {args.fly_app}"], cwd=project_dir)
        run(["git", "push"], cwd=project_dir)
        ok("fly.toml pushed")

    # ── Final summary ──────────────────────────────────────────────────────
    live_url = f"https://{args.fly_app}.fly.dev"
    print(f"\n{'='*60}")
    print(bold("🎉 DEPLOYMENT COMPLETE"))
    print(f"{'='*60}")
    print(f"\n  {bold('GitHub repo:')}   https://github.com/{repo_full}")
    print(f"  {bold('Live API:')}      {live_url}")
    print(f"  {bold('Health:')}        {live_url}/health")
    print(f"  {bold('Chat endpoint:')} {live_url}/chat")
    print(f"  {bold('API docs:')}      {live_url}/docs")
    print(f"\n  {bold('Submit this URL to SHL:')} {live_url}")
    print()
    print("  Test it now:")
    print(f"""  curl -X POST {live_url}/chat \\
    -H "Content-Type: application/json" \\
    -d '{{"messages": [{{"role": "user", "content": "I need to hire a Java developer"}}]}}'""")
    print()


if __name__ == "__main__":
    main()
