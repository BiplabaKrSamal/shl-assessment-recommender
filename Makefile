# ─────────────────────────────────────────────────────────────────────────────
# SHL Assessment Recommender — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install run test eval scrape deploy health check-env

PYTHON   := python3
UVICORN  := uvicorn
PORT     := 8000
APP_URL  := http://localhost:$(PORT)
FLY_APP  := shl-assessment-recommender

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  SHL Assessment Recommender"
	@echo "  ──────────────────────────────────────────"
	@echo "  make install     Install all dependencies"
	@echo "  make run         Start API server (port $(PORT))"
	@echo "  make test        Run full test suite (mocked LLM)"
	@echo "  make test-live   Run tests with live Gemini API"
	@echo "  make eval        Local Recall@10 evaluation"
	@echo "  make eval-live   Evaluation against deployed server"
	@echo "  make scrape      Re-scrape SHL product catalog"
	@echo "  make health      Check server health"
	@echo "  make deploy      Deploy to Fly.io"
	@echo "  make check-env   Validate .env configuration"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Dependencies installed"
	@echo "→ Copy .env.example to .env and add your GEMINI_API_KEY"

check-env:
	@test -f .env || (echo "✗ .env not found — copy .env.example and add your keys" && exit 1)
	@grep -q "GEMINI_API_KEY=." .env || (echo "✗ GEMINI_API_KEY not set in .env" && exit 1)
	@echo "✓ .env looks good"

# ── Development ───────────────────────────────────────────────────────────────
run: check-env
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port $(PORT) --log-level info

run-prod: check-env
	$(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT) --workers 1 --timeout-keep-alive 30

health:
	@curl -sf $(APP_URL)/health | python3 -m json.tool || echo "✗ Server not responding at $(APP_URL)"

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-live:
	pytest tests/ -v --tb=short --live

test-schema:
	@$(PYTHON) -c "\
from app.models import ChatRequest, ChatResponse, Message, Recommendation; \
import json; \
catalog = json.load(open('data/catalog.json')); \
valid = set('ABCDEKPMS'); \
bad = [a['name'] for a in catalog if a['test_type'] not in valid or not a['url'].startswith('https://www.shl.com')]; \
print(f'Catalog: {len(catalog)} items, {len(bad)} errors: {bad[:3]}' if bad else f'✓ Catalog: {len(catalog)} items, all valid')"

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	$(PYTHON) scripts/evaluate.py --local

eval-live:
	$(PYTHON) scripts/evaluate.py --endpoint $(APP_URL)

eval-deployed:
	$(PYTHON) scripts/evaluate.py --endpoint https://$(FLY_APP).fly.dev

# ── Data ──────────────────────────────────────────────────────────────────────
scrape:
	@echo "Scraping SHL catalog (takes ~5 minutes, be polite to shl.com)..."
	$(PYTHON) scripts/scrape_catalog.py
	@echo "✓ Catalog updated at data/catalog.json"

# ── Deployment ────────────────────────────────────────────────────────────────
deploy:
	@test -n "$(FLY_API_TOKEN)" || (echo "✗ Set FLY_API_TOKEN env var" && exit 1)
	flyctl deploy --remote-only --wait-timeout 300 --app $(FLY_APP)
	@echo "✓ Deployed to https://$(FLY_APP).fly.dev"

deploy-check:
	@curl -sf https://$(FLY_APP).fly.dev/health | python3 -m json.tool

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t shl-recommender:latest .

docker-run: check-env
	docker run --rm -p $(PORT):8000 \
		--env-file .env \
		-e CATALOG_PATH=/app/data/catalog.json \
		shl-recommender:latest

# ── Util ──────────────────────────────────────────────────────────────────────
smoke-test:
	@echo "Running smoke test against $(APP_URL)..."
	@curl -sf $(APP_URL)/health > /dev/null && echo "✓ /health OK" || echo "✗ /health FAILED"
	@curl -sf -X POST $(APP_URL)/chat \
		-H "Content-Type: application/json" \
		-d '{"messages":[{"role":"user","content":"I need an assessment"}]}' \
		| python3 -c "import sys,json; d=json.load(sys.stdin); \
		  assert 'reply' in d and 'recommendations' in d and 'end_of_conversation' in d; \
		  print(f'✓ /chat OK — reply={d[\"reply\"][:60]}..., recs={len(d[\"recommendations\"])}')"
