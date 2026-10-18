from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REF = DATA / "ref"
AGENTS_DIR = DATA / "agents"
SCORES = DATA / "scores"
ML_DIR = DATA / "ml"
ELECTIONS = DATA / "elections"
MARKETS = DATA / "markets"
NEWS = DATA / "news"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"

for p in (DATA, REF, AGENTS_DIR, SCORES, ML_DIR, ELECTIONS, MARKETS, NEWS, CACHE, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
