"""Sauvegarde des CSV mis à jour dans le dépôt GitHub (un seul commit par mise à jour).

Sur Streamlit Community Cloud, le disque est éphémère : sans ce commit, les données collectées
par le bouton « Mettre à jour » seraient perdues au prochain redémarrage.
Configuration (.streamlit/secrets.toml) :
    GITHUB_TOKEN  = "github_pat_…"   # jeton à portée limitée au dépôt, droit Contents: Read and write
    GITHUB_REPO   = "utilisateur/worldhorizon"
    GITHUB_BRANCH = "main"
"""
from __future__ import annotations

import base64
import datetime as dt

import requests

from .llm import _secret
from .paths import ROOT

API = "https://api.github.com"


def configured() -> bool:
    return bool(_secret("GITHUB_TOKEN") and _secret("GITHUB_REPO"))


def commit_data(message: str | None = None, folders=("data/agents", "data/scores", "data/ml", "data/elections",
                                                      "data/markets", "data/news", "data/ref")) -> str:
    token, repo, branch = _secret("GITHUB_TOKEN"), _secret("GITHUB_REPO"), _secret("GITHUB_BRANCH", "main")
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    ref = requests.get(f"{API}/repos/{repo}/git/ref/heads/{branch}", headers=h, timeout=30)
    ref.raise_for_status()
    base_sha = ref.json()["object"]["sha"]
    base_tree = requests.get(f"{API}/repos/{repo}/git/commits/{base_sha}", headers=h, timeout=30).json()["tree"]["sha"]
    tree = []
    for folder in folders:
        for p in sorted((ROOT / folder).glob("*.csv")):
            blob = requests.post(f"{API}/repos/{repo}/git/blobs", headers=h, timeout=120,
                                 json={"content": base64.b64encode(p.read_bytes()).decode(), "encoding": "base64"})
            blob.raise_for_status()
            tree.append({"path": str(p.relative_to(ROOT)), "mode": "100644", "type": "blob", "sha": blob.json()["sha"]})
    for p in sorted((ROOT / "models").glob("*.joblib")):
        blob = requests.post(f"{API}/repos/{repo}/git/blobs", headers=h, timeout=120,
                             json={"content": base64.b64encode(p.read_bytes()).decode(), "encoding": "base64"})
        blob.raise_for_status()
        tree.append({"path": str(p.relative_to(ROOT)), "mode": "100644", "type": "blob", "sha": blob.json()["sha"]})
    nt = requests.post(f"{API}/repos/{repo}/git/trees", headers=h, timeout=60, json={"base_tree": base_tree, "tree": tree})
    nt.raise_for_status()
    msg = message or f"Mise à jour des données WorldHorizon — {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC"
    cm = requests.post(f"{API}/repos/{repo}/git/commits", headers=h, timeout=60,
                       json={"message": msg, "tree": nt.json()["sha"], "parents": [base_sha]})
    cm.raise_for_status()
    up = requests.patch(f"{API}/repos/{repo}/git/refs/heads/{branch}", headers=h, timeout=30, json={"sha": cm.json()["sha"]})
    up.raise_for_status()
    return cm.json()["html_url"]
