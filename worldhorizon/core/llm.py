"""Avis d'expert rédigé par un modèle de langage (optionnel).

Fournisseurs pris en charge (configurés dans .streamlit/secrets.toml ou variables d'environnement) :
  ANTHROPIC_API_KEY  (+ ANTHROPIC_MODEL, défaut claude-sonnet-5-5)
  MISTRAL_API_KEY    (+ MISTRAL_MODEL, défaut mistral-large-latest)
  OLLAMA_URL         (+ OLLAMA_MODEL, défaut mistral) — modèle local, ex. http://localhost:11434

Le modèle reçoit UNIQUEMENT la fiche chiffrée produite par les agents, les titres de presse
collectés, le résumé Wikipédia et la liste des références d'experts. Il lui est demandé de ne
rien inventer et de citer les sources fournies.
"""
from __future__ import annotations

import os

import requests

SYSTEM = (
    "Tu es un analyste pays senior (risque souverain, développement, géopolitique). "
    "Tu rédiges en français une analyse rigoureuse, nuancée et sourcée. Règles : n'utilise QUE les faits "
    "fournis ; signale quand une donnée est ancienne ou manquante ; cite la source entre parenthèses ; "
    "ne fais pas de prédiction certaine ; distingue faits, interprétations et scénarios."
)

PROMPT = """Rédige l'analyse complète de la situation de {nom}, sur tous les aspects, structurée ainsi :
1. Synthèse exécutive (5-6 phrases)
2. Situation par grand domaine : santé, éducation (dont PISA), économie et marchés financiers, environnement des affaires,
   environnement et agriculture, gouvernance et démocratie, justice et sentiment d'impunité, corruption, sécurité,
   social et bien-être, innovation, tourisme — pour chacun : constat chiffré, tendance longue, enjeu principal
3. Ce que disent les experts : rattache les constats aux grilles de lecture des institutions de référence listées
   (FMI, Banque mondiale, OCDE, V-Dem, Transparency International, GIEC…) sans leur prêter de citations inventées
4. Politique et élections : risque politique, déterminants du vote, scénarios
5. Perspectives 2050 et 2100 : scénarios pessimiste / central / optimiste, conditions d'un avenir brillant
6. Points de vigilance

FICHE DE DONNÉES (agents WorldHorizon) :
{fiche}

ACTUALITÉS RÉCENTES (titres de presse) :
{actus}

CONTEXTE ENCYCLOPÉDIQUE :
{wiki}

RÉFÉRENCES D'EXPERTS À CONSULTER :
{experts}
"""


def _secret(name: str, default: str | None = None) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def provider() -> str | None:
    if _secret("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _secret("MISTRAL_API_KEY"):
        return "mistral"
    if _secret("OLLAMA_URL"):
        return "ollama"
    return None


def generate(prompt: str, max_tokens: int = 3500) -> str:
    p = provider()
    if p == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=180, headers={
            "x-api-key": _secret("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": _secret("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": max_tokens,
                  "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    if p == "mistral":
        r = requests.post("https://api.mistral.ai/v1/chat/completions", timeout=180, headers={
            "Authorization": f"Bearer {_secret('MISTRAL_API_KEY')}"},
            json={"model": _secret("MISTRAL_MODEL", "mistral-large-latest"), "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if p == "ollama":
        r = requests.post(_secret("OLLAMA_URL").rstrip("/") + "/api/chat", timeout=600,
                          json={"model": _secret("OLLAMA_MODEL", "mistral"), "stream": False,
                                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["message"]["content"]
    raise RuntimeError("Aucun modèle de langage configuré")


def expert_analysis(an: dict, news: dict | None, wiki: dict | None) -> str:
    from .narrative import fact_sheet
    actus = "\n".join(f"- {t}" for t in (news or {}).get("table", {}).get("titre", [])[:25]) if news and news.get("ok") else "(aucune)"
    experts = "\n".join(f"- {n} : {u}" for p in an["piliers"].values() for n, u in p["experts"])
    prompt = PROMPT.format(nom=an["nom"], fiche=fact_sheet(an), actus=actus,
                           wiki=(wiki or {}).get("texte", "(indisponible)"), experts=experts)
    return generate(prompt)
