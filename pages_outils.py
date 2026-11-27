"""Outils communs aux pages Projet et Étapes : lecture du vrai code source (toujours à jour) et utilitaires d'affichage."""
import ast
from pathlib import Path

import streamlit as st

import ui

ROOT = Path(__file__).parent


def lire(fichier: str) -> str:
    return (ROOT / fichier).read_text(encoding="utf-8")


def nb_lignes(fichier: str) -> int:
    try:
        return len(lire(fichier).splitlines())
    except OSError:
        return 0


def fonction(fichier: str, nom: str):
    """(source, ligne de début, ligne de fin) d'une fonction de premier niveau ; None si introuvable."""
    try:
        t = lire(fichier)
        for n in ast.parse(t).body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == nom:
                return ast.get_source_segment(t, n), n.lineno, n.end_lineno
    except (OSError, SyntaxError):
        pass
    return None


def montrer_code(fichier: str, nom: str, role: str = ""):
    """Légende (fichier, lignes) puis le code réel de la fonction, avec coloration Python."""
    f = fonction(fichier, nom)
    if f is None:
        return
    src, a, b = f
    ui.cap(f"{nom}()", f"{fichier}, lignes {a} à {b}" + (f" : {role}" if role else ""))
    st.code(src, language="python")


def commande(texte: str):
    st.code(texte, language="bash")
