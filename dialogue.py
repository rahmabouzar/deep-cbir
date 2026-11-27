"""Boîte de dialogue native de choix d'un fichier image (tkinter), lancée dans un processus séparé.

Le serveur Streamlit s'exécute sur votre machine : la fenêtre « Ouvrir » de votre système s'affiche donc devant le navigateur.
Un processus séparé évite les conflits entre tkinter et les threads de Streamlit.
"""
import subprocess
import sys

_CODE = """
import sys
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
chemin = filedialog.askopenfilename(
    parent=root, title="Choisir l'image requête", initialdir=sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None,
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"), ("Tous les fichiers", "*.*")])
print(chemin)
"""


def choisir_image(dossier: str = "") -> "str | None":
    """Ouvre la fenêtre de choix et retourne le chemin choisi.
    "" si l'utilisateur annule ; None si la fenêtre ne peut pas s'ouvrir (pas d'écran, tkinter absent, serveur distant)."""
    try:
        r = subprocess.run([sys.executable, "-c", _CODE, dossier or ""], capture_output=True, text=True, timeout=900)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    lignes = r.stdout.strip().splitlines()
    return lignes[-1].strip() if lignes else ""
