# -*- coding: utf-8 -*-

import os

import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"
# --- 1. Configuration ---

# Nom du fichier PDF à traiter
PDF_FILE =  ["documents/Allergenes-FPO-Enseigne-MAJ-Janv-2024.pdf",
             "documents/Allergenes-Pizza-Rhuys.pdf",
             "documents/marco-fuso-recipe-booklet---final.pdf",
             "documents/pizza-booklet-french-68623de5495cb223587169.pdf",
             "documents/Recette-pizza-au-fromage.pdf",
             "documents/Tableau-des-allergenes.pdf"
]
# Le chemin est relatif au script
# Nom de la collection dans ChromaDB
# Une collection est comme une table dans une base de données traditionnelle.
COLLECTION_NAME = "recettes"
# Nom du modèle d'embedding à utiliser avec Ollama.
# Ce modèle doit être téléchargé au préalable avec `ollama pull mxbai-embed-large`
EMBEDDING_MODEL = "mxbai-embed-large"

# --- 2. Chargement et découpage du document ---


def load_and_chunk_pdf(file_path, chunk_size=500, chunk_overlap=100):
    """
    Charge un fichier PDF, en extrait le texte et le découpe en morceaux (chunks).
    """
    print(f"Chargement du fichier : {file_path}")
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    print(f"Le document contient {len(text)} caractères.")

    print("Découpage du texte en chunks...")
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        # print(text[i : i + chunk_size])
        chunks.append(text[i : i + chunk_size])
    print(f"{len(chunks)} chunks ont été créés.")
    return chunks


# --- 3. Initialisation de ChromaDB et de la fonction d'embedding ---

print("Initialisation de ChromaDB...")
# Crée un client ChromaDB qui stockera les données sur le disque dans le dossier `chroma_db`
client = chromadb.PersistentClient(path="./chroma_db")

print("Initialisation de la fonction d'embedding via Ollama...")
# Crée une fonction d'embedding qui utilise le modèle spécifié via Ollama.
# C'est cette fonction qui sera appelée par ChromaDB pour convertir le texte en vecteurs.
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

# --- 4. Création de la collection et stockage des données ---

print(f"Création ou chargement de la collection : {COLLECTION_NAME}")
# Crée une nouvelle collection (ou la charge si elle existe déjà).
# La fonction d'embedding est passée à la création pour que la collection sache comment traiter les textes.
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=ollama_ef
)

pdf_chunks=0
# Charge et découpe le PDF
for i, _ in enumerate(PDF_FILE):
    pdf_chunks=load_and_chunk_pdf(PDF_FILE[i])
    print("PDF_file", PDF_FILE[i])
    print("Stockage des chunks dans ChromaDB (cela peut prendre un certain temps)...")

    collection.add(documents=pdf_chunks, ids=[f"chunk{PDF_FILE[i]}_{x}" for x in range(len(pdf_chunks))])

    print("\n--- Base de données vectorielle créée avec succès ! ---")
    print(f"Nombre de documents stockés : {collection.count()}")


