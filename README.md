# pizzIA
# 🍕 Chatbot Polo – Recettes & Allergènes de la Pizzeria Marco Fuso

**Chatbot Polo** est un assistant intelligent développé avec **LangChain**, **Ollama**, **ChromaDB** et **Gradio**, capable de répondre à des questions sur les **recettes de pizzas** et les **allergènes alimentaires** à partir de documents PDF.

Ce projet met en œuvre une architecture **RAG (Retrieval-Augmented Generation)** pour retrouver les informations pertinentes dans des documents vectorisés, puis générer une réponse contextuelle.

---

## 🔍 Fonctionnalités

- Recherche sémantique dans une base de connaissances vectorisée (PDFs de recettes et allergènes)
- Génération de réponses avec un LLM local via Ollama (`mistral`)
- Interface utilisateur conviviale avec **Gradio**
- Persona personnalisé : Polo, le pote de Marco Fuso 🍕

---

## 📁 Contenu

### Structure des fichiers :

```
.
├── app_chatbot.py              # Script principal de l'application
├── chroma_db/                  # Base de données vectorielle persistée
├── documents/                  # Fichiers PDF de recettes et allergènes
│   ├── marco-fuso-recipe-booklet---final.pdf
│   ├── pizza-booklet-french-68623de5495cb223587169.pdf
│   ├── Recette-pizza-au-fromage.pdf
│   └── Tableau-des-allergenes.pdf
└── README.md                   # Ce fichier
```

---

## ⚙️ Technologies utilisées

- **LangChain** : Orchestration RAG
- **Ollama** : Modèles locaux (`mistral` pour le LLM, `mxbai-embed-large` pour les embeddings)
- **ChromaDB** : Base vectorielle persistée
- **Gradio** : Interface utilisateur simple et efficace

---

## 🚀 Lancement

### 1. Pré-requis

- Python 3.10 ou supérieur
- [Ollama installé et lancé](https://ollama.com/)
- Modèles Ollama suivants téléchargés :
  - `mistral`
  - `mxbai-embed-large`
- Base Chroma existante (générée avec un script tel que `module5_creation_db.py`)
- Les documents PDF dans le dossier `documents/`

### 2. Installation

```bash
pip install -r requirements.txt
```

Exemple de `requirements.txt` :

```txt
requests
langchain
langchain-community
langchain-core
langchain-ollama
feedparser
uvicorn
fastapi
python-dotenv
numpy
matplotlib
scikit-learn
chromadb
PyPDF2
langchain-openai 
sqlalchemy 
psycopg2-binary
crewai
crewai_tools
gradio
```

### 3. Exécution

```bash
python agent.py
```

L'application Gradio se lancera automatiquement dans votre navigateur.

---

## 💬 Exemple de questions à poser

- _"Quels sont les ingrédients de la pizza au fromage ?"_
- _"Cette pizza contient-elle des œufs ou du gluten ?"_
- _"Donne-moi la recette de la pâte de Marco Fuso."_

---

## ✍️ Personnalisation

Deux prompts sont définis dans le script :

- `template` : classique, centré sur les recettes
- `template2` : personnage Polo, style amical et conversationnel

Par défaut, `template2` est utilisé :
```python
prompt = ChatPromptTemplate.from_template(template2)
```

Tu peux basculer vers le style classique en modifiant cette ligne.

---

## 🙋 À propos

Projet développé par **Ludivine Raby**  
Objectif : combiner IA générative et recherche documentaire vectorielle pour un assistant gastronomique intelligent 🍕🤖