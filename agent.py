# -*- coding: utf-8 -*-

import chromadb
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
from langchain_core.runnables import RunnablePassthrough

# --- 1. Configuration ---

# Le nom de la collection que nous avons créée dans le script précédent.
COLLECTION_NAME = "recettes"
# Le modèle d'embedding (doit être le même que celui utilisé pour la création).
EMBEDDING_MODEL = "mxbai-embed-large"
# Le modèle de LLM à utiliser pour la génération de la réponse.
LLM_MODEL = "mistral"

# --- 2. Initialisation des composants LangChain ---

print("Initialisation des composants LangChain...")

# Initialise le client Ollama pour les embeddings
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialise le client ChromaDB pour se connecter à la base de données existante.
# Le chemin doit correspondre à l'endroit où la DB a été créée par module5_creation_db.py
# (qui est dans le même dossier 'code')
vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./chroma_db"),
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)

# Crée un retriever à partir du vectorstore.
# Le retriever est responsable de la recherche des documents pertinents.
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Récupère les 3 chunks les plus pertinents

# Initialise le modèle de chat Ollama
llm = ChatOllama(model=LLM_MODEL)
pizza = [
        "documents/marco-fuso-recipe-booklet---final.pdf",
        "documents/pizza-booklet-french-68623de5495cb223587169.pdf",
        "documents/Recette-pizza-au-fromage.pdf",
        ]
allergene = [
             "documents/Recette-pizza-au-fromage.pdf",
             "documents/Tableau-des-allergenes.pdf"
             ]
# --- 3. Définition du prompt RAG ---

# Le template du prompt pour le LLM.
# Il inclut le contexte récupéré et la question de l'utilisateur.
template2 = """En tant que Polo, le meilleur pote de Marco et tu travailles dans la pizzeria de Marco Fuso, réponds en te basant uniquement sur le context :
{context}

Question: {question}
"""
template = """
Tu es un expert en cuisine italienne spécialisé dans les pizzas de Marco Fuso. 
Tu dois répondre uniquement en utilisant les informations suivantes extraites de documents :

{context}

Ne réponds pas si la réponse ne figure pas dans ces documents. Si tu ne sais pas, dis : 
"Je ne trouve pas cette information dans les documents disponibles."

Question : {question}
"""

prompt = ChatPromptTemplate.from_template(template2)

# --- 4. Construction de la chaîne RAG avec LangChain Expression Language (LCEL) ---

# La chaîne RAG est construite en utilisant LCEL pour une meilleure lisibilité et modularité.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} # Étape de recherche (Retrieval)
    | prompt                                                  # Étape d'augmentation (Augmented)
    | llm                                                     # Étape de génération (Generation)
    | StrOutputParser()                                       # Parse la sortie du LLM en chaîne de caractères
)

# --- 5. Boucle d'interaction ---

# if __name__ == "__main__":
#     print("\n--- Chatbot RAG avec LangChain ---")
#     print("Posez des questions sur le document. Tapez 'exit' pour quitter.")

#     while True:
#         user_question = input("\nVous: ")
#         if user_question.lower() == "exit":
#             break

#         print("Assistant: ...")
#         # Invoque la chaîne RAG avec la question de l'utilisateur
#         answer = rag_chain.invoke(user_question)
#         print(f"\rAssistant: {answer}")

# --- 5. Fonction de réponse pour Gradio ---
def respond_to_question(question):
    return rag_chain.invoke(question)

# --- 6. Interface Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("## 🍕 Chatbot Polo – Trouvez vos allergènes et recettes Marco Fuso")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(
                placeholder="Posez une question sur les recettes ou les allergènes...",
                label="Votre question"
            )
            submit_btn = gr.Button("Envoyer")

        def handle_user_input(message, history):
            response = respond_to_question(message)
            history = history + [(message, response)]
            return history, ""

        submit_btn.click(handle_user_input, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
        user_input.submit(handle_user_input, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

# Lancer l'interface
demo.launch()
