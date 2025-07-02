# -*- coding: utf-8 -*-
import chromadb
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# Nouveaux imports pour retrievers avancés
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# --- 1. Configuration ---
COLLECTION_NAME = "recettes"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"

# --- 2. Initialisation ---
print("Initialisation des composants LangChain...")

ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./chroma_db"),
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)

# --- 3. Retriever avancé : MultiQuery + Compression ---

# Étape 1 : Reformulation de la requête
reformulation_prompt = PromptTemplate.from_template(
    "Tu es un assistant intelligent. Reformule la question suivante en trois variantes différentes : {question}"
)
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOllama(model=LLM_MODEL),
    prompt=reformulation_prompt  # Tu peux l’enlever pour utiliser le prompt par défaut
)

# Étape 2 : Compression contextuelle
compressor = LLMChainExtractor.from_llm(ChatOllama(model=LLM_MODEL))
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever
)

# --- 4. Prompt RAG ---

template = """Tu es Polo, le meilleur pote de Marco et tu travailles dans la pizzeria de Marco Fuso, réponds :
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 5. RAG Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOllama(model=LLM_MODEL)
    | StrOutputParser()
)

# --- 6. Gradio UI ---
def respond_to_question(question):
    return rag_chain.invoke(question)

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

demo.launch()
