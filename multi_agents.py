from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dataclasses import dataclass
import time

# Config LLM
LLM_MODEL = "llama3:latest"
llm = ChatOllama(model=LLM_MODEL, temperature=0.4)

# Embeddings (exemple)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Se connecter à la base ChromaDB locale ou distante
vectordb = Chroma(persist_directory="./chroma_persist", embedding_function=embeddings)

@dataclass
class AgentMessage:
    sender: str
    content: str
    timestamp: float
    message_type: str  # "task", "result", "feedback"

class BaseAgent:
    def __init__(self, name, role, specialty):
        self.name = name
        self.role = role
        self.specialty = specialty
        self.llm = llm
        self.memory = []

    def add_to_memory(self, message: AgentMessage):
        self.memory.append(message)

    def process(self, **kwargs):
        raise NotImplementedError

class RecipeParserAgent(BaseAgent):
    def __init__(self):
        super().__init__("Décrypteur", "Parser", "Extraction d'information")

    def process(self, recipe_text):
        prompt = f"""
Tu es un agent expert en lecture de recettes. Analyse le texte suivant :

{recipe_text}

Ta tâche est de ressortir :
1. Le nom de la recette
2. Les ingrédients sous forme de liste
3. Les étapes de préparation

Format :
## Nom de la recette:
...

## Ingrédients:
- ...
- ...

## Étapes de préparation:
1. ...
2. ...
"""
        return self.llm.invoke(prompt).content

class AllergenDetectorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Allergologue", "AllergenDetector", "Détection des allergènes")

    def process(self, ingredients_text):
        prompt = f"""
Voici une liste d'ingrédients :

{ingredients_text}

Identifie les allergènes potentiels (gluten, lait, œuf, arachide, fruits à coque, poisson, soja, etc.).
Réponds au format :
## Allergènes détectés :
- ...
- ...
"""
        return self.llm.invoke(prompt).content

class NoiseFilterAgent(BaseAgent):
    def __init__(self):
        super().__init__("Filtre", "NoiseFilter", "Suppression du texte inutile")

    def process(self, raw_text):
        prompt = f"""
Voici un texte issu d'un site de recettes :

{raw_text}

Supprime tout le contenu inutile : histoires personnelles, pubs, anecdotes. Ne garde que les informations utiles à la recette.

Réponds uniquement avec le texte nettoyé.
"""
        return self.llm.invoke(prompt).content

class QAAgent(BaseAgent):
    def __init__(self):
        super().__init__("Répondeur", "QuestionAnswerer", "Réponse aux questions")

    def process(self, question, recipe_clean):
        prompt = f"""
Voici une recette nettoyée :

{recipe_clean}

QUESTION : {question}

Réponds de manière claire, concise et utile.
"""
        return self.llm.invoke(prompt).content

class RecipeOrchestrator:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.agents = {
            "parser": RecipeParserAgent(),
            "allergen": AllergenDetectorAgent(),
            "filter": NoiseFilterAgent(),
            "qa": QAAgent(),
        }
        self.log = []

    def log_message(self, sender, content, msg_type="result"):
        msg = AgentMessage(sender, content, time.time(), msg_type)
        self.log.append(msg)
        for agent in self.agents.values():
            agent.add_to_memory(msg)

    def retrieve_relevant_chunks(self, query, top_k=5):
        results = self.vectordb.similarity_search(query, k=top_k)
        combined_text = "\n\n".join([doc.page_content for doc in results])
        return combined_text

    def run_pipeline(self, query, question):
        results = {}

        print("🔍 Recherche des chunks pertinents dans ChromaDB...")
        relevant_chunks = self.retrieve_relevant_chunks(query)
        self.log_message("ChromaDB", relevant_chunks)

        print("🔍 Nettoyage du texte...")
        clean = self.agents["filter"].process(relevant_chunks)
        self.log_message("Filtre", clean)
        results["cleaned_text"] = clean

        print("📋 Parsing de la recette...")
        parsed = self.agents["parser"].process(clean)
        self.log_message("Décrypteur", parsed)
        results["parsed_recipe"] = parsed

        print("⚠️ Détection des allergènes...")
        allergens = self.agents["allergen"].process(parsed)
        self.log_message("Allergologue", allergens)
        results["allergens"] = allergens

        print("❓ Réponse à la question...")
        answer = self.agents["qa"].process(question, parsed)
        self.log_message("Répondeur", answer)
        results["answer"] = answer

        return results

    def display_results(self, results):
        print("\n" + "="*80)
        print("📝 RÉSULTATS DU SYSTÈME MULTI-AGENTS")
        print("="*80)
        for key, val in results.items():
            print(f"\n## {key.upper()}\n{val}\n")

def main():
    orchestrator = RecipeOrchestrator(vectordb)

    print("Bienvenue dans le système multi-agents avec ChromaDB")
    query = input("Tape ta requête pour récupérer des chunks pertinents (ex: 'recette tarte aux pommes'):\n> ")
    question = input("Pose ta question sur la recette:\n> ")

    results = orchestrator.run_pipeline(query, question)
    orchestrator.display_results(results)
if __name__ == "__main__":
    main()