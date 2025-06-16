# 🧠 ADVANCED-RAG

This repository demonstrates **Advanced Retrieval-Augmented Generation (RAG)** techniques implemented using **[LangGraph](https://github.com/langchain-ai/langgraph)** — a powerful graph-based orchestration library built on top of LangChain. The goal is to prototype, experiment, and deploy modular, agentic, and memory-enhanced RAG pipelines for complex LLM workflows.

---

## 🏗️ Tech Stack

- **LangGraph** – Core orchestration engine for building dynamic, agentic LLM pipelines.
- **LangChain** – Used for retrieval, embeddings, and chain management.
- **OpenAI / other LLM providers** – As the backend for generation, memory, and planning.

---

## 📁 Folder Structure

### 1. **LangGraph**
LangGraph-driven experiments focusing on agent design, interaction, and memory.

- `chatbot/` – Conversational agents with memory and basic dialogue handling.
- `human-in-the-loop/` – Combines human judgment into the LLM flow for decision-critical tasks.
- `reflection-agent/` – Implements agents that reflect on outcomes for self-improvement.
- `reflexion-agent/` – Inspired by the Reflexion paper, enabling agentic self-feedback loops.
- `state/` – Global shared state and utilities for managing LangGraph workflows.
- `screenshot/` – Visualizations or UI captures for demo purposes.

---

### 2. **RAG_advanced_techniques**
Advanced RAG enhancements for indexing, retrieval, and transformation:

#### 📌 `advanced-indexing/`
Strategies for improving document chunking, embedding, and retriever design.
- `embedding-Granular-Chunk-Expansion.py`
- `embedding-hypothetical-question.py`
- `embedding-stratergy-*`
- `splitting-stratergy.py`

#### 🧱 `foundational/`
Baseline implementation of a simple RAG pipeline using LangGraph.
- `simple_rag.py`

#### 🧠 `query-construction/`
Reformulates queries to better align with vector store retrieval.
- `self-metadata-query.py`

#### 🔁 `question-transformation/`
Improves query relevance using hypothetical document generation, multi-query, and rephrasing techniques.
- `HyDE-hypothetical-Document-embeddings.py`
- `rewrite-retrieve-read*.py`
- `step-back-question-transformation.py`

#### 📂 `data/`
Local or experimental data used for pipeline validation.

---

### 3. **Types_of_RAG_implementation_using_LangGraph**
This section showcases **LangGraph-powered agentic RAG variants**:

- `adaptive-RAG/` – Dynamically alters behavior (e.g. retriever type or embedding strategy) based on user context or query.
- `agentic-RAG/` – Full-fledged LangGraph agents with memory, planning, and tools.
- `corrective_RAG/` – Implements mechanisms for self-correction using LangGraph state updates.
- `self-reflection-RAG/` – Enables introspective agents that learn from prior mistakes using feedback loops.

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/your-username/advanced-rag.git
cd advanced-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Goals of This Repo

1. Demonstrate real-world use cases of LangGraph in RAG scenarios.
2. Implement modular and flexible graph-based architectures.
3. Integrate agent memory, self-reflection, and adaptive reasoning in LLM workflows.
4. Push the boundary of how LLMs retrieve, reason, and improve iteratively.

---

## 📝 Blog Series – Advanced RAG Techniques

Read the full blog series explaining the ideas behind each RAG enhancement in detail:

### 🔄 Query Translation
- **Advanced Query Translation Techniques in RAG Systems**  
  [🔗 Read here](https://medium.com/@gauravbansalutd/advanced-query-translation-techniques-in-retrieval-augmented-generation-rag-systems-0fad5ad6f500)

### 🚦 Routing
- **Elevating LLM Interactions with Intelligent Routing**  
  [🔗 Read here](https://medium.com/@gauravbansalutd/advanced-rag-techniques-elevating-llm-interactions-with-intelligent-routing-d8c2f55a017b)

### 🧠 Query Construction
- **Tailoring Queries for Different Data Sources**  
  [🔗 Read here](https://medium.com/@gauravbansalutd/query-construction-in-rag-systems-tailoring-queries-for-different-data-sources-b736f18e6f52)

### 🗃️ Indexing
- **Part I: Beyond Basic Chunking**  
  [🔗 Read here](https://medium.com/@gauravbansalutd/advanced-indexing-techniques-in-rag-systems-beyond-basic-chunking-ea6a84e4627c)

- **Part II: Advanced Embedding Strategies**  
  [🔗 Read here](https://medium.com/@gauravbansalutd/advanced-indexing-techniques-in-rag-systems-beyond-basic-chunking-part-ii-0d5e190c7a57)

### 📡 Retrieval
- **CRAG, Re-ranking, and Active Retrieval**  
  [🔗 Read here](https://medium.com/@gauravbansalutd/advanced-retrieval-techniques-in-rag-systems-crag-re-ranking-and-active-retrieval-a4796a7a5f4f)

---

Feel free to ⭐️ this repo if you find it helpful. Contributions are welcome!
