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

Dive deeper with this series of blogs exploring the theory and code behind each advanced technique:

### 📦 Indexing

- **Part 1: Splitting Strategy**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-f9c82290b0d1)

- **Part 2: Embedding Strategy**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-part-2-6dbcdb5fbe59)

### ❓ Question Transformation

- **Part 3: Rewrite-Retrieve-Read**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-7d71d25323c5)

- **Part 4: Generating Multiple Queries**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-part-4-d433c103d7ef)

- **Part 5: Step-Back Questioning**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-part-5-d362271189e8)

- **Part 6: Hypothetical Document Embeddings (HyDE)**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-part-6-d572a859a83f)

### 🧠 Self-Querying and SQL

- **Part 7: Metadata Query Enrichment**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-part-7-843ecd3199f0)

- **Part 8: Text-to-SQL (Natural Language to Structured Queries)**  
  [🔗 Read on Medium](https://medium.com/@roberto.g.infante/advanced-rag-techniques-with-langchain-part-8-5c0832da2329)

---

Feel free to ⭐️ this repo if you find it helpful. Contributions are welcome!
