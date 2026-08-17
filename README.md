# Enterprise RAG Agent 🚀

A complete, production-ready Retrieval-Augmented Generation (RAG) pipeline built to run efficiently on limited hardware (like Google Colab's T4 GPU). 

This project evolves from a basic document retriever into a fine-tuned, multimodal AI agent capable of processing text, voice, and images.

## 🏗️ Project Architecture
The repository is structured to follow a 5-step evolution:

1. **Phase 1 (Production RAG):** Ingesting PDFs, chunking, semantic embeddings (`bge-large`), and vector storage (`ChromaDB`).
2. **Phase 2 (Local LLMs):** Replacing cloud APIs with local 4-bit quantized models (`Llama-3 8B`) to run completely offline.
3. **Phase 3 (Observability):** Integrating telemetry (`Langfuse`) and RAG metrics (`Ragas`) to monitor hallucination rates and latency.
4. **Phase 4 (Fine-Tuning):** Using QLoRA (`PEFT`) to train the base model on domain-specific datasets (e.g., Financial QA).
5. **Phase 5 (Multimodal):** Adding Voice-to-Text (`Whisper`), Vision (`Llava`), and a web UI (`Gradio`).

## 📂 Repository Structure
```text
enterprise-rag/
├── data/               # Raw PDFs and HuggingFace datasets
├── notebooks/          # Google Colab execution notebooks
├── rag_core/           # Python modules for RAG pipeline
├── observability/      # Telemetry and evaluation scripts
├── fine_tuning/        # QLoRA training scripts
├── multimodal/         # Audio and Vision integration
└── app/                # Gradio UI interfaces
```

## 🚀 Getting Started (Colab)
1. Open Google Colab and set the Runtime to **T4 GPU**.
2. Clone this repository or upload the notebooks from the `notebooks/` folder.
3. Run `01_Production_RAG.ipynb` to build the foundational Vector Database.

## 🛠️ Tech Stack
* **Orchestration:** Langchain
* **Vector DB:** ChromaDB
* **Embeddings:** Sentence-Transformers (BGE)
* **LLM Inference:** HuggingFace Transformers, BitsAndBytes
* **Fine-Tuning:** TRL, PEFT (LoRA)
* **UI:** Gradio
