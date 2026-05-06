# 🎭 Character Portrait Generator

A powerful, AI-driven pipeline for generating cinematic character portraits directly from literary works. This application uses Retrieval-Augmented Generation (RAG) to analyze books, identify major characters, and generate context-aware image prompts for local ComfyUI instances.

![Project Preview](https://img.shields.io/badge/Aesthetics-Glassmorphism-blueviolet?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Tech-FastAPI%20%7C%20React%20%7C%20LangChain-blue?style=for-the-badge)
![AI Model](https://img.shields.io/badge/LLM-Ollama%20(Gemma4E4B)-orange?style=for-the-badge)

## ✨ Features

- **📖 Intelligent Book Parsing**: Automatically load `.txt` and `.epub` books and build a high-performance vector index using ChromaDB and HuggingFace Embeddings.
- **🔍 Flexible Character Extraction**: Four strategies to identify major characters:
  - **Auto** (default) — tries Wikipedia → HanLP NER → LLM sampling, fully automatic
  - **HanLP NER** — uses a local HanLP MTL pipeline (no LLM needed); best for Chinese web novels (`pip install hanlp`)
  - **Local File (LLM)** — samples head/middle/tail of the file and uses the LLM to extract names; works without HanLP
  - **Wikipedia** — scrapes Wikipedia; best for well-known English novels
- **🤖 Deep RAG Analysis**: Retrieves specific scenes from the book to understand character appearance, clothing, and environment in different contexts.
- **🎬 AI Casting Director**: Suggests real-world actors (Hollywood, Bollywood, etc.) to serve as the visual "base" for the character, with support for specific decades (e.g., 1920s Noir, 1980s Sci-Fi).
- **🎭 Genre Adaptation**: Dynamically modifies clothing, hairstyles, and cinematic styles to fit genres (Horror, Cyberpunk, Fantasy, etc.) while preserving the character's core identity (age, ethnicity, facial features).
- **🖼️ ComfyUI Integration**: Seamlessly connect to your local ComfyUI server. Inject prompts into API-format workflows, track generation progress via Server-Sent Events (SSE), and preview images instantly.
- **✨ Premium UI**: A sleek, dark glassmorphism dashboard built with React and Vite for a modern, responsive feel.

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, FastAPI, LangChain
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **LLM**: Ollama (`Gemma4E4B:latest` or similar)
- **NER (optional)**: HanLP (`pip install hanlp`) — recommended for Chinese novels
- **Frontend**: React, Vite, Vanilla CSS
- **Image Gen**: ComfyUI (Local Instance)

## 🚀 Getting Started

Create and activate a conda or venv environment
conda create -n charactergen python=3.12 
activate charactergen 
        or 
python -m venv venv
.\venv\Scripts\activate

git clone https://github.com/snorcack/CharacterGeneration
cd CharacterGeneration/charactergenerate/frontend
npm install
cd CharacterGeneration/charactergenerate/backend
pip install -r requirements.txt
cd CharacterGeneration/charactergenerate

### Prerequisites

1.  **Ollama**: Install [Ollama](https://ollama.com/) and pull the required model:
    ```bash
    open command prompt  
    ollama pull Gemma4E4B:latest
    ```
1.  **ComfyUI**: Have a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance running.
2.  **Python & Node**: Ensure you have Python 3.10+ and Node.js installed.
3.  **https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes**  === needs to be installed in ComfyUI/custom_nodes
4.  **HanLP** *(optional, recommended for Chinese novels)*:
    ```bash
    pip install hanlp
    ```
    If not installed, the app falls back to LLM-based extraction automatically.


### Running the App

You can start both services with a single command on Windows:
```bash
./run_app.bat
```

Alternatively, run them manually:
- **Backend**: `cd backend && python main.py` (Runs on port 8000)
- **Frontend**: `cd frontend && npm run dev` (Runs on port 5173)

## 📖 How to Use

1.  **Load a Book**: Upload a `.txt` or `.epub` file, enter the book name, and choose a **Character List Source**:
    - *Auto* (default) — tries Wikipedia, then HanLP NER, then LLM extraction automatically
    - *HanLP NER* — recommended for Chinese web novels; requires `pip install hanlp`
    - *Local File (LLM)* — LLM-based name extraction; works without HanLP
    - *Wikipedia* — best for well-known English books
2.  **Choose a Character**: Select a character from the identified list.
3.  **Analyze & Customize**: The AI will generate a description and retrieve several scenes (scenarios). You can cast an actor to ground the visual appearance.
4.  **Configure Generation**: Select a genre, decade, and image generation parameters.
5.  **Generate**: Connect to ComfyUI, paste your API workflow JSON, and hit generate. Watch the progress in real-time until your portrait appears!

## 📁 Project Structure

```text
CharacterGenerationFromBook/
├── backend/            # FastAPI Server
│   ├── character_gen.py # RAG & LLM Logic
│   ├── comfyui.py      # ComfyUI API Integration
│   └── main.py          # API Endpoints
├── frontend/           # Vite + React UI
│   ├── src/            # Components & Styles
│   └── index.html
└── run_app.bat         # Windows Launcher
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Wowed by the portraits? Star the repo!* 🌟
