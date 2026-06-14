# 🔍 Interactive Local Text Document Search Engine

An interactive local document search engine built using Information Retrieval techniques. Search a local text corpus and compare different retrieval methods including **TF-IDF**, **Semantic Search**, **Hybrid Retrieval**, and **LLM-based Query Expansion**.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow?style=flat-square)](https://huggingface.co/spaces/apurv20/Local_Document_Search_Engine)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)

---

## 🎯 Live Demo

**[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/apurv20/Local_Document_Search_Engine)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Files](#-project-files)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Example Queries](#-example-queries)
- [Evaluation Metrics](#-evaluation-metrics)
- [Method Comparison](#-method-comparison)
- [Deployment](#-deployment)
- [Jupyter Notebook](#-jupyter-notebook)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Overview

In real-world systems, organizations often have large collections of text documents. This project builds a **controlled test environment** to explore how different retrieval methods perform on the same document corpus.

Instead of claiming to build a new search engine, this system focuses on:
- ✅ Comparing retrieval strategies in a unified application
- ✅ Implementing multiple IR techniques
- ✅ Evaluating performance with standard metrics
- ✅ Providing an interactive interface for exploration

The app uses the **20 Newsgroups dataset** as the document collection and provides a **Gradio-based web interface** for searching, evaluating, and comparing retrieval results.

---

## ✨ Features

- 🔎 **Multi-Method Search**: Compare 4 different retrieval approaches
- 📊 **Evaluation Metrics**: Precision, Recall, F1, and Average Precision calculations
- 🎯 **Category-Based Relevance**: Evaluate against ground-truth newsgroup categories
- 🔄 **Hybrid Retrieval**: Combine TF-IDF and semantic similarity with adjustable weights
- 🤖 **LLM Query Expansion**: Experiment with AI-powered query enhancement
- 📈 **Interactive Comparison**: Side-by-side evaluation of different methods
- ⚡ **Fast Deployment**: Precomputed indexes for quick startup
- 🎨 **User-Friendly Interface**: Gradio-based web application

---

## 🧠 How It Works

### Retrieval Methods

#### 1. **TF-IDF Search** 
- Traditional keyword-based retrieval
- High weight for important but uncommon terms
- Best for exact keyword matching
- Fast and interpretable

#### 2. **Semantic Search**
- Uses sentence embeddings (SentenceTransformers)
- Retrieves semantically similar documents
- Works even with different wording
- Better for meaning-based queries

#### 3. **Hybrid Search**
- Combines TF-IDF and semantic scores
- Adjustable `alpha` parameter controls the weight distribution
- Balances keyword matching with semantic similarity
- Best of both worlds

#### 4. **LLM Query Expansion**
- Generates related keywords using language models
- Uses expanded terms for document retrieval
- Experimental approach to improve recall
- Can help with topic drift exploration

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | Gradio |
| **Language** | Python 3.8+ |
| **ML/NLP** | Scikit-learn, SentenceTransformers, Transformers, PyTorch |
| **Data Processing** | Pandas, NumPy |
| **Deployment** | Hugging Face Spaces |

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio application with UI and search logic |
| `requirements.txt` | Python dependencies |
| `Interactive_Local_Text_Document_Search_Engine_DEPLOYMENT_READY.ipynb` | 📓 Full development notebook (see [Jupyter Notebook](#-jupyter-notebook) section) |
| `README.md` | Project documentation |
| `search_engine_index.pkl` | Precomputed search index (used in deployment) |

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/apurvchokshi/Local_Text_Document_Search_Engine.git
   cd Local_Text_Document_Search_Engine
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the app**
   - Open your browser and go to the local Gradio link (usually `http://127.0.0.1:7860`)

---

## 🎮 Usage

### Search Tab
1. Enter your search query
2. Select a retrieval method (TF-IDF, Semantic, Hybrid, or Query Expansion)
3. Adjust parameters if needed (e.g., `alpha` for hybrid search, `top_k` for result count)
4. View ranked search results with relevance scores

### Evaluation Tab
1. Select a category from the 20 Newsgroups dataset as ground truth
2. Choose a retrieval method
3. Enter a test query
4. View evaluation metrics (Precision, Recall, F1, Average Precision)

### Comparison Tab
1. Enter a query
2. Compare all retrieval methods side-by-side
3. Analyze how different approaches handle the same query

---

## 📝 Example Queries

Try these sample queries to explore different retrieval behaviors:

```
space nasa moon              → Astronomy/Space topic
computer graphics rendering → Graphics/Technology topic
baseball game score         → Sports topic
religion and belief         → Religion/Philosophy topic
car engine performance      → Automotive topic
windows operating system    → Computing topic
medicine health treatment   → Healthcare topic
```

---

## 📊 Evaluation Metrics

The system calculates the following performance metrics:

| Metric | Definition |
|--------|-----------|
| **Precision@K** | Fraction of retrieved documents that are relevant |
| **Recall@K** | Fraction of relevant documents that were retrieved |
| **F1@K** | Harmonic mean of Precision and Recall |
| **Average Precision@K** | Mean of precision scores at each relevant document |
| **Relevant Documents Found** | Count of retrieved relevant documents |
| **Total Relevant Documents** | Count of all relevant documents in corpus |

---

## 🔄 Method Comparison

### Performance Characteristics

| Method | Strengths | Weaknesses | Best For |
|--------|-----------|-----------|----------|
| **TF-IDF** | Fast, interpretable, exact matching | Misses paraphrased content | Keyword-heavy queries |
| **Semantic** | Understands meaning, handles paraphrasing | Slower, can miss exact terms | Conceptual queries |
| **Hybrid** | Balances both approaches | Requires parameter tuning | General-purpose search |
| **Query Expansion** | Can improve recall | May introduce drift | Exploratory search |

---

## 🚀 Deployment

### Faster Deployment with Precomputed Index

The heavy preprocessing (indexing, embedding computation) is done offline in Google Colab and saved as:

```
search_engine_index.pkl
```

This file contains:
- Document collection
- TF-IDF index
- Document vectors
- Semantic embeddings
- Labels and metadata

**Benefit**: The Hugging Face Space doesn't need to rebuild everything on startup—just loads the precomputed index.

### Deploying to Hugging Face Spaces

1. Create a Space on Hugging Face
2. Upload `app.py`, `requirements.txt`, and `search_engine_index.pkl`
3. Set the Space to run Python
4. The app will automatically deploy

---

## 📓 Jupyter Notebook

The repository includes a comprehensive Jupyter notebook: **`Interactive_Local_Text_Document_Search_Engine_DEPLOYMENT_READY.ipynb`**

### What's Covered:

1. **Data Loading & Preprocessing** - Loading the 20 Newsgroups dataset, text cleaning and tokenization
2. **Index Building** - TF-IDF vectorization, semantic embedding computation
3. **Search Implementation** - TF-IDF, semantic, hybrid, and query expansion search logic
4. **Evaluation** - Category-based relevance, metric calculation, performance analysis
5. **Deployment Export** - Saving precomputed indexes for fast deployment

### How to Use:
- **View**: Open in Jupyter Notebook or Jupyter Lab
- **Run in Colab**: Execute in Google Colab for cloud computing
- **Learn**: Study the code to understand the full implementation
- **Experiment**: Modify parameters and test different approaches

---

## 🔮 Future Improvements

- [ ] User-uploaded document support (drag & drop)
- [ ] Multi-format support (PDF, DOCX, TXT)
- [ ] BM25 retrieval method
- [ ] Neural reranking models (cross-encoders)
- [ ] Improved query expansion quality
- [ ] Search history and bookmarking
- [ ] Better visualization for longer results
- [ ] Charts and dashboards for evaluation metrics
- [ ] Batch evaluation mode
- [ ] Export results functionality

---

## 📚 References

- [SentenceTransformers](https://www.sentencetransformers.net/)
- [Scikit-learn](https://scikit-learn.org/)
- [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/)
- [Gradio](https://gradio.app/)
- [Hugging Face](https://huggingface.co/)

---

## 👤 Author

**Apurv Chokshi**
- GitHub: [@apurvchokshi](https://github.com/apurvchokshi)
- Hugging Face: [@apurv20](https://huggingface.co/apurv20)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Share improvements

---

## 📞 Questions & Feedback

If you have any questions, suggestions, or feedback about this project, feel free to:
- Open an issue on GitHub
- Reach out via email
- Check out the live demo on Hugging Face Spaces

---

**⭐ If you find this project useful, please consider giving it a star!**
