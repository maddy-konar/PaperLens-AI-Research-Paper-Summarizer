<div align="center">

# 📄 PaperLens
### AI-Powered Research Paper Summarizer

**Turn lengthy research papers into concise, traceable insights using Natural Language Processing.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Extractive%20Summarization-00897B)
![Tests](https://img.shields.io/badge/Tests-Pytest-2E7D32)

</div>

---

## Overview

**PaperLens** is a local-first research paper summarization application that helps readers explore lengthy academic documents. Upload a text-selectable PDF, paste text, or try the built-in sample paper to generate an **extractive summary**, discover key terms, and locate supporting sentences from the source.

Rather than generating new claims, PaperLens selects and organizes important sentences from the original document, making its summaries easier to verify.

> **How it works:** PaperLens uses classical, unsupervised NLP (TF-IDF, cosine similarity, graph-based sentence ranking, and Maximal Marginal Relevance). It does **not** use a generative LLM or require an API key.

## Features

- **PDF and text input:** Upload a searchable PDF, paste research text, or use the sample paper.
- **Configurable summaries:** Choose between 3 and 20 sentences and adjust relevance versus diversity.
- **Source-linked results:** Review extracted sentences alongside available source-page references.
- **Keyword extraction:** Discover prominent terms using TF-IDF-based keyphrases.
- **Research insights:** Inspect document statistics and detected sections.
- **Evidence finder:** Search for relevant original sentences using a text query.
- **Export:** Download summaries as TXT or structured JSON.
- **Optional evaluation:** Compare against a human-written reference with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
- **Local processing:** No external LLM calls or paid API credentials are needed.

## Application Preview

`assets/dashboard.png`

## NLP Pipeline

```text
Research PDF / Pasted Text
          |
          v
   Text Extraction (PyMuPDF for PDFs)
          |
          v
   Text Cleaning & Sentence Splitting
          |
          v
   TF-IDF Sentence Representations
          |
          v
   Cosine Similarity Graph
          |
          v
   Graph-Based Sentence Ranking
          |
          v
   MMR: Relevance + Diversity
          |
          v
   Source-Ordered Extractive Summary
          |
          +--> Keywords / Insights
          +--> Evidence Search
          +--> TXT / JSON Export
          +--> Optional ROUGE Evaluation
```

The summary contains selected **original sentences**, not AI-written paraphrases.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python |
| Web interface | Streamlit |
| PDF extraction | PyMuPDF |
| NLP and vectorization | scikit-learn TF-IDF |
| Numerical and tabular processing | NumPy, Pandas |
| Ranking | Cosine similarity, graph centrality, MMR |
| Testing | Pytest |

## Getting Started

### Prerequisites

- Python 3.10 or later (3.10–3.12 recommended)
- pip
- A modern web browser

### 1. Clone the repository

```bash
git clone https://github.com/maddy-konar/PaperLens-AI-Research-Paper-Summarizer.git
cd PaperLens-AI-Research-Paper-Summarizer
```


### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** if the browser does not open automatically.

## How to Use

1. Select **Upload PDF**, **Paste text**, or **Demo paper**.
2. For PDFs, upload a document with selectable text (not an image-only scan).
3. Adjust the **Sentences in summary** and **Importance vs. diversity** controls.
4. Click **Analyze paper**.
5. Explore the **Summary**, **Key insights**, **Evidence finder**, and **Evaluation** tabs.
6. Download the summary or inspect original evidence before citing findings.

## Project Structure

```text
PaperLens/
├── .streamlit/
│   └── config.toml
├── assets/
│   └── architecture.png
├── research_summarizer/
│   ├── __init__.py
│   ├── __main__.py
│   ├── evaluation.py
│   ├── fixtures.py
│   ├── pdf_reader.py
│   └── pipeline.py
├── samples/
│   ├── Sample_Research_Paper.pdf
│   ├── Sample_Research_Paper.txt
│   └── Sample_Summary_Output.txt
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── run_windows.bat
└── run_linux_mac.sh
```

## Run Tests

```bash
python -m pytest -q
```

A command-line version is also available:

```bash
python -m research_summarizer --pdf samples/Sample_Research_Paper.pdf --length 6
```

## Limitations

- Supports **extractive** summarization; it does not rewrite or fact-check scientific claims.
- Works best with **English, text-selectable PDFs**; scanned PDFs need OCR, which is not included.
- Multi-column layouts, equations, tables, and figures may not be extracted perfectly.
- Analysis is capped at **20 MB, 80 PDF pages, 450,000 characters, and 550 sentences**; content beyond these limits may be excluded.
- ROUGE requires a human-written reference summary and measures text overlap, **not model accuracy or factual correctness**.
- Source-page links should be checked against the original document before quoting or citing.

## References

- [TextRank: Bringing Order into Texts (Mihalcea & Tarau, 2004)](https://aclanthology.org/W04-3252/)
- [ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004)](https://aclanthology.org/W04-1013/)
- [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">

**PaperLens** · Research summarization with transparent, source-linked NLP

</div>
