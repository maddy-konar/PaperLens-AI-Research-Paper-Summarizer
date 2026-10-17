"""Bundled didactic demo text (not a published or peer-reviewed research paper)."""
DEMO_TEXT = '''Abstract

Scientific articles are often long and difficult to read under time constraints. This educational demonstration explores an extractive text summarizer for research papers. The system retrieves salient sentences from the document instead of generating new factual claims. Each selected sentence retains a page reference when a PDF is supplied. The workflow uses term frequency inverse document frequency features, graph ranking, and redundancy reduction.

Introduction

Students frequently need to review large numbers of research papers for coursework and literature surveys. Manual summarization can be time consuming and may overlook relevant methods or limitations. Automated summarization can provide a first-pass reading aid, but selected sentences should always be checked against the original source. Earlier graph-based methods model sentences as connected nodes and use their similarity to estimate importance. Such techniques do not require a labeled training corpus or an external language-model API.

Methods

The system first extracts selectable text from each PDF page using a PDF parsing library. The preprocessing stage merges wrapped lines and splits text into candidate sentences. Sentence vectors are constructed using TF-IDF features with unigram and bigram terms. Cosine similarity between vectors forms a weighted sentence graph. A PageRank-style iterative procedure assigns importance scores based on sentence connections. A small section prior favors sentences from abstracts, results, and conclusions. Maximum marginal relevance balances high ranking scores against similarity to sentences already included in the summary. Selected sentences are sorted back into their original document order for readability. A simple keyword component ranks content phrases by their TF-IDF importance.

Results

In this illustrative example, the software produces a short extractive summary with links back to its supporting source sentences. The demonstration does not report benchmark accuracy, because no human-authored reference summaries were collected for controlled evaluation. The application provides optional ROUGE scores when the user supplies a reference summary. A keyword list and an evidence-search panel offer alternative ways to inspect extracted document content. The complete pipeline runs locally after installing Python dependencies.

Discussion

Graph ranking can prefer commonly repeated topics while omitting unusual but essential contributions. Text extraction quality may degrade for equations, scanned pages, tables, and two-column PDF layouts. The rule-based sentence splitter may struggle with domain-specific abbreviations and formulas. Reported ROUGE values measure lexical overlap with a reference rather than factual correctness or usefulness. For this reason, a summary should be treated as reading assistance instead of authoritative evidence.

Conclusion

This educational prototype combines PDF parsing, TF-IDF features, sentence similarity, graph centrality, and redundancy-aware selection. Its main advantage is traceability because summary sentences can be checked in the original PDF. Future extensions could add OCR, scientific section detection, multilingual support, and controlled comparisons with neural summarization systems.'''
