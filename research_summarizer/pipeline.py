"""Evidence-linked extractive summarizer using TF-IDF, graph centrality and MMR.

No external model downloads, API access, network calls, or invented citations.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Iterable
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from .pdf_reader import PageText

MAX_SENTENCES = 550
HEADINGS = ('abstract','introduction','background','related work','literature review',
            'materials and methods','methods','methodology','approach','proposed system',
            'experimental setup','experiments','results','evaluation','discussion',
            'conclusion','conclusions','future work','limitations','references','bibliography')
SECTION_ALIASES = {
    'materials and methods':'Methods', 'methodology':'Methods','methods':'Methods',
    'approach':'Methods','proposed system':'Methods','experimental setup':'Methods',
    'experiments':'Methods','evaluation':'Results','conclusions':'Conclusion',
    'references':'References','bibliography':'References','related work':'Background',
    'literature review':'Background','future work':'Conclusion'
}

@dataclass(frozen=True)
class Sentence:
    id: int
    text: str
    page: int | None
    section: str

@dataclass(frozen=True)
class Evidence:
    id: int
    text: str
    page: int | None
    section: str
    score: float

@dataclass
class DocumentAnalysis:
    summary: list[Evidence]
    keywords: list[str]
    full_text: str
    sentence_count: int
    word_count: int
    summary_word_count: int
    reduction_percent: float
    section_counts: dict[str, int]
    sentences: list[Sentence]
    scores: list[float]

    def summary_text(self, include_citations: bool = True) -> str:
        lines=[]
        for item in self.summary:
            citation = f' [p. {item.page}]' if include_citations and item.page else ''
            lines.append(item.text + citation)
        return '\n\n'.join(lines)

    def as_export(self) -> dict:
        return {
            'method':'Extractive TF-IDF similarity graph + PageRank-style centrality + MMR',
            'disclaimer':'Extractive sentences copied from input; page labels indicate source location, not verification of claims.',
            'summary':[asdict(item) for item in self.summary],
            'keywords':self.keywords,
            'original_words':self.word_count,
            'summary_words':self.summary_word_count,
            'reduction_percent':self.reduction_percent,
            'sentence_count':self.sentence_count,
            'section_counts':self.section_counts,
        }


def _normalize_heading(s: str) -> str | None:
    val = re.sub(r'^\s*(?:\d+(?:\.\d+)*[\.)]?\s+|[IVX]+\.\s+)', '', s.strip())
    val = re.sub(r'\s*[:.]\s*$', '', val).lower()
    if len(val) <= 35 and val in HEADINGS:
        return SECTION_ALIASES.get(val, val.title())
    return None


def _split_sentences(text: str) -> list[str]:
    # Rule-based sentence segmentation; protects common abbreviations and decimals.
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\b(e\.g|i\.e|et al|Fig|Dr|Prof|Eq|No)\.', lambda m:m.group(0).replace('.', '<DOT>'), text, flags=re.I)
    text = re.sub(r'(?<=\d)\.(?=\d)', '<DOT>', text)
    pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z\(\[0-9])', text)
    return [p.replace('<DOT>', '.').strip() for p in pieces if p.strip()]


def make_sentences(pages: Iterable[PageText]) -> list[Sentence]:
    result: list[Sentence] = []
    current_section = 'General'
    for page in pages:
        # Original page extraction is whitespace-normalized but blank lines preserved.
        for paragraph in re.split(r'\n{2,}', page.text):
            fragment = paragraph.strip()
            if not fragment:
                continue
            # Headings can stand alone; PDFs may also collapse them into a paragraph.
            header = _normalize_heading(fragment)
            if header:
                current_section = header
                if current_section == 'References':
                    break
                continue
            if current_section == 'References':
                break
            for sent in _split_sentences(fragment):
                words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", sent)
                if 7 <= len(words) <= 130 and len(sent) >= 38 and sent.rstrip().endswith(('.', '?', '!')):
                    result.append(Sentence(len(result)+1, sent, page.page, current_section))
                    if len(result) >= MAX_SENTENCES:
                        return result
        if current_section == 'References':
            break
    return result


def _pagerank(similarities: np.ndarray, damping: float = .85) -> np.ndarray:
    count = similarities.shape[0]
    if count == 1:
        return np.array([1.0])
    graph = np.maximum(similarities.copy(), 0.0)
    np.fill_diagonal(graph, 0.0)
    sums=graph.sum(axis=1)
    trans=graph / np.where(sums == 0, 1, sums)[:,None]
    ranks=np.ones(count)/count
    teleport=(1-damping)/count
    for _ in range(120):
        next_rank=teleport+damping*(trans.T @ ranks + ranks[sums==0].sum()/count)
        if np.max(np.abs(next_rank-ranks)) < 1e-9:
            ranks=next_rank;break
        ranks=next_rank
    return ranks


def _scale(a: np.ndarray) -> np.ndarray:
    mn=float(np.min(a));mx=float(np.max(a))
    return np.zeros_like(a) if mx-mn < 1e-12 else (a-mn)/(mx-mn)


def _keywords(sentences: list[Sentence], count: int=12) -> list[str]:
    if not sentences:
        return []
    docs=[s.text for s in sentences]
    try:
        v=TfidfVectorizer(stop_words='english', ngram_range=(1,2),
                           max_features=12000, token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b')
        x=v.fit_transform(docs)
    except ValueError:
        return []
    strength=np.asarray(x.sum(axis=0)).reshape(-1)
    terms=v.get_feature_names_out()
    ordered=np.argsort(-strength)
    selected=[]
    for idx in ordered:
        t=str(terms[idx]); toks=t.split()
        if any(tok in {'paper','study','research','results','method','methods','using','based','proposed'} for tok in toks):
            continue
        if any(set(toks).issubset(set(k.split())) or set(k.split()).issubset(set(toks)) for k in selected):
            continue
        selected.append(t)
        if len(selected)==count:
            break
    return selected


def analyze_document(pages: list[PageText], length: int = 8, diversity: float=.70) -> DocumentAnalysis:
    if not 0.45 <= diversity <= 1.0:
        raise ValueError('Diversity/relevance balance must lie between 0.45 and 1.0.')
    if not 1 <= length <= 30:
        raise ValueError('Summary length must be 1 to 30 sentences.')
    sentences=make_sentences(pages)
    if not sentences:
        raise ValueError('Not enough complete sentences found. Try a longer text or another paper.')
    original='\n\n'.join(p.text for p in pages)
    corpus=[s.text for s in sentences]
    try:
        vectorizer=TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=15000, sublinear_tf=True)
        features=vectorizer.fit_transform(corpus)
    except ValueError as exc:
        raise ValueError('No usable words found for NLP analysis.') from exc
    similarities=linear_kernel(features,features)
    centrality=_scale(_pagerank(similarities))
    # Structural bias is a small prior, not an evaluation score.
    section_weight={'Abstract':.80,'Introduction':.24,'Methods':.37,
                    'Results':.90,'Discussion':.63,'Conclusion':.95,
                    'Background':.10,'General':.10}
    structure=np.array([section_weight.get(s.section,.18) for s in sentences])
    position=np.array([1.0-(i/max(1,len(sentences)-1)) for i in range(len(sentences))])
    relevance=.74*centrality+.19*structure+.07*position
    max_allowed=min(length,len(sentences))
    chosen=[]
    while len(chosen) < max_allowed:
        best_idx=-1;best_score=-1e12
        for i in range(len(sentences)):
            if i in chosen:
                continue
            novelty=max((similarities[i,j] for j in chosen), default=0.0)
            # Greedy maximum marginal relevance discourages near-duplicate summary sentences.
            score=diversity*relevance[i]-(1-diversity)*novelty
            if score > best_score:
                best_score=score;best_idx=i
        chosen.append(best_idx)
    # Present in original reading order (not rank order) for coherence.
    chosen.sort()
    evidence=[Evidence(sentences[i].id,sentences[i].text,sentences[i].page,
                       sentences[i].section,round(float(relevance[i]),4)) for i in chosen]
    total_words=len(re.findall(r'\b\w+\b',original))
    summary_words=sum(len(re.findall(r'\b\w+\b',s.text)) for s in evidence)
    sections=dict(Counter(s.section for s in sentences))
    return DocumentAnalysis(evidence,_keywords(sentences),original,len(sentences),
                            total_words,summary_words,
                            round(100*(1-summary_words/max(1,total_words)),1),
                            sections,sentences,[float(x) for x in relevance])


def search_evidence(query: str, sentences: list[Sentence], top_k: int=4) -> list[dict]:
    """Retrieve relevant original sentences; no generated answers or fabricated claims."""
    if not query.strip() or not sentences:
        return []
    docs=[s.text for s in sentences]
    try:
        v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=15000)
        x=v.fit_transform(docs)
        q=v.transform([query])
    except ValueError:
        return []
    scores=cosine_similarity(q,x).reshape(-1)
    order=np.argsort(-scores)
    return [{'text':docs[i], 'page':sentences[i].page,'section':sentences[i].section,
             'similarity':round(float(scores[i]),3)} for i in order[:top_k] if scores[i] > 0.025]
