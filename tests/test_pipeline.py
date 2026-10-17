from pathlib import Path
import fitz
import pytest
from research_summarizer.fixtures import DEMO_TEXT
from research_summarizer.pdf_reader import PageText,extract_pdf,clean_page_text
from research_summarizer.pipeline import analyze_document,make_sentences,search_evidence
from research_summarizer.evaluation import rouge_scores

@pytest.fixture
def analysis():
    return analyze_document([PageText(1,DEMO_TEXT)],8)

def test_sample_summary_is_extractive(analysis):
    assert len(analysis.summary)==8
    assert all(e.text in [s.text for s in analysis.sentences] for e in analysis.summary)
    assert all(e.page==1 for e in analysis.summary)

def test_summary_order_and_metrics(analysis):
    assert [s.id for s in analysis.summary]==sorted(s.id for s in analysis.summary)
    assert 0 < analysis.summary_word_count < analysis.word_count
    assert 0 < analysis.reduction_percent < 100

def test_keywords(analysis):
    assert len(analysis.keywords)>=3
    assert all(isinstance(x,str) and x for x in analysis.keywords)

def test_reference_heading_is_excluded():
    text='Introduction\n\nThis detailed study evaluates NLP approaches for extracting research paper sentences.\n\nReferences\n\nSmith published a research paper about summarization systems in a journal.'
    assert all('Smith' not in x.text for x in make_sentences([PageText(3,text)]))

def test_evidence_search(analysis):
    found=search_evidence('TF-IDF sentence vectors cosine similarity',analysis.sentences)
    assert found and any('TF-IDF' in item['text'] or 'Cosine' in item['text'] for item in found)

def test_rouge_identical_and_disjoint():
    values=rouge_scores('the paper uses sentence graphs','the paper uses sentence graphs')
    assert all(v==1.0 for v in values.values())
    assert all(v==0 for v in rouge_scores('alpha beta gamma','red blue green').values())

def test_pdf_extraction():
    doc=fitz.open();page=doc.new_page();page.insert_text((72,90),DEMO_TEXT[:200]);page.insert_text((72,820),'Printed footer should not appear in the extracted body')
    raw=doc.tobytes();doc.close()
    pages=extract_pdf(raw)
    assert pages[0].page==1 and 'Scientific' in pages[0].text
    assert 'Printed footer' not in pages[0].text

def test_empty_pdf_rejected():
    with pytest.raises(ValueError):extract_pdf(b'')

def test_short_text_fails():
    with pytest.raises(ValueError):analyze_document([PageText(1,'Short text.')])

def test_invalid_parameters():
    with pytest.raises(ValueError):analyze_document([PageText(1,DEMO_TEXT)],length=0)

def test_clean_hyphen_wrap():
    assert clean_page_text('summariza-\ntion is useful')=='summarization is useful'
