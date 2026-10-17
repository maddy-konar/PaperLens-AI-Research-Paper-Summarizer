"""PaperLens — polished, local-first Streamlit interface for extractive NLP."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from research_summarizer.evaluation import rouge_scores
from research_summarizer.fixtures import DEMO_TEXT
from research_summarizer.pdf_reader import PageText, extract_pdf
from research_summarizer.pipeline import analyze_document, search_evidence

st.set_page_config(page_title="PaperLens — Research intelligence", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--ink:#162b3e;--muted:#64748b;--mint:#1ca99c;--soft:#f6f9fc;--line:#e6edf2}
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif}
.stApp {background:linear-gradient(180deg,#f5f9fc 0,#ffffff 480px)}
.block-container {max-width:1240px;padding-top:1.75rem;padding-bottom:3rem}
section[data-testid="stSidebar"] {background:#102c3c}
section[data-testid="stSidebar"] * {color:#f1f7f9!important}
section[data-testid="stSidebar"] [data-baseweb="slider"] * {color:inherit!important}
section[data-testid="stSidebar"] hr {border-color:#33505e}
section[data-testid="stSidebar"] .stCaption {opacity:.75}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em;color:#173347}
h3 {font-size:1.18rem!important}
.hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#102f41,#135b62 67%,#1c897e);border-radius:23px;padding:37px 42px;margin-bottom:24px;color:#fff;box-shadow:0 18px 42px rgba(18,64,80,.13)}
.hero::after {content:"";position:absolute;width:260px;height:260px;border:1px solid #ffffff38;border-radius:50%;right:-70px;top:-95px;box-shadow:0 0 0 52px #ffffff08,0 0 0 108px #ffffff08}
.hero .eyebrow {font-size:.7rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:#9ef6de;margin-bottom:10px}
.hero h1 {color:#fff;font-weight:800;font-size:2.55rem;margin:0 0 6px;line-height:1.18}
.hero p {color:#d2eaeb;font-size:1.02rem;max-width:650px;margin:0 0 18px}
.hero .pill {display:inline-block;border:1px solid #ffffff50;border-radius:40px;padding:6px 12px;font-size:.76rem;color:#f1ffff;margin-right:8px;margin-bottom:4px;background:#ffffff14}
.section-title {font-family:'Manrope';font-size:1.13rem;font-weight:800;color:var(--ink);margin:12px 0}
.small-help {color:var(--muted);font-size:.88rem;margin-bottom:12px}
.step-tag {display:inline-flex;align-items:center;gap:6px;color:#158b82;background:#e8f7f4;border-radius:9px;padding:5px 9px;font-weight:800;font-size:.73rem;letter-spacing:.06em}
.result-card {background:#fff;border:1px solid var(--line);border-radius:15px;padding:18px 20px;margin:0 0 11px;box-shadow:0 3px 12px #183e4d07}
.result-card .label {font-weight:800;font-size:.73rem;letter-spacing:.06em;color:#15948b;text-transform:uppercase;margin-bottom:8px}
.result-card .body {font-size:.96rem;line-height:1.73;color:#263e4f}
.keyword {display:inline-block;background:#edf7f6;color:#176e71;border:1px solid #d5eae8;border-radius:50px;padding:7px 13px;margin:4px;font-size:.85rem;font-weight:700}
.stButton>button[kind="primary"] {background:linear-gradient(90deg,#138e87,#126c7e);border:0;border-radius:12px;padding:13px 19px;font-weight:800;box-shadow:0 5px 18px #138e872b}
.stButton>button[kind="primary"]:hover {background:#115e70}
[data-testid="stMetric"] {background:white;border:1px solid var(--line);border-radius:15px;padding:16px 17px;box-shadow:0 5px 20px #103c4d08}
[data-testid="stMetricLabel"] {color:#6b8291}
[data-testid="stMetricValue"] {color:#143b4b;font-family:Manrope,sans-serif}
[data-testid="stFileUploader"] {border-radius:14px}
[data-baseweb="tab-list"] {gap:12px}
button[data-baseweb="tab"] {font-weight:700}
.footer {text-align:center;margin-top:28px;color:#788997;font-size:.82rem}
@media(max-width:650px){.hero{padding:27px 23px}.hero h1{font-size:1.95rem}.block-container{padding-top:1rem}}
</style>""", unsafe_allow_html=True)

st.markdown('''<div class="hero"><div class="eyebrow">Research intelligence workspace / NLP mini project</div>
<h1>PaperLens <span style="color:#9df1dc">✦</span></h1>
<p>Go from a dense research paper to clear, traceable insights — in just a few clicks.</p>
<span class="pill">✦ Extractive NLP</span><span class="pill">⌕ Source-linked summaries</span><span class="pill">♧ Private & local</span>
</div>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## ◈ PaperLens')
    st.caption('YOUR RESEARCH ASSISTANT')
    st.divider()
    st.markdown('### ⚙️ Summary controls')
    length = st.slider('Sentences in summary', 3, 20, 8, help='Select how many sentences will appear in the extractive summary.')
    diversity = st.slider('Importance vs. diversity', .45, 1.0, .70, step=.05, help='Higher favors ranked importance; lower favors non-redundant coverage.')
    st.divider()
    st.markdown('### How it works')
    st.markdown('**01**  Extract and split text  \n\n**02**  TF-IDF + similarity graph  \n\n**03**  Rank and diversify sentences  \n\n**04**  Explore traceable results')
    st.divider()
    st.caption('🔒 Processing is local after dependencies are installed. No API key needed.')

st.markdown('<div class="step-tag">STEP 01 &nbsp; / &nbsp; ADD YOUR PAPER</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Start a new analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="small-help">Upload a searchable PDF, paste text, or explore the built-in sample document.</div>', unsafe_allow_html=True)

source = st.radio('Choose input method', ['📄 Upload PDF', '✍️ Paste text', '🧪 Demo paper'], horizontal=True, label_visibility='collapsed')
pages = None
source_name = ''
if source == '📄 Upload PDF':
    file = st.file_uploader('Drop your research paper here', type=['pdf'], help='Text-based PDF only; up to 20 MB and 80 pages.')
    if file:
        source_name = file.name
        try:
            pages = extract_pdf(file.getvalue())
            st.success(f'✓ {len(pages)} pages loaded from {file.name}')
        except ValueError as exc:
            st.error(str(exc))
elif source == '✍️ Paste text':
    raw_text = st.text_area('Paper text', height=180, placeholder='Paste the abstract, introduction, methods, findings, and conclusion...')
    if raw_text.strip():
        pages = [PageText(1, raw_text)]
        source_name = 'Pasted document'
else:
    pages = [PageText(1, DEMO_TEXT)]
    source_name = 'Educational demonstration paper'
    st.info('Try the full workflow with a fictional, educational sample paper included in this project.')

left, right = st.columns([1, 3], vertical_alignment='center')
with left:
    analyze = st.button('✦ Analyze paper', type='primary', disabled=pages is None, use_container_width=True)
with right:
    st.caption('No external LLM calls. Extractive summaries reuse original sentences rather than generating new claims.')

signature = (source, source_name, hash(''.join(p.text for p in pages)) if pages else None, length, diversity)
if analyze:
    try:
        with st.spinner('Extracting key insights…'):
            result = analyze_document(pages, length=length, diversity=diversity)
        st.session_state['paperlens_result'] = result
        st.session_state['paperlens_signature'] = signature
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get('paperlens_result')
if result is not None and st.session_state.get('paperlens_signature') != signature:
    st.info('Your input or settings changed. Click **Analyze paper** to refresh the results.')
    result = None

if result is not None:
    st.divider()
    st.markdown('<div class="step-tag">STEP 02 &nbsp; / &nbsp; YOUR RESULTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Paper at a glance</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Original words', f'{result.word_count:,}')
    m2.metric('Summary words', f'{result.summary_word_count:,}')
    m3.metric('Text reduced', f'{result.reduction_percent:.1f}%')
    m4.metric('Sentences analyzed', f'{result.sentence_count:,}')

    tab_summary, tab_insights, tab_evidence, tab_eval = st.tabs(['✦  Summary', '◈  Key insights', '⌕  Evidence finder', '▥  Evaluation'])
    with tab_summary:
        st.markdown('### Your research summary')
        st.caption('Each extracted sentence preserves the original wording. Page labels locate evidence; they do not verify its claims.')
        for index, item in enumerate(result.summary, 1):
            location = f'PAGE {item.page}' if source == '📄 Upload PDF' else 'SOURCE TEXT'
            # Escape uploaded text: never inject user PDF contents into HTML.
            import html
            safe_text = html.escape(item.text)
            safe_section = html.escape(item.section)
            st.markdown(f'<div class="result-card"><div class="label">{index:02d} &nbsp; · &nbsp; {location} &nbsp; · &nbsp; {safe_section}</div><div class="body">{safe_text}</div></div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        dl1.download_button('↓ Export summary (TXT)', data=result.summary_text(), file_name='paperlens_summary.txt', mime='text/plain', use_container_width=True)
        dl2.download_button('↓ Export full analysis (JSON)', data=json.dumps(result.as_export(), indent=2), file_name='paperlens_analysis.json', mime='application/json', use_container_width=True)
    with tab_insights:
        st.markdown('### Important topics & keywords')
        if result.keywords:
            import html
            st.markdown(''.join(f'<span class="keyword">{html.escape(word)}</span>' for word in result.keywords), unsafe_allow_html=True)
        else:
            st.info('No keywords identified.')
        st.markdown('### Content by section')
        section_data = pd.DataFrame([{'Section': name, 'Sentences': count} for name, count in result.section_counts.items()])
        if not section_data.empty:
            st.bar_chart(section_data.set_index('Section'), color='#15998f')
        st.caption('Section detection uses heading heuristics; unusual PDF layouts may cause inaccuracies.')
    with tab_evidence:
        st.markdown('### Search within the paper')
        st.caption('Locate original sentences related to a method, result, research question, or key term.')
        query = st.text_input('What do you want to find?', placeholder='e.g. dataset, evaluation method, limitations')
        if query.strip():
            matches = search_evidence(query, result.sentences)
            if not matches:
                st.warning('No lexical match found. Try a different phrase.')
            for match in matches:
                import html
                location = f'PAGE {match["page"]}' if match['page'] is not None else 'TEXT'
                st.markdown(f'<div class="result-card"><div class="label">{html.escape(match["section"])} · {location} · SIMILARITY {match["similarity"]:.2f}</div><div class="body">{html.escape(match["text"])}</div></div>', unsafe_allow_html=True)
        st.caption('Similarity scores represent text overlap, not truth or confidence in a factual claim.')
    with tab_eval:
        st.markdown('### Compare against a reference summary')
        st.caption('Paste a human-written reference summary to calculate ROUGE overlap scores.')
        reference = st.text_area('Reference summary', height=140, placeholder='Paste a benchmark or instructor-provided summary...')
        if reference.strip():
            scores = rouge_scores(result.summary_text(include_citations=False), reference)
            st.dataframe(pd.DataFrame([{'Metric': key, 'F1': round(value, 4)} for key, value in scores.items()]), hide_index=True, use_container_width=True)
        else:
            st.info('A reference summary is required for ROUGE evaluation. No benchmark performance is assumed.')
        st.caption('ROUGE estimates lexical overlap, not factual correctness. ROUGE-L truncates inputs to 2,500 tokens.')
else:
    st.markdown('''<div style="margin-top:20px;padding:22px;border:1px dashed #ccdae0;border-radius:16px;background:#ffffffa8;color:#64798b;text-align:center">
    <div style="font-size:1.7rem">◈</div><strong style="color:#23465a">Your insights will appear here</strong><br>
    Select a document above and click <strong>Analyze paper</strong> to see a summary, keywords, statistics, and evidence.
    </div>''', unsafe_allow_html=True)

st.markdown('<div class="footer">PaperLens · Explainable NLP research summarization · Verify findings against the original research paper</div>', unsafe_allow_html=True)
