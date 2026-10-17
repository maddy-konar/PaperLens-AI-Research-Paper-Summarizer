import argparse
from pathlib import Path
from .pdf_reader import extract_pdf, PageText
from .pipeline import analyze_document
from .fixtures import DEMO_TEXT

def main():
    parser=argparse.ArgumentParser(description='Research Summarizer CLI demo')
    parser.add_argument('--pdf',type=Path,help='optional input PDF')
    parser.add_argument('--length',type=int,default=8)
    args=parser.parse_args()
    pages=extract_pdf(args.pdf.read_bytes()) if args.pdf else [PageText(1,DEMO_TEXT)]
    result=analyze_document(pages,args.length)
    print('RESEARCH PAPER SUMMARY\n')
    print(result.summary_text())
    print('\nKEYWORDS:',', '.join(result.keywords))
    print('WORDS:',result.word_count,'SUMMARY:',result.summary_word_count)

if __name__=='__main__':main()
