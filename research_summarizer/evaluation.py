"""Reference-based ROUGE-1, ROUGE-2 and ROUGE-L F1 (lightweight implementation)."""
from __future__ import annotations
import re
from collections import Counter


def tokens(text: str) -> list[str]:
    return re.findall(r'\b[a-z0-9]+\b',text.lower())


def _f1(match: int, predicted: int, expected: int) -> float:
    if predicted == 0 or expected == 0 or match == 0:
        return 0.0
    precision=match/predicted
    recall=match/expected
    return 2*precision*recall/(precision+recall)


def _ngrams(items: list[str], n: int) -> Counter:
    return Counter(tuple(items[i:i+n]) for i in range(len(items)-n+1))


def _lcs_length(x: list[str], y: list[str]) -> int:
    prev=[0]*(len(y)+1)
    for a in x:
        cur=[0]
        for j,b in enumerate(y,1):
            cur.append(prev[j-1]+1 if a == b else max(prev[j],cur[j-1]))
        prev=cur
    return prev[-1]


def rouge_scores(prediction: str, reference: str) -> dict[str,float]:
    p=tokens(prediction);r=tokens(reference)
    outcome={}
    for n in (1,2):
        a=_ngrams(p,n);b=_ngrams(r,n)
        matched=sum((a & b).values())
        outcome[f'ROUGE-{n} F1']=round(_f1(matched,sum(a.values()),sum(b.values())),3)
    # Exact LCS can be quadratic; bound lengths and declare it in documentation.
    pp=p[:2500];rr=r[:2500]
    outcome['ROUGE-L F1']=round(_f1(_lcs_length(pp,rr),len(pp),len(rr)),3)
    return outcome
