import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from collections import defaultdict
import numpy as np

def calculate_bleu_scores(references, hypotheses):
    smoothie = SmoothingFunction().method4
    scores = {
        "BLEU-1": corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothie),
        "BLEU-2": corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        "BLEU-3": corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        "BLEU-4": corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }
    return scores

def calculate_rouge_scores(references, hypotheses):
    rouge = Rouge()
    refs = [" ".join(ref[0]) for ref in references]  # use first reference
    hyps = [" ".join(hyp) for hyp in hypotheses]
    try:
        scores = rouge.get_scores(hyps, refs, avg=True)
    except ValueError:
        scores = {"rouge-1": {"f":0}, "rouge-2": {"f":0}, "rouge-l": {"f":0}}
    return {k: v["f"] for k, v in scores.items()}

def calculate_cider(references, hypotheses):
    """
    Simplified CIDEr based on TF-IDF weighting
    """
    word_to_doc_count = defaultdict(int)
    for refs in references:
        unique = set([w for ref in refs for w in ref])
        for w in unique: word_to_doc_count[w] += 1

    num_docs = len(references)
    cider_scores = []
    for refs, hyp in zip(references, hypotheses):
        hyp_tf = defaultdict(int)
        for w in hyp: hyp_tf[w] += 1
        hyp_vec = {w: (cnt/len(hyp)) * np.log((num_docs+1)/(word_to_doc_count[w]+1)) for w,cnt in hyp_tf.items()}

        ref_vecs = []
        for ref in refs:
            ref_tf = defaultdict(int)
            for w in ref: ref_tf[w] += 1
            ref_vec = {w: (cnt/len(ref)) * np.log((num_docs+1)/(word_to_doc_count[w]+1)) for w,cnt in ref_tf.items()}
            ref_vecs.append(ref_vec)

        sim_scores = []
        for ref_vec in ref_vecs:
            common = set(hyp_vec.keys()) & set(ref_vec.keys())
            num = sum(hyp_vec[w]*ref_vec[w] for w in common)
            den = (np.sqrt(sum(v**2 for v in hyp_vec.values())) * 
                   np.sqrt(sum(v**2 for v in ref_vec.values())))
            sim_scores.append(num/den if den > 0 else 0)
        cider_scores.append(np.mean(sim_scores) if sim_scores else 0)
    return np.mean(cider_scores)

def compute_all_metrics(references, hypotheses):
    bleu = calculate_bleu_scores(references, hypotheses)
    rouge = calculate_rouge_scores(references, hypotheses)
    cider = {"CIDEr": calculate_cider(references, hypotheses)}
    return {**bleu, **rouge, **cider}
