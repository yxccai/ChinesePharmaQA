"""Answer-set scoring for ChPSetQA."""
import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def score(rows, references):
    tp = fp = fn = exact = 0
    ids = [r['query_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prediction IDs')
    for row in rows:
        predicted, gold = set(row['answer_ids']), set(references[row['query_id']])
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        exact += predicted == gold
    return dict(n=len(rows), TP=tp, FP=fp, FN=fn,
                P=tp/(tp+fp) if tp+fp else 0,
                R=tp/(tp+fn) if tp+fn else 0,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                EM=exact/len(rows) if rows else 0, exact=exact)

def evaluate(predictions, references, expected_ids):
    scores = {}
    for name, rows in predictions.items():
        if {r['query_id'] for r in rows} != set(expected_ids):
            raise ValueError(f'{name}: predictions must cover the entire split')
        scores[name] = score(rows, references)
    return scores

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['dev','test'], default='test')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--seeds', type=Path, help='Average a Random-K .seeds.json file')
    args = parser.parse_args()
    refs = read(ROOT/'data/reference_sets.json')
    splits = read(ROOT/'data/splits.json')
    path = args.predictions or ROOT/f'results/{args.split}_predictions.json'
    results = {} if args.seeds else evaluate(read(path), refs, splits[args.split])
    if args.seeds or (args.predictions is None and args.split == 'test'):
        seeds = read(args.seeds or ROOT/'results/test_random_seed_predictions.json')
        seed_scores = evaluate(seeds, refs, splits['test'])
        results['Random-K'] = {k:mean(s[k] for s in seed_scores.values()) for k in ('P','R','F1','EM')}
    print('| Method | Micro P | Micro R | Micro F1 | EM |')
    print('|---|---:|---:|---:|---:|')
    order=['Random-K','BM25','BGE','RAG-Filter','Direct LLM','LLM-Parser','RAG+QD','Ours']
    for name in sorted(results, key=lambda n: order.index(n) if n in order else len(order)):
        print('| '+name+' | '+' | '.join(f'{results[name][k]:.4f}' for k in ('P','R','F1','EM'))+' |')

if __name__ == '__main__':
    main()
