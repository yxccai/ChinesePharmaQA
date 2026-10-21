"""Verify dataset consistency and reproduce the released result tables offline."""
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from evaluation import read, score, evaluate
from data_utils import read_jsonl, constraints_to_concept_ids
from structured_executor import StructuredExecutor
from methods.validated_retrieval_backoff import apply_backoff

ROOT=Path(__file__).resolve().parents[1]

def main():
    queries=read_jsonl(ROOT/'data/queries.jsonl')
    corpus=read_jsonl(ROOT/'data/corpus.jsonl')
    universe={r['_id'] for r in corpus}
    refs=read(ROOT/'data/reference_sets.json')
    splits=read(ROOT/'data/splits.json')
    assert len(corpus)==len(universe)==5138
    assert len(queries)==len(refs)==1065
    assert Counter(q['split'] for q in queries)=={'train':639,'dev':214,'test':212}
    assert set(refs)=={q['_id'] for q in queries}
    assert len(read_jsonl(ROOT/'data/single_drug_qa.jsonl'))==200
    executor=StructuredExecutor(ROOT/'resources/ontology/postings.json')
    for q in queries:
        assert executor.execute_constraints(constraints_to_concept_ids(q['constraints']))==set(refs[q['_id']])
        assert set(refs[q['_id']])<=universe
    pairs={(qid,cid) for qid,ids in refs.items() for cid in ids}
    with (ROOT/'data/fusion_qrels.tsv').open(encoding='utf-8') as f:
        qrels=list(csv.DictReader(f,delimiter='\t'))
    assert len(qrels)==len(pairs)==56709
    assert pairs=={(r['query-id'],r['corpus-id']) for r in qrels}
    assert sum(not r for r in refs.values())==4
    exact={(c['field'],c['canonical']):c['concept_id'] for c in read(ROOT/'resources/ontology/active_concepts.json')['concepts']}
    for split,ids in splits.items():
        expected={q['_id'] for q in queries if q['split']==split}
        assert set(ids)==expected
        model_rows=read_jsonl(ROOT/f'data/questions/{split}.jsonl')
        assert {r['_id'] for r in model_rows}==expected
        assert all(set(r)=={'_id','text'} for r in model_rows)
        assert {r['_id']:r['text'] for r in model_rows}=={q['_id']:q['text'] for q in queries if q['split']==split}
    for split in ('dev','test'):
        predictions=read(ROOT/f'results/{split}_predictions.json')
        scores=evaluate(predictions,refs,splits[split])
        if split=='test':
            seed_scores=evaluate(read(ROOT/'results/test_random_seed_predictions.json'),refs,splits[split])
            scores['Random-K']={k:mean(s[k] for s in seed_scores.values()) for k in ('P','R','F1','EM')}
        expected=read(ROOT/f'results/{split}_metrics.json')
        for name,result in scores.items():
            for key in ('P','R','F1','EM'):
                assert abs(result[key]-expected[name][key])<1e-12,(split,name,key)
        outputs=read(ROOT/f'results/model_outputs/{split}.json')
        rag={r['query_id']:r for r in predictions.get('RAG-Filter',[])}
        if not rag:
            rag={r['query_id']:r for r in read(ROOT/f'results/{split}_rag_filter.json')}
        for name,rows in outputs.items():
            final={r['query_id']:set(r['answer_ids']) for r in predictions[name]}
            for row in rows:
                cids=row.get('predicted_concept_ids',row.get('concept_ids'))
                if cids is None:
                    cs=row.get('constraints',[])
                    cids=[exact.get((c.get('field'),c.get('canonical'))) for c in cs]
                    if not cids or not all(cids) or len(cids)!=len(set(cids)) or row.get('repair_errors'): cids=[]
                core=sorted(executor.execute_constraints(cids)) if cids else []
                replay=apply_backoff({**row,'answer_ids':core},rag[row['query_id']],exact,executor.postings)
                assert set(replay['answer_ids'])==final[row['query_id']],(split,name,row['query_id'])
    depth=read(ROOT/'results/topk_results.json')
    for k in (1,3,5,10):
        rows=read(ROOT/f'results/topk_{k}_dev_predictions.json')
        actual=evaluate({str(k):rows},refs,splits['dev'])[str(k)]
        for key in ('P','R','F1','EM'): assert abs(actual[key]-depth[str(k)][key])<1e-12
    groups=read(ROOT/'results/answer_size_results.json')
    test_predictions=read(ROOT/'results/test_predictions.json')
    bounds=[(0,0),(1,7),(8,20),(21,50),(51,100),(101,100000)]
    for group,(lo,hi) in zip(groups,bounds):
        ids={qid for qid in splits['test'] if lo<=len(refs[qid])<=hi}
        assert len(ids)==group['n']
        for method in ('LLM-Parser','Ours'):
            actual=score([r for r in test_predictions[method] if r['query_id'] in ids],refs)
            assert actual['EM']==group['results'][method]['EM']
            assert actual['exact']==group['results'][method]['exact']
    print('PASS: 5,138 entries; 1,065 queries; 639/214/212 splits; 56,709 answer pairs.')
    print('PASS: Main results, Development comparisons, answer-size analysis, Top-K results and structured-output replay.')

if __name__=='__main__': main()
