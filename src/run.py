"""Run a paper method on question-only input; score saved outputs separately."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import numpy as np
from data_utils import read_jsonl
from evaluation import read
from methods.llm_parser import LLMParser, FIELD_DEFINITIONS
from methods.validated_retrieval_backoff import apply_backoff
from structured_executor import StructuredExecutor
from agents.evidence_canonicalizer import EvidenceCanonicalizer
from agents.query_interpreter import QueryInterpreter
from evidence import EvidenceResources
from retrieval import RetrievalEngine, BM25Index
from build_indexes import load_encoder
from qd_rag import QDRAGLLMParser
import predictors
import adaptive_dev
import feedback

ROOT=Path(__file__).resolve().parents[1]
METHODS=['Random-K','BM25','BGE','RAG-Filter','Direct LLM','LLM-Parser','RAG+QD','Ours','w/o Evidence Agent','w/o Evidence Agent (+ Feedback-only Repair)','w/o Field-aware Retrieval','w/o Re-grounding']

def feedback_repair(row,initial,exact,posts):
    cs=initial['predicted_constraints']
    bad=[{'index':i,**c,'validation_feedback':{'code':'INVALID_CANONICAL','field_valid':c['field'] in FIELD_DEFINITIONS,'message':'The exact canonical string was not found for this field. Reconsider its standard expression without changing the condition.'}} for i,c in enumerate(cs) if (c['field'],c['canonical']) not in exact]
    if not bad:return initial
    payload={'question':row['text'],'field_definitions':FIELD_DEFINITIONS,'initial_constraints':cs,'failed_constraints':bad}
    value,_,_=predictors.qwen_json_call(model='qwen3.7-flash',system_prompt=feedback.PROMPT,user_payload=payload,max_tokens=600,retries=1)
    repaired,answers,errors=feedback.validate_output(value,payload,exact,posts)
    return {'constraints':repaired,'answer_ids':answers,'repair_errors':errors}

def make_retrieval(root,model_path,device,dense):
    rows=read_jsonl(root/'cache/retrieval.jsonl')
    engine=object.__new__(RetrievalEngine)
    engine.rows=rows
    engine.corpus_ids=sorted({r['corpus_id'] for r in rows})
    engine.titles={r['corpus_id']:r['drug_title'] for r in rows}
    engine.bm25=BM25Index(rows)
    if dense:
        import hashlib
        meta=read(root/'cache/retrieval.meta.json')
        if hashlib.sha256((root/'cache/retrieval.jsonl').read_bytes()).hexdigest()!=meta['index_sha256']:raise ValueError('Retrieval index changed')
        engine.model=load_encoder(model_path,device,128)
        engine.embeddings=np.load(root/'cache/retrieval.npy',mmap_mode='r')
        if len(engine.embeddings)!=len(rows):raise ValueError('Retrieval embedding row count differs')
    return engine

class NoRegroundCanonicalizer(EvidenceCanonicalizer):
    def run(self,**kwargs):
        decision,log=super().run(**kwargs)
        if decision['decision']=='REGROUND': decision={'decision':'UNRESOLVED','reason':decision.get('reason','')}
        return decision,log

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--parser-predictions',type=Path,help='Reuse first-pass outputs saved by LLM-Parser')
    parser.add_argument('--rag-predictions',type=Path,help='Reuse outputs saved by RAG-Filter')
    parser.add_argument('--evidence-k',type=int,choices=[1,3,5,10],default=5)
    parser.add_argument('--limit',type=int,help='Optional smoke run; full evaluation requires the entire split')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output path')
    if args.split=='test' and (args.method.startswith('w/o ') or args.evidence_k!=5):raise ValueError('Ablations and retrieval-depth comparisons use Development')
    questions=read_jsonl(ROOT/f'data/questions/{args.split}.jsonl')
    assert all(set(q)=={'_id','text'} for q in questions)
    if args.limit:questions=questions[:args.limit]
    corpus=read_jsonl(ROOT/'data/corpus.jsonl');ids=sorted(r['_id'] for r in corpus)
    titles=defaultdict(set)
    for r in corpus:titles[r['title']].add(r['_id'])
    exact={(c['field'],c['canonical']):c['concept_id'] for c in read(ROOT/'resources/ontology/active_concepts.json')['concepts']}
    executor=StructuredExecutor(ROOT/'resources/ontology/postings.json')
    parser_model=LLMParser(model='qwen3.7-flash',output_root=ROOT)
    cached_parser={r['query_id']:r for r in read(args.parser_predictions)['core_outputs']} if args.parser_predictions else {}
    cached_rag={r['query_id']:r for r in read(args.rag_predictions)['core_outputs']} if args.rag_predictions else {}
    structured=args.method not in ['Random-K','BM25','BGE','RAG-Filter','Direct LLM']
    needs_retrieval=args.method in ['BM25','BGE','RAG-Filter'] or (structured and not args.rag_predictions)
    retrieval=make_retrieval(ROOT,args.model_path,args.device,args.method!='BM25') if needs_retrieval else None
    evidence=None
    if args.method in ['Ours','RAG+QD','w/o Field-aware Retrieval','w/o Re-grounding']:
        evidence=EvidenceResources(ROOT,args.model_path,args.device,global_fields=args.method=='w/o Field-aware Retrieval')
    canonicalizer=(NoRegroundCanonicalizer if args.method=='w/o Re-grounding' else EvidenceCanonicalizer)('qwen3.7-flash',retries=3)
    predictors.EVIDENCE_TOP_K=adaptive_dev.EVIDENCE_TOP_K=args.evidence_k
    qd=QDRAGLLMParser(model='qwen3.7-flash',output_root=ROOT,evidence=evidence,top_k=10) if args.method=='RAG+QD' else None
    predictions=[];outputs=[]
    for row in questions:
        qid=row['_id']
        if args.method=='Random-K':
            seeds={str(seed):sorted(random.Random(seed).sample(ids,600)) for seed in (11,22,33,44,55)}
            result={'answer_ids':seeds['11'],'seed_predictions':seeds}
        elif args.method in ['BM25','BGE']:
            ranking=retrieval.rank_bm25(row['text'],50) if args.method=='BM25' else retrieval.rank_bge(row['text'],100)
            result={'answer_ids':sorted(x['corpus_id'] for x in ranking)}
        elif args.method=='RAG-Filter':result=predictors.rag_predict(row,retrieval.top_chunk_for(retrieval.rank_bge(row['text'],600)))
        elif args.method=='Direct LLM':result=predictors.direct_predict(row,titles)
        elif args.method=='RAG+QD':
            mentions,_=QueryInterpreter('qwen3.7-flash').run(row['text'])
            passages=qd.retrieve_decomposed(row['text'],mentions)
            prepared,_=qd.prepare(row['text'],passages)
            result=predictors.parser_prediction(row,prepared)
        else:
            initial=cached_parser[qid] if args.parser_predictions else predictors.parser_prediction(row,parser_model)
            if args.method in ['LLM-Parser','w/o Evidence Agent']:result=initial
            elif args.method=='w/o Evidence Agent (+ Feedback-only Repair)':result=feedback_repair(row,initial,exact,executor.postings)
            elif args.split=='dev':result,_=adaptive_dev.process_query(dataset_row=row,parser_item=initial,exact_map=exact,executor=executor,evidence=evidence,canonicalizer=canonicalizer)
            else:result=predictors.adaptive_prediction(row,initial,exact,executor,evidence,canonicalizer)
        outputs.append({'query_id':qid,**result})
        final=result
        if structured and not result['answer_ids']:
            rag=cached_rag[qid] if args.rag_predictions else predictors.rag_predict(row,retrieval.top_chunk_for(retrieval.rank_bge(row['text'],600)))
            final=apply_backoff(result,rag,exact,executor.postings)
        predictions.append({'query_id':qid,'answer_ids':final['answer_ids']})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({args.method:predictions},ensure_ascii=False),encoding='utf-8')
        args.output.with_suffix('.core.json').write_text(json.dumps({'core_outputs':outputs},ensure_ascii=False),encoding='utf-8')
        if args.method=='Random-K':
            args.output.with_suffix('.seeds.json').write_text(json.dumps({seed:[{'query_id':r['query_id'],'answer_ids':r['seed_predictions'][seed]} for r in outputs] for seed in seeds}),encoding='utf-8')
        print(f'{len(predictions)}/{len(questions)} {qid}',flush=True)

if __name__=='__main__':main()
