"""Offline tests for published metrics, prompts and cached repair decisions."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluation import read,score
from data_utils import read_jsonl
from methods.llm_parser import LLMParser
from structured_executor import StructuredExecutor
from agents.evidence_canonicalizer import EvidenceCanonicalizer
import predictors
import adaptive_dev

class ReplayEvidence:
    def __init__(self,traces):self.traces=traces
    def retrieve(self,*,mention,feedback=None,**kwargs):
        t=next(t for t in self.traces if t['failed_constraint']['canonical']==mention)
        return t['reground_evidence' if feedback else 'initial_evidence']

class ReplayCanonicalizer:
    def __init__(self,traces):self.traces=traces
    def run(self,*,mention,final=False,**kwargs):
        t=next(t for t in self.traces if t['failed_constraint']['canonical']==mention['text'])
        decision=t['final_decision' if final else 'initial_decision']
        if decision is None:raise ValueError('Recorded model failure')
        return decision,{'raw':json.dumps(decision),'usage':{}}

class ReleaseTests(unittest.TestCase):
    def test_empty_and_extra_answers(self):
        m=score([{'query_id':'a','answer_ids':[]},{'query_id':'b','answer_ids':['x','z']}],{'a':[],'b':['x','y']})
        self.assertEqual((m['P'],m['R'],m['F1'],m['EM']),(0.5,0.5,0.5,0.5))

    def test_parser_payload_excludes_annotations(self):
        calls=[]
        def fake(**kwargs):
            calls.append(kwargs)
            return {'constraints':[{'field':'reagent','canonical':'高氯酸'}]},'',{}
        with patch('methods.llm_parser.qwen_json_call',side_effect=fake):
            model=LLMParser(model='qwen3.7-flash',output_root=ROOT)
            result=model.run('检验过程中需使用高氯酸的药品是哪些？')
        self.assertEqual(set(calls[0]['user_payload']),{'question'})
        self.assertNotIn('硫酸钠',calls[0]['system_prompt'])
        self.assertEqual(result['parse_status'],'OK')

    def test_repair_decisions_reproduce_all_core_answers(self):
        exact={(c['field'],c['canonical']):c['concept_id'] for c in read(ROOT/'resources/ontology/active_concepts.json')['concepts']}
        executor=StructuredExecutor(ROOT/'resources/ontology/postings.json')
        checked=0
        for split in ('dev','test'):
            outputs=read(ROOT/f'results/model_outputs/{split}.json')
            parsers={r['query_id']:r for r in outputs['LLM-Parser']}
            questions={q['_id']:q for q in read_jsonl(ROOT/f'data/questions/{split}.jsonl')}
            ours=outputs['Ours']
            if split=='dev':ours=ours+outputs['w/o Field-aware Retrieval']
            for row in ours:
                qid=row['query_id'];initial=dict(parsers[qid])
                initial['answer_ids']=sorted(executor.execute_constraints(initial['predicted_concept_ids'])) if initial['predicted_concept_ids'] else []
                ev=ReplayEvidence(row.get('evidence_trace',[]))
                can=ReplayCanonicalizer(row.get('evidence_trace',[]))
                if split=='dev':
                    result,_=adaptive_dev.process_query(dataset_row=questions[qid],parser_item=initial,exact_map=exact,executor=executor,evidence=ev,canonicalizer=can)
                else:result=predictors.adaptive_prediction(questions[qid],initial,exact,executor,ev,can)
                expected=executor.execute_constraints(row['predicted_concept_ids']) if row['predicted_concept_ids'] else set()
                self.assertEqual(set(result['answer_ids']),expected,(split,qid))
                self.assertEqual(result['predicted_concept_ids'],row['predicted_concept_ids'],(split,qid))
                checked+=1
        self.assertEqual(checked,640)

if __name__=='__main__':unittest.main()
