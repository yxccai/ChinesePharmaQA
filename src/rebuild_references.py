"""Rebuild field postings and reference answer sets from the released corpus."""
import argparse
import json
import re
from pathlib import Path
from evaluation import read
from data_utils import read_jsonl, constraints_to_concept_ids
from source_text_matching import build_matching_view, INSPECTION_SECTIONS
from source_boundary_matching import matches_with_boundaries, DISTINCT_NAMES

ROOT=Path(__file__).resolve().parents[1]

def matches(view,field,canonical,rules):
    if field=='reagent':
        patterns=([DISTINCT_NAMES[canonical]] if canonical in DISTINCT_NAMES else [])
        patterns += [re.escape(n) for n in rules['contained_names'].get(canonical,[])]
        for section in INSPECTION_SECTIONS:
            text=view['normalized_sections'].get(section,'')
            blocked=[m.span() for p in patterns for m in re.finditer(p,text)]
            if any(not any(a<=m.start() and m.end()<=b for a,b in blocked) for m in re.finditer(re.escape(canonical),text)):
                return True
        return False
    if field=='instrument' and canonical=='红外分光光度':
        return matches_with_boundaries(view,field,canonical) or any(re.search(rules['infrared_expression'],view['normalized_sections'].get(s,'')) for s in INSPECTION_SECTIONS)
    return matches_with_boundaries(view,field,canonical)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    registry=read(ROOT/'resources/chapter_registry.json')
    rules=read(ROOT/'resources/matching_rules.json')
    views=[build_matching_view(r,registry) for r in read_jsonl(ROOT/'data/corpus.jsonl')]
    frozen=read(ROOT/'resources/ontology/postings.json')['postings']
    posts={key:sorted(v['id'] for v in views if matches(v,*key.split('::',1),rules)) for key in frozen}
    assert posts==frozen,'Reconstructed field postings differ'
    queries=read_jsonl(ROOT/'data/queries.jsonl')
    refs={q['_id']:sorted(set.intersection(*(set(posts[c]) for c in constraints_to_concept_ids(q['constraints'])))) for q in queries}
    assert refs==read(ROOT/'data/reference_sets.json'),'Reference reconstruction differs'
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for name,value in [('postings.json',{'postings':posts}),('reference_sets.json',refs)]:
            (args.output_dir/name).write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
    print('PASS: all field postings and 1,065 reference sets rebuilt from the corpus.')

if __name__=='__main__': main()
