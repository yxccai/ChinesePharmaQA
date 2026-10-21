"""Build the corpus-only retrieval indexes used by the paper."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import retrieval
from data_utils import write_jsonl
from evidence import evidence_representation

ROOT=Path(__file__).resolve().parents[1]

def load_encoder(model_path,device,max_length):
    from sentence_transformers import SentenceTransformer
    config=json.loads((ROOT/'configs/paper.json').read_text(encoding='utf-8'))
    model=SentenceTransformer(str(model_path),device=device,local_files_only=True) if model_path else SentenceTransformer(config['embedding_model'],revision=config['embedding_revision'],device=device)
    model.max_seq_length=max_length
    return model

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--text-only',action='store_true',help='Build text indexes without loading BGE')
    args=parser.parse_args()
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    retrieval.CORPUS_PATH=ROOT/'data/corpus.jsonl'
    retrieval.RETRIEVAL_CORPUS_PATH=cache/'retrieval.jsonl'
    chunks,_=retrieval.build_formal_retrieval_corpus()
    evidence,_=retrieval.build_rows(ROOT)
    write_jsonl(cache/'evidence.jsonl',evidence)
    if not args.text_only:
        model=load_encoder(args.model_path,args.device,128)
        for name,rows,texts,length in [('retrieval',chunks,[r['text'] for r in chunks],128),('evidence',evidence,[evidence_representation(r) for r in evidence],256)]:
            model.max_seq_length=length
            vectors=model.encode(texts,batch_size=args.batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype('float32')
            np.save(cache/f'{name}.npy',vectors)
            meta={'index_sha256':hashlib.sha256((cache/f'{name}.jsonl').read_bytes()).hexdigest(),'rows':len(rows),'max_seq_length':length,'model':json.loads((ROOT/'configs/paper.json').read_text(encoding='utf-8'))['embedding_model'],'revision':'79e7739b6ab944e86d6171e44d24c997fc1e0116'}
            (cache/f'{name}.meta.json').write_text(json.dumps(meta),encoding='utf-8')
    print(f'Built {len(chunks)} retrieval chunks and {len(evidence)} evidence records.')

if __name__=='__main__': main()
