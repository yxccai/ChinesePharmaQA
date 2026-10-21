"""Shared retrieval fallback, filtered by validated predicted constraints."""
def validated_constraints(item, exact):
    values = item.get('canonical_outputs')
    if values is None:
        values = item.get('constraints', item.get('predicted_constraints', []))
    return [c for c in values if (c.get('field'), c.get('canonical')) in exact]

def apply_backoff(item, rag_item, exact, postings):
    original = set(item['answer_ids'])
    if original:
        return {'answer_ids': sorted(original), 'backoff_used': False, 'backoff_llm_calls': 0}
    candidates = set(rag_item['answer_ids'])
    for c in validated_constraints(item, exact):
        candidates.intersection_update(postings[exact[(c['field'], c['canonical'])]])
    return {'answer_ids': sorted(candidates), 'backoff_used': True,
            'backoff_llm_calls': rag_item.get('llm_calls', 0)}
