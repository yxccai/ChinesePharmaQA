from __future__ import annotations

import json
from pathlib import Path


class UnknownConceptError(ValueError):
    pass


class StructuredExecutor:
    def __init__(self, postings_path: Path, ontology_path: Path | None = None):
        payload = json.loads(postings_path.read_text(encoding="utf-8"))
        raw_postings = payload.get("postings")
        if not isinstance(raw_postings, dict):
            raise ValueError("postings payload must contain a postings object")
        self.postings = {
            str(concept_id): frozenset(str(corpus_id) for corpus_id in corpus_ids)
            for concept_id, corpus_ids in raw_postings.items()
        }
        self.active_concepts: frozenset[str] | None = None
        if ontology_path is not None:
            ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
            concepts = ontology.get("concepts")
            if not isinstance(concepts, list):
                raise ValueError("ontology payload must contain a concepts list")
            self.active_concepts = frozenset(
                str(item["concept_id"])
                for item in concepts
                if item.get("active") is True
            )

    def execute_constraints(self, concept_ids: list[str]) -> set[str]:
        if not concept_ids:
            raise ValueError("at least one concept ID is required")
        missing = [concept_id for concept_id in concept_ids if concept_id not in self.postings]
        if missing:
            raise UnknownConceptError(f"unknown concept IDs: {missing}")
        if self.active_concepts is not None:
            inactive = [concept_id for concept_id in concept_ids if concept_id not in self.active_concepts]
            if inactive:
                raise UnknownConceptError(f"inactive concept IDs: {inactive}")
        sets = [self.postings[concept_id] for concept_id in concept_ids]
        result = set(sets[0])
        for values in sets[1:]:
            result.intersection_update(values)
        return result


def execute_constraints(
    concept_ids: list[str], postings_path: Path, ontology_path: Path | None = None
) -> set[str]:
    return StructuredExecutor(postings_path, ontology_path).execute_constraints(concept_ids)
