# Source attribution

ChPSetQA is based on Volumes I and II of the 2020 *Pharmacopoeia of the People's Republic of China*. General-chapter references use auxiliary information from Volumes III and IV. Rights in these source publications and their text remain with the respective rights holders; the project license applies to project-authored material.

Dense retrieval uses [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5). LLM experiments use Alibaba Cloud's `qwen3.7-flash`.

The RAG+QD baseline adapts the no-reranker configuration of [Ammann et al. (ACL 2025 Student Research Workshop)](https://aclanthology.org/2025.acl-srw.32/) to canonical-constraint parsing and deterministic answer-set execution.
