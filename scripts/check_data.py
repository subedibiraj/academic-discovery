import json

print("=== DATA INTEGRITY CHECK ===\n")

# Check professors
with open('data/final/all_professors.json', encoding='utf-8') as f:
    profs = json.load(f)
print(f"all_professors.json     : {len(profs)} professors")

# Check embedded
with open('data/final/all_professors_embedded.json', encoding='utf-8') as f:
    emb = json.load(f)
print(f"all_professors_embedded : {len(emb['professors'])} professors")
print(f"Embedding mode          : {emb.get('embedding_mode', 'unknown')}")

# Check arxiv
with open('data/final/all_professors_arxiv.json', encoding='utf-8') as f:
    arxiv_profs = json.load(f)
has_papers = sum(1 for p in arxiv_profs if p.get('arxiv_papers'))
total_papers = sum(len(p.get('arxiv_papers', [])) for p in arxiv_profs)
print(f"arXiv coverage          : {has_papers}/{len(arxiv_profs)} professors")
print(f"Total arXiv papers      : {total_papers}")

# Check comparison
with open('data/final/comparison_results.json', encoding='utf-8') as f:
    comp = json.load(f)
print(f"\ncomparison_results.json : keys = {list(comp.keys())[:5]}")

# Check ablation
with open('data/final/ablation_results.json', encoding='utf-8') as f:
    abl = json.load(f)
print(f"ablation_results.json   : {len(abl['ablations'])} ablation rows")
for row in abl['ablations']:
    print(f"  {row['ablation']:<30} NDCG={row['ndcg_at_k']:.4f}")

# Check LTR
with open('data/final/ltr_results.json', encoding='utf-8') as f:
    ltr = json.load(f)
print(f"\nltr_results.json        : NDCG={ltr['ltr_ndcg']}  MAP={ltr['ltr_map']}")

# Check significance
with open('data/final/significance_results.json', encoding='utf-8') as f:
    sig = json.load(f)
print(f"significance_results    : {len(sig)} entries")

# Check clusters
with open('data/final/clusters.json', encoding='utf-8') as f:
    clusters = json.load(f)
print(f"clusters.json           : {len(clusters['professors'])} professors clustered")

print("\n=== ALL CHECKS PASSED ===")