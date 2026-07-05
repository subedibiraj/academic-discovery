# Runs the full analysis pipeline end-to-end.
# Usage: python main.py [--quick] [--arxiv]
import subprocess
import sys
import argparse
import time
import shutil
import os

PYTHON = sys.executable

def run(script, args=None, label=""):
    cmd = [PYTHON, script] + (args or [])
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    start = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=".", env=env)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] {label} (exit code {result.returncode})")
        sys.exit(1)
    print(f"\n[OK] {label} ({elapsed:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description="Academic Discovery Pipeline")
    parser.add_argument("--arxiv", action="store_true",
                        help="Include arXiv enrichment steps")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: core pipeline only")
    args = parser.parse_args()

    print("=" * 60)
    print("  ACADEMIC DISCOVERY PIPELINE")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print("=" * 60)

    # Core pipeline
    run("data/merge.py",          label="1. Merge scraped data")
    run("embeddings/embed.py",    label="2. Generate embeddings")
    run("matcher/match.py",       label="3. Rank professors")
    run("matcher/compare.py",     label="4. Compare 6 methods + NDCG/MAP")

    if not args.quick:
        run("data/analyze.py",            label="5. Dataset analysis")
        run("analysis/cluster.py",        label="6. UMAP + KMeans clustering")
        run("analysis/topic_model.py",    label="7. LDA topic modeling")
        run("analysis/explain.py",        label="8. Explainable recommendations")
        run("analysis/ablation.py",       label="9. Ablation study")
        run("analysis/significance.py",   label="10. Bootstrap significance")
        run("analysis/learn_to_rank.py",  label="11. Learning-to-Rank")
        run("analysis/error_analysis.py", label="12. Error analysis")

    # Export UI data
    run("app/export_ui_data.py",  label="13. Export UI data")

    # Copy analytics data to app
    for f in ["lda_topics.json", "ltr_results.json", "ablation_results.json"]:
        src = os.path.join("data", "final", f)
        dst = os.path.join("app", "data", f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [COPY] {f} -> app/data/")

    if args.arxiv:
        run("crawler/arxiv_fetcher.py", label="A1. Fetch arXiv papers")
        run("embeddings/embed.py", args=["--arxiv"],
            label="A2. Re-embed with arXiv")
        run("data/arxiv_analysis.py",   label="A3. arXiv impact analysis")

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print("  Outputs:  data/final/")
    print("  Web app:  app/data/")
    print("  Report:   docs/research_paper.tex")
    print("  Serve:    python app/run.py")

if __name__ == "__main__":
    main()

