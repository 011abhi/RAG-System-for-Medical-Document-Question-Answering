

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import os

matplotlib.use('Agg')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline — Medical QA Evaluation",
    page_icon="",
    layout="wide"
)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = '#1F3A8A'
ORANGE = '#E87B3C'
GREEN  = '#2E8B57'
RED    = '#C0392B'

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Building and Evaluating a RAG System for Medical QA")
st.markdown("""
**Abhinav Sreenivas | H00513628 | MSc Data Science | Heriot-Watt University**

This dashboard presents the experimental results from a controlled evaluation of
18 RAG pipeline configurations across three chunking strategies, three retrieval
methods, and two language models on the PubMedQA benchmark.
""")

st.divider()

# tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview",
    " RQ1 — RAG vs No-RAG",
    " RQ2 & RQ3 — Retrieval & Chunking",
    " RQ4 — Metric Agreement",
    " Explore Results"
])


# TAB 1 — OVERVIEW

with tab1:
    st.header("Experimental Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experimental Conditions", "18")
    col2.metric("Test Instances per Condition", "300")
    col3.metric("Language Models", "2")
    col4.metric("Evaluation Metrics", "3")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pipeline Design")
        st.markdown("""
        **Chunking Strategies**
        - Fixed-size (256 tokens, 128 stride)
        - Sentence-level (NLTK Punkt)
        - Semantic (PubMedBERT, threshold 0.7)

        **Retrieval Methods**
        - BM25 sparse retrieval
        - PubMedBERT dense retrieval
        - Hybrid Reciprocal Rank Fusion (k=60)

        **Language Models**
        - Mistral-7B-Instruct (MLX, 4-bit)
        - BioMistral-7B (llama-cpp, GGUF Q4_K_M)
        """)

    with col2:
        st.subheader("Evaluation Metrics")
        st.markdown("""
        **Exact Match (EM)**
        Measures whether the generated yes/no/maybe label
        matches the PubMedQA gold label.

        **BERTScore**
        Measures semantic similarity between the model's
        generated justification and the gold context passage
        using DistilBERT-base-uncased (num_layers=5).

        **Custom Faithfulness Proxy**
        Measures lexical overlap between question terms and
        retrieved chunk text as a proxy for evidence grounding.
        """)

    st.divider()
    st.subheader("Complete 18-Condition Results")

    try:
        df_m = pd.read_csv("results/grid_summary.csv")
        df_m['model'] = 'Mistral-7B-Instruct'
        df_b = pd.read_csv("results/grid_summary_biomistral.csv")
        df_b['model'] = 'BioMistral-7B'
        df_all = pd.concat([df_m, df_b], ignore_index=True)
        df_all.columns = [c.replace('_pct', ' (%)').replace('_', ' ').title()
                         for c in df_all.columns]
        st.dataframe(df_all, use_container_width=True)
    except FileNotFoundError:
        st.warning("Results CSVs not found. Make sure you are running from ~/Desktop/Desertation/")

# TAB 2 — RQ1

with tab2:
    st.header("RQ1 — Does RAG Reduce Unsupported Model Outputs?")

    col1, col2, col3 = st.columns(3)
    col1.metric("No-RAG Accuracy", "33.7%", help="Mistral baseline without retrieval")
    col2.metric("RAG Accuracy", "41.3%", "+7.6 pp", help="Mistral Fixed+BM25")
    col3.metric("Framing Failure Rate", "73.3%",
                help="No-RAG outputs falsely referencing non-existent source material")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("No-RAG vs RAG Accuracy")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['No RAG\n(Mistral)', 'RAG\nFixed+BM25'],
                      [33.7, 41.3], color=[RED, BLUE],
                      width=0.45, edgecolor='white')
        for bar, val in zip(bars, [33.7, 41.3]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        ax.axhline(y=78, color=GREEN, linestyle='--',
                   linewidth=1.5, label='Human expert (78%)')
        ax.set_ylabel('Exact Match Accuracy (%)')
        ax.set_ylim(0, 90)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Failure Type Breakdown")
        try:
            counts = {'success': 0, 'generation_failure': 0,
                      'retrieval_failure': 0, 'lucky_guess': 0}
            with open('results/grid_fixed_bm25.csv') as f:
                for r in csv.DictReader(f):
                    ft = r.get('failure_type', '')
                    if ft in counts:
                        counts[ft] += 1

            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ['Success', 'Generation\nFailure',
                      'Retrieval\nFailure', 'Lucky\nGuess']
            vals = [counts['success'], counts['generation_failure'],
                    counts['retrieval_failure'], counts['lucky_guess']]
            colours = [GREEN, RED, ORANGE, '#888888']
            bars = ax.bar(labels, vals, color=colours,
                          edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 2,
                        f'{val}\n({100*val/300:.1f}%)',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
            ax.set_ylabel('Number of Instances (n=300)')
            ax.set_ylim(0, 200)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        except FileNotFoundError:
            st.warning("grid_fixed_bm25.csv not found.")

    st.divider()
    st.info("""
    **Key Finding:** 73.3% of no-retrieval outputs falsely referenced non-existent
    source material despite the prompt containing no mention of context. This framing
    failure persisted after prompt revision, suggesting it reflects a learned generation
    pattern rather than a prompt artefact. RAG substantially reduces this failure mode
    by providing real evidence to reference.

    **McNemar's test:** p=0.0192 (statistically significant at p<0.05)
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RQ2 & RQ3
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("RQ2 & RQ3 — Retrieval Method and Chunking Strategy")

    try:
        df_m = pd.read_csv("results/grid_summary.csv")
        df_b = pd.read_csv("results/grid_summary_biomistral.csv")

        chunkers   = ['fixed', 'sentence', 'semantic']
        retrievers = ['bm25', 'dense', 'hybrid']
        labels_c   = ['Fixed-size', 'Sentence-level', 'Semantic']
        labels_r   = ['BM25', 'Dense', 'Hybrid RRF']

        mistral    = {(r['chunker'], r['retriever']): float(r['em_pct'])
                      for _, r in df_m.iterrows()}
        biomistral = {(r['chunker'], r['retriever']): float(r['em_pct'])
                      for _, r in df_b.iterrows()}

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("EM by Chunking Strategy")
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(chunkers))
            w = 0.35
            m_em = [np.mean([mistral[(c, r)] for r in retrievers])
                    for c in chunkers]
            b_em = [np.mean([biomistral[(c, r)] for r in retrievers])
                    for c in chunkers]
            ax.bar(x - w/2, m_em, w, label='Mistral', color=BLUE)
            ax.bar(x + w/2, b_em, w, label='BioMistral', color=ORANGE)
            for i, (mv, bv) in enumerate(zip(m_em, b_em)):
                ax.text(i - w/2, mv + 0.5, f'{mv:.1f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                ax.text(i + w/2, bv + 0.5, f'{bv:.1f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels_c)
            ax.set_ylabel('Exact Match Accuracy (%)')
            ax.set_ylim(0, 75)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("EM by Retrieval Method")
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(retrievers))
            m_r = [np.mean([mistral[(c, r)] for c in chunkers])
                   for r in retrievers]
            b_r = [np.mean([biomistral[(c, r)] for c in chunkers])
                   for r in retrievers]
            ax.bar(x - w/2, m_r, w, label='Mistral', color=BLUE)
            ax.bar(x + w/2, b_r, w, label='BioMistral', color=ORANGE)
            for i, (mv, bv) in enumerate(zip(m_r, b_r)):
                ax.text(i - w/2, mv + 0.5, f'{mv:.1f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                ax.text(i + w/2, bv + 0.5, f'{bv:.1f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels_r)
            ax.set_ylabel('Exact Match Accuracy (%)')
            ax.set_ylim(0, 75)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.subheader("Generation Bottleneck — Retrieval Precision vs EM Accuracy")
        retrieval_m = {(r['chunker'], r['retriever']): float(r['retrieval_pct'])
                       for _, r in df_m.iterrows()}
        conditions = [f'{c[:3].title()} x {r.title()}'
                      for c in chunkers for r in retrievers]
        em_vals  = [mistral[(c, r)] for c in chunkers for r in retrievers]
        ret_vals = [retrieval_m[(c, r)] for c in chunkers for r in retrievers]

        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(conditions))
        w2 = 0.38
        ax.bar(x - w2/2, ret_vals, w2,
               label='Retrieval Precision (%)', color=GREEN, alpha=0.85)
        ax.bar(x + w2/2, em_vals, w2,
               label='Exact Match Accuracy (%)', color=BLUE, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=8, rotation=15)
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

        st.info("""
        **RQ2 Finding:** All three retrieval methods perform within 1.5 percentage
        points of each other on EM. Retrieval method choice has minimal effect on
        downstream accuracy.

        **RQ3 Finding:** Fixed-size chunking outperforms sentence-level and semantic
        alternatives by 6-7 percentage points consistently across all conditions and
        both models, contradicting the pre-registered hypothesis.

        **Generation Bottleneck:** Retrieval precision (92.7-98.3%) far exceeds EM
        accuracy (37.0-61.3%), confirming generation is the primary bottleneck.
        """)

    except FileNotFoundError:
        st.warning("Results CSVs not found.")


with tab4:
    st.header("RQ4 — Do Grounding Metrics and BERTScore Agree?")

    col1, col2, col3 = st.columns(3)
    col1.metric("Spearman ρ (Lexical vs BERT)", "0.167")
    col2.metric("p-value", "0.667")
    col3.metric("Agreement", "Weak / Not significant")

    st.divider()

    try:
        # Load lexical overlap proxy
        faith_data = {}
        with open('results/rq4_faithfulness_fixed.csv') as f:
            for r in csv.DictReader(f):
                faith_data[(r['chunker'], r['retriever'])] = float(r['faithfulness'])

        # Load BERTScore
        bert_data = {}
        with open('results/rq2_rq3_summary.csv') as f:
            for r in csv.DictReader(f):
                bert_data[(r['chunker'], r['retriever'])] = float(r['bertscore'])

        # Load justification grounding scores — average per condition
        import glob
        grounding_data = {}
        for filepath in sorted(glob.glob('results/grounding/grid_*.csv')):
            if 'biomistral' in filepath:
                continue
            scores = []
            chunker_val = retriever_val = None
            with open(filepath) as f:
                for r in csv.DictReader(f):
                    chunker_val = r['chunker']
                    retriever_val = r['retriever']
                    scores.append(float(r['grounding_score']))
            if chunker_val and retriever_val and scores:
                grounding_data[(chunker_val, retriever_val)] = round(
                    sum(scores) / len(scores), 4)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Lexical Overlap vs BERTScore")
            fig, ax = plt.subplots(figsize=(6, 5))
            colours = {
                ('fixed','bm25'): BLUE, ('fixed','dense'): ORANGE,
                ('fixed','hybrid'): GREEN, ('sentence','bm25'): '#8B0000',
                ('sentence','dense'): '#FF6B6B', ('sentence','hybrid'): '#FF9F43',
                ('semantic','bm25'): '#6C5CE7', ('semantic','dense'): '#A29BFE',
                ('semantic','hybrid'): '#74B9FF'
            }
            for (c, r), faith in faith_data.items():
                if (c, r) in bert_data:
                    ax.scatter(faith, bert_data[(c, r)], s=120,
                               color=colours.get((c, r), '#888888'), zorder=5)
                    ax.annotate(f'{c[:3]}\nx{r[:3]}', (faith, bert_data[(c, r)]),
                                textcoords='offset points', xytext=(6, 4), fontsize=8)
            fx = list(faith_data.values())
            fy = [bert_data[k] for k in faith_data if k in bert_data]
            z = np.polyfit(fx, fy, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(fx), max(fx), 100)
            ax.plot(xs, p(xs), '--', color='#888888', alpha=0.6,
                    linewidth=1.5, label='Trend (ρ=0.167, p=0.667)')
            ax.set_xlabel('Lexical Overlap Score')
            ax.set_ylabel('BERTScore F1')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Full RQ4 Results Table")
            rq4_rows = []
            chunkers   = ['fixed', 'sentence', 'semantic']
            retrievers = ['bm25', 'dense', 'hybrid']
            for c in chunkers:
                for r in retrievers:
                    key = (c, r)
                    rq4_rows.append({
                        'Chunker': c.title(),
                        'Retriever': r.upper(),
                        'Lexical Overlap': round(faith_data.get(key, 0), 4),
                        'Grounding Score': round(grounding_data.get(key, 0), 4),
                        'BERTScore': round(bert_data.get(key, 0), 4)
                    })
            df_rq4 = pd.DataFrame(rq4_rows)
            st.dataframe(df_rq4, use_container_width=True)

        st.divider()
        st.subheader("Justification Grounding vs BERTScore")
        st.markdown("""
        | Model | Spearman ρ | p-value | Significant? |
        |---|---|---|---|
        | Mistral-7B-Instruct | 0.879 | 0.002 | Yes |
        | BioMistral-7B | -0.717 | 0.030 | Yes |

        BioMistral shows a **negative** correlation — its justifications are grounded
        in retrieved evidence but lexically distant from the gold context, explaining
        the paradox of higher EM accuracy but lower BERTScore.
        """)

        st.info("""
        **RQ4 Finding:** The lexical overlap proxy and BERTScore show weak,
        non-significant rank correlation (ρ=0.167, p=0.667). The justification
        grounding score reveals divergent patterns between models (ρ=0.879 for
        Mistral, ρ=-0.717 for BioMistral), confirming that no single metric
        fully captures RAG system quality.
        """)

    except FileNotFoundError as e:
        st.warning(f"Results file not found: {e}")

#results 
with tab5:
    st.header("Explore Individual Conditions")

    col1, col2, col3 = st.columns(3)
    model   = col1.selectbox("Model", ["Mistral-7B-Instruct", "BioMistral-7B"])
    chunker = col2.selectbox("Chunker", ["fixed", "sentence", "semantic"])
    retriever = col3.selectbox("Retriever", ["bm25", "dense", "hybrid"])

    filename = f"results/grid_{'biomistral_' if 'Bio' in model else ''}{chunker}_{retriever}.csv"

    try:
        df = pd.read_csv(filename)
        total = len(df)
        correct = int(df['correct'].sum()) if 'correct' in df.columns else 0
        em = round(100 * correct / total, 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Instances", total)
        col2.metric("Correct Answers", correct)
        col3.metric("Exact Match %", f"{em}%")

        if 'failure_type' in df.columns:
            st.subheader("Failure Type Distribution")
            ft_counts = df['failure_type'].value_counts()
            st.bar_chart(ft_counts)

        st.subheader("Sample Outputs")
        n = st.slider("Number of examples to show", 1, 20, 5)
        show_correct = st.checkbox("Show correct answers only", value=False)

        if show_correct and 'correct' in df.columns:
            sample = df[df['correct'] == 1].head(n)
        else:
            sample = df.head(n)

        for _, row in sample.iterrows():
            with st.expander(f"Q: {row.get('question', 'N/A')[:100]}..."):
                st.write(f"**Gold label:** {row.get('gold', 'N/A')}")
                st.write(f"**Model answer:** {row.get('answer', 'N/A')}")
                st.write(f"**Raw output:** {row.get('raw_output', 'N/A')}")
                st.write(f"**Correct:** {row.get('correct', 'N/A')}")
                st.write(f"**Failure type:** {row.get('failure_type', 'N/A')}")
                if 'retrieved_text' in row:
                    st.write(f"**Retrieved text:** {str(row.get('retrieved_text', ''))[:300]}...")

    except FileNotFoundError:
        st.warning(f"File not found: {filename}")


st.divider()
st.caption("Abhinav Sreenivas | H00513628 | MSc Data Science | Heriot-Watt University | 2026")