"""
generate_charts.py — Dissertation Chart Generator
Abhinav Sreenivas | H00513628 | Heriot-Watt University
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results/charts', exist_ok=True)

BLUE='#1F3A8A'; ORANGE='#E87B3C'; GREEN='#2E8B57'; RED='#C0392B'; GRAY='#888888'; PURPLE='#6C5CE7'

mistral={}; retrieval_m={}
with open('results/grid_summary.csv') as f:
    for r in csv.DictReader(f):
        k=(r['chunker'],r['retriever']); mistral[k]=float(r['em_pct']); retrieval_m[k]=float(r['retrieval_pct'])

biomistral={}
with open('results/grid_summary_biomistral.csv') as f:
    for r in csv.DictReader(f):
        k=(r['chunker'],r['retriever']); biomistral[k]=float(r['em_pct'])

faith_data={}
with open('results/rq4_faithfulness_fixed.csv') as f:
    for r in csv.DictReader(f):
        k=(r['chunker'],r['retriever']); faith_data[k]=float(r['faithfulness'])

bert_data={}
with open('results/rq2_rq3_summary.csv') as f:
    for r in csv.DictReader(f):
        k=(r['chunker'],r['retriever']); bert_data[k]=float(r['bertscore'])

chunkers=['fixed','sentence','semantic']; retrievers=['bm25','dense','hybrid']
labels_c=['Fixed-size','Sentence-level','Semantic']; labels_r=['BM25','Dense','Hybrid RRF']
w=0.35

# Figure 5.1
fig,ax=plt.subplots(figsize=(9,5))
x=np.arange(len(chunkers))
m_em=[np.mean([mistral[(c,r)] for r in retrievers]) for c in chunkers]
b_em=[np.mean([biomistral[(c,r)] for r in retrievers]) for c in chunkers]
ax.bar(x-w/2,m_em,w,label='Mistral-7B-Instruct',color=BLUE)
ax.bar(x+w/2,b_em,w,label='BioMistral-7B',color=ORANGE)
for i,(mv,bv) in enumerate(zip(m_em,b_em)):
    ax.text(i-w/2,mv+0.5,f'{mv:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
    ax.text(i+w/2,bv+0.5,f'{bv:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels_c,fontsize=11)
ax.set_ylabel('Exact Match Accuracy (%)'); ax.set_title('Figure 5.1: EM Accuracy by Chunking Strategy',fontweight='bold')
ax.set_ylim(0,75); ax.legend(); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig('results/charts/fig5_1_chunking_em.png',dpi=150,bbox_inches='tight'); plt.close()
print('Figure 5.1 saved')

# Figure 5.2
fig,ax=plt.subplots(figsize=(9,5))
x=np.arange(len(retrievers))
m_r=[np.mean([mistral[(c,r)] for c in chunkers]) for r in retrievers]
b_r=[np.mean([biomistral[(c,r)] for c in chunkers]) for r in retrievers]
ax.bar(x-w/2,m_r,w,label='Mistral-7B-Instruct',color=BLUE)
ax.bar(x+w/2,b_r,w,label='BioMistral-7B',color=ORANGE)
for i,(mv,bv) in enumerate(zip(m_r,b_r)):
    ax.text(i-w/2,mv+0.5,f'{mv:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
    ax.text(i+w/2,bv+0.5,f'{bv:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels_r,fontsize=11)
ax.set_ylabel('Exact Match Accuracy (%)'); ax.set_title('Figure 5.2: EM Accuracy by Retrieval Method',fontweight='bold')
ax.set_ylim(0,75); ax.legend(); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig('results/charts/fig5_2_retrieval_em.png',dpi=150,bbox_inches='tight'); plt.close()
print('Figure 5.2 saved')

# Figure 5.3
fig,ax=plt.subplots(figsize=(12,5))
conditions=[f'{c[:3].title()}\nx\n{r.title()}' for c in chunkers for r in retrievers]
em_vals=[mistral[(c,r)] for c in chunkers for r in retrievers]
ret_vals=[retrieval_m[(c,r)] for c in chunkers for r in retrievers]
x=np.arange(len(conditions)); w2=0.38
ax.bar(x-w2/2,ret_vals,w2,label='Retrieval Precision (%)',color=GREEN,alpha=0.85)
ax.bar(x+w2/2,em_vals,w2,label='Exact Match Accuracy (%)',color=BLUE,alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(conditions,fontsize=8)
ax.set_ylabel('Percentage (%)'); ax.set_title('Figure 5.3: Retrieval Precision vs EM Accuracy',fontweight='bold')
ax.set_ylim(0,110); ax.legend(); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig('results/charts/fig5_3_bottleneck.png',dpi=150,bbox_inches='tight'); plt.close()
print('Figure 5.3 saved')

# Figure 5.4
fig,ax=plt.subplots(figsize=(7,6))
colour_map={('fixed','bm25'):BLUE,('fixed','dense'):ORANGE,('fixed','hybrid'):GREEN,
            ('sentence','bm25'):'#8B0000',('sentence','dense'):'#FF6B6B',('sentence','hybrid'):'#FF9F43',
            ('semantic','bm25'):PURPLE,('semantic','dense'):'#A29BFE',('semantic','hybrid'):'#74B9FF'}
for (c,r),faith in faith_data.items():
    if (c,r) in bert_data:
        ax.scatter(faith,bert_data[(c,r)],s=120,color=colour_map.get((c,r),GRAY),zorder=5)
        ax.annotate(f'{c[:3]}\nx{r[:3]}',(faith,bert_data[(c,r)]),textcoords='offset points',xytext=(6,4),fontsize=8)
fx=list(faith_data.values()); fy=[bert_data[k] for k in faith_data if k in bert_data]
z=np.polyfit(fx,fy,1); p=np.poly1d(z); xs=np.linspace(min(fx),max(fx),100)
ax.plot(xs,p(xs),'--',color=GRAY,alpha=0.6,linewidth=1.5,label='Trend (rho=0.167)')
ax.set_xlabel('Faithfulness Score'); ax.set_ylabel('BERTScore F1')
ax.set_title('Figure 5.4: Faithfulness vs BERTScore (RQ4)',fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/charts/fig5_4_scatter_rq4.png',dpi=150,bbox_inches='tight'); plt.close()
print('Figure 5.4 saved')

# Figure 5.5
counts={'success':0,'generation_failure':0,'retrieval_failure':0,'lucky_guess':0}
with open('results/grid_fixed_bm25.csv') as f:
    for r in csv.DictReader(f):
        ft=r.get('failure_type','')
        if ft in counts: counts[ft]+=1
fig,ax=plt.subplots(figsize=(7,5))
labels_ft=['Success','Generation\nFailure','Retrieval\nFailure','Lucky\nGuess']
vals_ft=[counts['success'],counts['generation_failure'],counts['retrieval_failure'],counts['lucky_guess']]
bars=ax.bar(labels_ft,vals_ft,color=[GREEN,RED,ORANGE,GRAY],edgecolor='white',linewidth=1.5)
for bar,val in zip(bars,vals_ft):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{val}\n({100*val/300:.1f}%)',
            ha='center',va='bottom',fontsize=10,fontweight='bold')
ax.set_ylabel('Number of Instances (n=300)'); ax.set_title('Figure 5.5: Failure Type Breakdown\nMistral — Fixed Chunking + BM25',fontweight='bold')
ax.set_ylim(0,200); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig('results/charts/fig5_5_failure_types.png',dpi=150,bbox_inches='tight'); plt.close()
print('Figure 5.5 saved')

# Figure 5.6
fig,ax=plt.subplots(figsize=(7,5))
bars=ax.bar(['No RAG\n(Mistral)','RAG Fixed+BM25\n(Mistral)'],[33.7,41.3],
            color=[RED,BLUE],width=0.45,edgecolor='white',linewidth=1.5)
for bar,val in zip(bars,[33.7,41.3]):
    ax.text(bar.get_x()+bar.get_width()/2,val+0.5,f'{val}%',ha='center',va='bottom',fontsize=13,fontweight='bold')
ax.axhline(y=78,color=GREEN,linestyle='--',linewidth=1.5,label='Human expert (78%)')
ax.set_ylabel('Exact Match Accuracy (%)'); ax.set_title('Figure 5.6: No-Retrieval vs RAG Accuracy (RQ1)',fontweight='bold')
ax.set_ylim(0,90); ax.legend(); ax.grid(axis='y',alpha=0.3)
ax.annotate('73.3% false\nsource attribution',xy=(0,33.7),xytext=(0.3,55),
            arrowprops=dict(arrowstyle='->',color=RED),fontsize=9,color=RED)
plt.tight_layout(); plt.savefig('results/charts/fig5_6_rq1_comparison.png',dpi=150,bbox_inches='tight'); plt.close()
print('Figure 5.6 saved')

print('\nAll 6 charts saved to results/charts/')
# ── Chart 7: Model comparison all 18 conditions ───────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
conditions_18 = [f'{c[:3].title()}\n{r[:3].title()}' 
                 for c in chunkers for r in retrievers]
m_vals = [mistral[(c,r)] for c in chunkers for r in retrievers]
b_vals = [biomistral[(c,r)] for c in chunkers for r in retrievers]
x = np.arange(len(conditions_18))
w3 = 0.38
ax.bar(x - w3/2, m_vals, w3, label='Mistral-7B-Instruct', color=BLUE)
ax.bar(x + w3/2, b_vals, w3, label='BioMistral-7B', color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(conditions_18, fontsize=8)
ax.set_ylabel('Exact Match Accuracy (%)', fontsize=11)
ax.set_title('Figure 5.7: Model Comparison Across All 18 Conditions\n'
             '(BioMistral consistently outperforms Mistral by 18-20 points)',
             fontsize=11, fontweight='bold')
ax.set_ylim(0, 80)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/charts/fig5_7_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure 5.7 saved')

# ── Chart 8: BERTScore vs EM for both models ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
models = ['Mistral-7B-Instruct', 'BioMistral-7B']
avg_em = [
    np.mean(list(mistral.values())),
    np.mean(list(biomistral.values()))
]
avg_bs = [
    np.mean(list(bert_data.values())),
    np.mean([0.6877, 0.6785, 0.6815, 0.6963, 0.6887,
             0.6956, 0.6961, 0.6907, 0.6970])
]
x = np.arange(len(models))
w4 = 0.35
ax.bar(x - w4/2, avg_em, w4, label='Avg Exact Match (%)', color=BLUE)
ax.bar(x + w4/2, [s * 100 for s in avg_bs], w4,
       label='Avg BERTScore F1 (x100)', color=ORANGE)
for i, (em, bs) in enumerate(zip(avg_em, avg_bs)):
    ax.text(i - w4/2, em + 0.5, f'{em:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(i + w4/2, bs * 100 + 0.5, f'{bs:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Figure 5.8: EM Accuracy vs BERTScore by Model\n'
             '(BioMistral higher EM but lower BERTScore than Mistral)',
             fontsize=11, fontweight='bold')
ax.set_ylim(0, 90)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/charts/fig5_8_em_vs_bertscore.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure 5.8 saved')

# ── Chart 9: Faithfulness by retrieval method ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
faith_by_ret = {r: np.mean([faith_data[(c, r)]
                for c in chunkers if (c, r) in faith_data])
                for r in retrievers}
colours_r = [BLUE, ORANGE, GREEN]
bars = ax.bar(labels_r, list(faith_by_ret.values()),
              color=colours_r, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, faith_by_ret.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.005, f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Faithfulness Score', fontsize=11)
ax.set_title('Figure 5.9: Average Faithfulness Score by Retrieval Method\n'
             '(Dense retrieval consistently scores lowest)',
             fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.75)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/charts/fig5_9_faithfulness_retriever.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Figure 5.9 saved')

print('\nAll 3 additional charts saved to results/charts/')
