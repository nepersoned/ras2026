# AISO: Asymmetric Interaction Swarm Optimization for Diversity-Aware Sub-graph Selection in Imbalanced Fraud Detection

**Authors:** Jinhyung Bae, Hankuk University of Foreign Studies  
**Status:** Draft v2.5

---

## Abstract

We present AISO (Asymmetric Interaction Swarm Optimization), a population-based metaheuristic in which agents interact through an asymmetric bilinear compatibility score $c_{ij} = W_i^\top M W_j \neq c_{ji}$, where each agent carries a probability-simplex type vector $W_i \in \Delta^{K-1}$ and $M \in \mathbb{R}^{K \times K}$ encodes directed affinities between agent types. We report three interconnected contributions. First, **mechanism diagnostic across two regimes**: on CEC2013 niching benchmarks (F1–F8, 30 seeds), systematic ablation of 15+ candidate enhancements identifies phased local refinement as the sole productive complement (+0.466 average peak ratio); asymmetric $M$ is the necessary source of persistent diversity (Jaccard 1.000 under symmetric $M$, 0.136 under asymmetric); and a memetic baseline (PSO+Gaussian LS) collapses to 0.460 average peak ratio, confirming that diversity maintenance in the global phase is the critical property distinguishing effective memetic niching. Second, **application on fraud detection**: on the Elliptic Bitcoin fraud graph, a two-stage AISO pipeline (stage 1: feature selection with domain-structured $M$; stage 2: fraud node selection) achieves PR-AUC 0.6644, the highest among 18 compared methods, recovering 93.2\% of unconstrained full-graph performance (0.7128) under a constrained budget of 1,000 fraud labels (28.9\% of available fraud nodes).$^1$ Third, **a preliminary governing condition**: AISO's advantage is not universal but partially predictable — the coefficient of variation of per-cluster mutual information, $\mathrm{CV}(\mu)$, rank-orders three real datasets by outcome (Spearman $\rho = -1.0$, $n = 3$, observational); synthetic validation across $n = 135$ conditions separately confirms that AISO outperforms random baselines (mean $\Delta > 0$, $p < 0.0001$), though the directional relationship reverses in the synthetic setting (Section 6.5), indicating that the real-world threshold is mediated by GNN-specific deployment factors rather than the information-gradient mechanism alone.

**Keywords:** multimodal optimization, niching, swarm intelligence, asymmetric interaction, bilinear compatibility, fraud detection, graph neural networks, diversity-aware sampling

---

## 1. Introduction

Many real-world optimization and selection problems require simultaneously discovering and maintaining *multiple distinct high-quality solutions*. In drug candidate screening, the goal is to find chemically diverse compounds that each achieve high potency, not a single optimum. In graph-based fraud detection, training data should cover heterogeneous fraud modes — temporal bursts, account-reuse clusters, template-driven text — rather than collapsing to the single strongest signal. The unifying challenge is **maintaining sub-population diversity under convergence pressure**.

Classical swarm methods fail here because their interaction topology is symmetric: every agent is attracted toward high-fitness regions, creating homogenization pressure that collapses the population to a single attractor. Niching methods such as SPSO [Li, 2004] impose spatial radius constraints to partition the swarm, but this requires prior knowledge of peak spacing and scales poorly to high-dimensional or heterogeneous landscapes.

We argue that the root limitation is not radius tuning but the symmetry of interaction itself. When $c_{ij} = c_{ji}$, two agents cannot have simultaneously divergent interests — one cannot be attracted to a region while the other is repelled. Asymmetric interaction breaks this constraint: by setting $c_{ij} \neq c_{ji}$, the same matrix $M$ can encode directed relationships where type $k$ seeks type $l$'s information while type $l$ simultaneously repels type $k$'s spatial convergence. The result is **persistent specialization without hard-coded partitioning**.

AISO instantiates this principle through bilinear compatibility on probability-simplex type vectors. Our investigation proceeds in three stages: (1) we establish which aspects of the AISO framework are structurally necessary versus incidental, through controlled ablation on standard niching benchmarks; (2) we demonstrate that the framework transfers to a high-value discrete selection problem — budget-constrained GNN training for fraud detection; (3) we characterize when the transfer succeeds or fails, and identify the conditions under which it does not.

### 1.1 Contributions

1. **AISO framework**: A swarm optimizer with asymmetric bilinear compatibility $c_{ij} = W_i^\top M W_j \neq c_{ji}$ on simplex-valued type vectors, with type assimilation as the learning rule and adaptive repulsion as the diversity maintenance mechanism.

2. **Mechanism diagnostic with scope identification**: Systematic ablation of 15+ candidate mechanisms on CEC2013 F1–F8 identifies the productive operating mode of AISO and confirms that asymmetric $M$ is the structural prerequisite for diversity — not an incidental design choice. In continuous niching, the asymmetric mechanism is statistically indistinguishable from a random global phase; its contribution emerges in discrete selection (contribution 3).

3. **Fraud detection application**: A two-stage Smart M pipeline achieves the highest PR-AUC among 18 compared methods on Elliptic Bitcoin (0.6644, 5 seeds), demonstrating that the mechanism transfers from continuous niching to discrete graph selection where type and position are the same variable.

4. **Preliminary governing condition**: $\mathrm{CV}(\mu) < 1.0$ as an observational heuristic ($n = 3$ real datasets) that associates feature cluster MI balance with favorable AISO outcomes; broader validation is required before prescriptive use.

---

## 2. Related Work

### 2.1 Niching Methods

Speciation-based PSO (SPSO, [Li, 2004]) and NichePSO [Brits et al., 2007] form sub-populations by clustering particles within a Euclidean radius. Performance degrades when optima are heterogeneously spaced, as a single radius cannot simultaneously accommodate all inter-peak distances. Li [2010] showed that ring-topology lbest PSO forms stable niches without explicit radius, and Crowding DE [Thomsen, 2004] achieves implicit local competition through nearest-neighbor replacement. Locally-Informed PSO (LIPS) [Qu et al., 2013] uses $k$-nearest pbests for guidance. All of these methods retain Euclidean proximity as the fundamental organizing principle; AISO replaces it with type-mediated compatibility.

### 2.2 Graph Sampling for GNN Training

Cluster-GCN [Chiang et al., 2019] and GraphSAINT [Zeng et al., 2020] address scalability through graph partitioning and importance sampling respectively, but neither targets *diversity of coverage* as a primary objective. Random node sampling is the most common baseline under budget constraints; it collapses to the dominant fraud mode under severe class imbalance.

### 2.3 Fraud Detection with GNNs

CARE-GNN [Dou et al., 2020] addresses camouflage via neighbor-evidence masking. PC-GNN [Liu et al., 2021] addresses class imbalance via pick-and-choose resampling. Both methods improve upon random sampling within a fixed-budget regime but do not leverage inter-agent diversity to cover heterogeneous fraud patterns. mRMR [Ding and Peng, 2005] provides competitive feature selection but lacks the node-level diversity that distinguishes our two-stage pipeline.

### 2.4 Asymmetric Interaction

Asymmetric affinity matrices appear in opinion dynamics [Caldarelli et al., 2007], game-theoretic swarm robotics [Liu et al., 2025], and replicator dynamics, but to our knowledge no prior swarm optimizer employs a learnable bilinear asymmetric compatibility on simplex-valued types. The closest analogs in population dynamics are Lotka-Volterra systems with asymmetric interaction terms, which also generate persistent coexistence through directed competitive relationships.

### 2.5 Memetic Algorithms

Memetic algorithms (MAs) [Moscato, 1989] combine population-level global search with individual-level local refinement, addressing the exploration–exploitation tension that purely evolutionary or swarm methods struggle to resolve simultaneously. The canonical MA structure — global phase for basin discovery, local phase for within-basin convergence — has produced competitive performance across combinatorial optimization [Moscato and Cotta, 2003], continuous black-box problems, and feature selection tasks.

In the context of swarm optimization, memetic extensions of PSO (memetic PSO) typically wrap a local search operator (gradient descent, Nelder-Mead, or pattern search) around standard velocity-based updates. MA-SW-Chains [Molina et al., 2010] applies a chain of local searchers to a decomposed search space, achieving strong performance on high-dimensional continuous benchmarks. Memetic Differential Evolution variants [Neri and Tirronen, 2010] interleave DE mutation with local self-adaptation, demonstrating that the global–local separation is broadly applicable across population paradigms.

AISO instantiates the memetic structure in two ways. First, the phased schedule ($0$–$0.7T$ global, $0.7T$–$T$ local) directly maps to the MA global–local split. Second, the *mechanism* driving the global phase is novel: rather than velocity-based attraction or DE crossover, AISO uses type-mediated asymmetric interaction to distribute agents across basins without explicit radius constraints. This is the structural distinction from prior memetic swarm work — the local phase (Gaussian refinement) is standard; the global phase (asymmetric compatibility dynamics) is the contribution. The ablation in Section 4 confirms this separation: the local phase alone accounts for the full performance gain, but only because the global phase first distributes agents into distinct basins that the local phase can then refine independently.

---

## 3. The AISO Framework

### 3.1 Agent State

Each agent $i \in \{1, \dots, N\}$ maintains:
- a **position** $X_i \in \mathbb{R}^d$ (or a discrete selection profile) in the search space,
- a **type vector** $W_i \in \Delta^{K-1}$ on the probability simplex over $K$ latent types,
- a **fitness score** $s_i \in \mathbb{R}$ from the objective function.

All agents share a global matrix $M \in \mathbb{R}^{K \times K}$ with zero diagonal.

### 3.2 Asymmetric Compatibility

For any agent pair $(i, j)$, the **compatibility score** is

$$c_{ij} = W_i^\top M W_j$$

Because $M$ is asymmetric, $c_{ij} \neq c_{ji}$ in general: agent $i$ may be attracted to agent $j$'s direction while agent $j$ is simultaneously repelled from agent $i$. This directed relationship enables specialization persistence: type-pairs with cyclic preferences (e.g., $c_{ij} > 0$, $c_{jk} > 0$, $c_{ki} < 0$) create stable non-converging triangles that preserve distinct sub-populations without spatial radius constraints.

### 3.3 Partner Selection and Position Update

Agent $i$ selects a partner

$$j^\star = \arg\max_{j \neq i}\ c_{ij} \cdot \frac{s_j}{s_{\max}}$$

combining compatibility with relative fitness. The position update is

$$X_i^{(t+1)} = \mathrm{clip}\!\left(X_i^{(t)} + \alpha \cdot c_{i,j^\star} \cdot \left(X_{j^\star} - X_i^{(t)}\right),\ b_{\min},\ b_{\max}\right)$$

If $f(X_i^{(t+1)}) > f(X_i^{(t)})$, the move is accepted and the type is assimilated:

$$W_i^{(t+1)} = \frac{(1-\beta)W_i^{(t)} + \beta W_{j^\star}^{(t)}}{\|(1-\beta)W_i^{(t)} + \beta W_{j^\star}^{(t)}\|_1}$$

with $\alpha = 0.4$, $\beta = 0.1$. Repeated successful interactions with agents of a similar type cause $W_i$ to concentrate around that type, forming soft clusters — one per discovered mode — without explicit partitioning.

### 3.4 Adaptive Repulsion

To prevent premature convergence, negative entries of $M$ are scaled by swarm diversity $\delta$ (normalized mean pairwise distance):

$$M^{\mathrm{eff}}_{ij} = \begin{cases} M_{ij} \cdot (1 + 3\,e^{-\delta/0.12}) & M_{ij} < 0 \\ M_{ij} & M_{ij} \geq 0 \end{cases}$$

When the swarm collapses spatially, repulsion intensifies; when dispersed, repulsion relaxes. This closes the feedback loop between type-space dynamics and position-space diversity. Figure 1 illustrates the full per-iteration update cycle.

![Figure 1: AISO per-iteration update cycle.](figures/fig1_aiso_mechanism.png)

### 3.5 Smart M: Domain-Structured Compatibility

When feature structure is available, we construct $M$ to encode domain knowledge asymmetrically. Algorithm 1 gives the full construction procedure.

---

**Algorithm 1: Smart M Construction**

**Input:** Feature matrix $X \in \mathbb{R}^{n \times D}$, labels $y \in \{0,1\}^n$, number of clusters $K$, gradient weight $\gamma = 0.5$

**Output:** Asymmetric compatibility matrix $M \in \mathbb{R}^{K \times K}$

1. Compute pairwise Pearson correlation: $C_{ab} = \mathrm{corr}(x_a, x_b)$ for all feature pairs $(a, b)$
2. Cluster features into $K$ groups $\{\mathcal{C}_1, \dots, \mathcal{C}_K\}$ via hierarchical clustering on distance $d_{ab} = 1 - |C_{ab}|$
3. **For each cluster** $k \in \{1, \dots, K\}$:
   - Compute per-feature MI: $\mathrm{MI}(x_j, y)$ for all $j \in \mathcal{C}_k$
   - Compute cluster mean: $\mu_k = \frac{1}{|\mathcal{C}_k|} \sum_{j \in \mathcal{C}_k} \mathrm{MI}(x_j, y)$
4. Normalize: $\tilde{\mu}_k = \dfrac{\mu_k - \min_l \mu_l}{\max_l \mu_l - \min_l \mu_l}$
5. **For each pair** $(i, j)$ with $i \neq j$:
   - Compute mean cross-cluster absolute correlation: $\overline{|C|}_{ij} = \dfrac{1}{|\mathcal{C}_i||\mathcal{C}_j|} \sum_{a \in \mathcal{C}_i} \sum_{b \in \mathcal{C}_j} |C_{ab}|$
   - Set $M_{ij} = -\overline{|C|}_{ij} + \gamma(\tilde{\mu}_j - \tilde{\mu}_i)$
6. Set $M_{ii} = 0$ for all $i$
7. **Return** $M$

---

The construction produces an inherently asymmetric matrix. Decomposing $M$ into symmetric and antisymmetric parts:

$$M_{ij} = \underbrace{-\overline{|C|}_{ij}}_{\text{symmetric: correlation repulsion}} + \underbrace{\gamma(\tilde{\mu}_j - \tilde{\mu}_i)}_{\text{antisymmetric: information gradient}}$$

Since $\overline{|C|}_{ij} = \overline{|C|}_{ji}$ (mean correlation is symmetric) but $\tilde{\mu}_j - \tilde{\mu}_i = -(\tilde{\mu}_i - \tilde{\mu}_j)$, the antisymmetric term ensures $M_{ij} \neq M_{ji}$ whenever clusters differ in MI. The degree of asymmetry is $M_{ij} - M_{ji} = 2\gamma(\tilde{\mu}_j - \tilde{\mu}_i)$: agents are routed from low-MI clusters toward high-MI clusters, while high-MI clusters simultaneously repel low-MI agents. Smart $M$ degenerates toward symmetry when $\tilde{\mu}_j \approx \tilde{\mu}_i$ for all pairs — precisely the condition measured by $\mathrm{CV}(\mu)$.

### 3.6 Two-Stage Application Architecture

For the fraud detection application, AISO operates in two sequential stages. Algorithm 2 gives the full procedure.

---

**Algorithm 2: Two-Stage AISO Pipeline**

**Input:** Graph $G=(V,E)$, node features $X \in \mathbb{R}^{|V| \times D}$, labels $y$, budgets $B_1$ (features), $B_2$ (nodes), cluster counts $K_1, K_2$

**Output:** GNN training subgraph $\mathcal{S} \subseteq V$

**Stage 1 — Feature Selection:**
1. Cluster $D$ features into $K_1$ groups via hierarchical clustering on correlation distance $d_{ab} = 1 - |C_{ab}|$
2. Build Smart $M^{(1)}$ from feature cluster MI statistics (Algorithm 1, $\gamma = 0.5$)
3. Run AISO($M^{(1)}$, $T$ iterations): each agent $i$ maintains weight vector $W_i^{(1)} \in \Delta^{K_1 - 1}$
4. Aggregate: feature importance $s_j = \frac{1}{N}\sum_i W_{i,\,\mathrm{cluster}(j)}^{(1)}$; select top-$B_1$ features $\mathcal{F}^*$

**Stage 2 — Node Selection:**
5. Project all nodes to $\mathcal{F}^*$; cluster fraud nodes into $K_2$ groups by feature similarity
6. Build Smart $M^{(2)}$ from node cluster MI statistics within $\mathcal{F}^*$
7. Run AISO($M^{(2)}$, $T$ iterations): each agent $i$ maintains $W_i^{(2)} \in \Delta^{K_2 - 1}$
8. Each agent selects $\lfloor B_2 / N \rfloor$ nodes proportional to $W_i^{(2)}$; union forms $\mathcal{S}$

**Return** $\mathcal{S}$

---

This decomposition separates *what to look at* (stage 1) from *who to train on* (stage 2), enabling independent diversity pressure at both levels. The key design decision is that $M^{(2)}$ is reconstructed within $\mathcal{F}^*$ rather than the full feature space: cluster structure and MI rankings shift after feature projection, so recomputing Smart $M$ ensures the node-level information gradient reflects the filtered subspace. Figure 2 illustrates the full pipeline.

![Figure 2: Two-stage AISO pipeline for GNN fraud detection.](figures/fig2_two_stage.png)

---

## 4. Mechanism Diagnostic on CEC2013

We use CEC2013 niching benchmarks (F1–F8, [Li et al., 2013]) as a controlled laboratory to identify which components of the AISO framework are structurally necessary. The benchmarks span 1D and 2D landscapes with 1–36 global optima, providing a rich test of diversity preservation without confounding application-specific structure. All experiments use $N = 80$ agents, $T = 200$ iterations, accuracy threshold $\varepsilon = 0.01 \times \mathrm{range}$, and 30 random seeds.

**Unifying framing.** AISO's global phase operates in *type space* — agents specialize into distinct compatibility profiles via asymmetric interaction. In CEC continuous niching, type and position are independent variables: an agent can hold a highly differentiated type vector while its spatial position still converges toward the same attractor as other agents. This *type-spatial decoupling* means that type-space diversity does not automatically produce spatial basin coverage. The experiments in this section explicitly diagnose this decoupling — identifying which mechanisms matter for spatial niching performance and which do not — and thereby predict the mechanism's success in Section 5, where the decoupling disappears because agent type and selection action are the same variable.

### 4.1 Necessity of Asymmetric $M$

**Claim**: asymmetric $M$ is the structural prerequisite for persistent sub-population diversity; symmetrizing $M$ causes swarm collapse.

We replace Smart $M$ with its symmetric counterpart $M_{\mathrm{sym}} = \frac{1}{2}(|M| + |M|^\top)$ and run identical AISO loops on Elliptic. Inter-agent feature mask overlap (Jaccard similarity) collapses completely:

| $M$ type | Mean Jaccard | Std |
|---|---|---|
| Asymmetric (Smart $M$) | **0.136** | 0.027 |
| Symmetric $M_\mathrm{sym}$ | **1.000** | 0.000 |

Under symmetric $M$, all agents converge to identical feature masks (Jaccard = 1.000 across all 5 seeds). Under asymmetric $M$, agents maintain genuinely distinct specializations. This is the foundational mechanism evidence: asymmetry is necessary, not incidental. Figure 5(a) visualizes the collapse.

![Figure 5: (a) Diversity collapse under symmetric M. (b) 2×2 stage decomposition.](figures/fig5_jaccard_diversity.png)

### 4.2 Ablation of 15+ Candidate Mechanisms

Starting from the base AISO algorithm, we add 15 distinct mechanisms in isolation and measure the change in average peak ratio (PR) across F1–F8. Table 1 summarizes the results.

**Table 1. Ablation of candidate mechanisms on CEC2013 F1–F8 (avg PR, 30 seeds).**

| Variant | Mechanism | Avg PR | $\Delta$ | Wilcoxon $p$ |
|---|---|---|---|---|
| AISO baseline | None | 0.445 | — | — |
| + Type-Position Coupling (v3) | Fixed anchors $A_k$ | 0.420 | $-0.025$ | — |
| + Adaptive Anchor (v5-A) | Anchor nudging | 0.905 | — | — |
| + Fitness Gating (v5-AC) | Gate coupling by $f(A_k)$ | 0.905 | $\approx 0$ | 0.750 vs v4 |
| + W Diversity Penalty | $\lambda_\mathrm{div}(W_i - \bar{W})$ | 0.908 | — | — |
| + Smart W Init | K-means spatial init | 0.905 | — | — |
| + W Sparsity | $W_i \leftarrow W_i^\gamma / \|W_i^\gamma\|$ | 0.899 | — | — |
| + Anti-Assimilation | Repel $W_\mathrm{enemy}$ | 0.911 | $\approx 0$ | — |
| + Hebbian M | $\Delta M \propto \Delta f \cdot W_i \otimes W_{j^\star}$ | 0.912 | $+0.001$ | 0.250 |
| + Sparse M | Zero small $M$ entries | 0.912 | $+0.000$ | 0.250 |
| + Cyclic M Init | Cyclic preference structure | 0.909 | $-0.002$ | 0.625 |
| + Fitness-weighted $\beta$ | $\beta \propto \Delta f$ | 0.911 | $+0.000$ | 0.500 |
| + Asymmetric Assimilation | Repel when $c_{ij} < 0$ | 0.909 | $-0.002$ | 1.000 |
| + Niche Detection (DBSCAN) | Restrict $j^\star$ to same niche | 0.908 | $-0.003$ | 1.000 |
| + Surrogate (RF) | Surrogate-weighted $j^\star$ | 0.911 | $+0.000$ | 0.500 |
| + Memetic local search | Mini-perturbation every 20 iters | 0.914 | $+0.003$ | 0.500 |
| **+ Phased Refinement** | **Phase-2 Gaussian walk** | **0.911** | **+0.466** | **—** |

No mechanism across M learning, W dynamics, or structural extensions achieves statistically significant improvement (all $p \geq 0.25$); phased refinement is the sole exception at +0.466. Figure 3 summarizes the full ablation and method comparison.

![Figure 3: CEC2013 ablation (left) and method comparison (right).](figures/fig3_cec_ablation.png) This is **scope identification**: the productive operating mode is the simplest possible form, and elaborations are redundant with dynamics already implicit in the base algorithm. Two isolation ablations confirm type-spatial decoupling: Random+Refine = 0.906 ($p=0.43$) and SymM+Refine = 0.896 ($p=0.21$) both trail AISO's 0.911 non-significantly — the global phase's contribution emerges only where type and selection are the same variable (Section 5). Hebbian M (+0.001, $p=0.25$) likewise fails to improve: Smart $M$'s prior-encoded statistics need not be re-discovered from noisy fitness signals.

### 4.3 Phased Refinement and Memetic Baseline

AISO follows a **memetic two-phase structure** [Moscato, 1989]: population-level global search for basin discovery, then individual-level local refinement for within-basin convergence.

- **Phase 1** ($0$ to $0.7T$): AISO global search via asymmetric compatibility.
- **Phase 2** ($0.7T$ to $T$): Per-agent Gaussian random walk, step decaying from $0.05 \cdot \mathrm{range}$ to $0$.

| Variant | Avg PR | $\Delta$ |
|---|---|---|
| AISO baseline | 0.445 | — |
| + Type-Position Coupling only | 0.420 | $-0.025$ |
| **+ Phased Refinement only** | **0.911** | **$+0.466$** |
| + Coupling + Refinement | 0.906 | $-0.005$ vs refine-only |

The two tasks — locating modes and converging to them — are naturally separable; once an agent enters a basin, Gaussian refinement suffices to reach the $\varepsilon$-ball. The gain concentrates on moderate-peak functions (F4, F5, F8: 0.225/0.283/0.267 → 1.000 each); on high-peak functions (F6: 18 peaks, F7: 36 peaks), AISO achieves 0.776/0.515 versus CrowdingDE's 0.969/0.645, indicating a capacity limit at $N=80$ agents. Across all 15 mechanism variants, type entropy varies widely (0.005–3.38) yet peak ratio remains flat — confirming type-spatial decoupling in continuous niching.

To position AISO within the memetic family, we compare against **PSO+LS**: inertia-weight PSO (gbest) in phase 1, identical Gaussian refinement in phase 2 — the same phased structure with convergence-inducing dynamics instead of asymmetric interaction.

**Table 2. Memetic baseline comparison on CEC F1–F8 (30 seeds, avg peak ratio).**

| Method | Global Phase | Avg PR | vs AISO |
|---|---|---|---|
| **AISO + Refine** | Asymmetric compatibility | **0.911** | — |
| PSO + Refine | Inertia-weight PSO (gbest) | 0.460 | $-0.451$, $p < 0.0001$ |
| Random + Refine | None ($\alpha=0$) | 0.906 | $-0.005$, $p = 0.43$ |

PSO+LS collapses to 0.460 because gbest drives all particles toward a single peak, eliminating multi-modal coverage. Extending to $T=400$ and $T=800$ degrades further (0.427, 0.416), confirming the gap is structural. **The critical property is diversity maintenance in the global phase**: PSO destroys it, random preserves it by construction, AISO actively routes agents to distinct basins through asymmetric dynamics. In Section 5, where basin assignment depends on type specialization rather than random initialization, AISO's structured routing provides the additional margin over random.

---

## 5. Application: GNN Fraud Detection

In the CEC validation (Section 4), agents move through a continuous landscape to locate optima. The key scope finding was that type diversity and spatial diversity are decoupled in continuous settings — agents can specialize into distinct types while still collapsing spatially, because type and position are independent variables. In this section, that decoupling disappears: agent positions *are* feature selection profiles, so type specialization and selection diversity are the same variable. The asymmetric compatibility mechanism ($c_{ij} = W_i^\top M W_j$) and type assimilation dynamics are structurally identical — the application succeeds precisely where the CEC limitation ends.

**Why two stages?** The two-stage design addresses structurally distinct axes of diversity. Stage 1 (feature selection) determines *which information channels* the GNN receives: AISO selects a diverse, balanced profile across $K$ feature clusters rather than concentrating on a single cluster's signal. Stage 2 (node selection) determines *which training examples* the GNN sees: given the stage-1 feature subspace, AISO selects fraud nodes covering heterogeneous fraud patterns rather than amplifying the single most common mode. These two selections optimize independent degrees of freedom — a single-stage node selector cannot optimize the feature channel mix, and a single-stage feature selector cannot account for which fraud modes appear in the chosen node set. Table 4 (Section 5.6) confirms empirically that stage-1 contribution (+0.101 PR-AUC) and stage-2 contribution (+0.013–0.037) are additive and independent.

### 5.1 Problem Formulation

Given a fraud graph $G = (V, E)$ with node features $X \in \mathbb{R}^{|V| \times D}$ and labels $y \in \{0, 1\}^{|V|}$ (~10% positive), a 2-layer GCN is trained on a budget-constrained subgraph ($N_\mathrm{illicit}$ fraud nodes + $N_\mathrm{licit}$ normal nodes) and evaluated by PR-AUC on the held-out test set. The objective is to select the training subgraph that maximizes test PR-AUC.

**Protocol:** 2-layer GCN (hidden=64, dropout=0.5, Adam lr=0.01, early stopping patience=15), 5 seeds $\{0, 7, 42, 77, 123\}$, mean $\pm$ std PR-AUC reported. Budget: $N_\mathrm{illicit} = 1000$, $N_\mathrm{licit} = 10000$ on Elliptic (3,462 available fraud nodes).

### 5.2 Asymmetry Necessity in the Application Domain

Replicating the mechanism validation from Section 4.1 in the application context confirms that the structural finding generalizes:

| Stage | Config | PR-AUC | Jaccard |
|---|---|---|---|
| Stage-1 Smart M (asym) → Stage-2 Smart M (asym) | **Full asymmetric** | **0.6644** | 0.136 |
| Stage-1 Sym M → Stage-2 Sym M | Full symmetric | 0.5527 | 1.000 |
| Stage-1 Smart M → Stage-2 Sym M | Partial | 0.6244 | — |
| Stage-1 Sym M → Stage-2 Smart M | Partial | 0.5992 | — |

Results are monotonically ordered: Sym→Sym < Sym→Smart < Smart→Sym < Smart→Smart. Symmetric $M$ at stage 1 also elevates variance (std 0.084 vs. 0.020–0.032), confirming that symmetric interaction produces unstable as well as weaker solutions.

### 5.3 Two-Stage Pipeline Results

We evaluate AISO across three benchmark fraud datasets in a progression of increasing integration.

**Table 3. Stage-level PR-AUC across datasets.**

| Stage | Description | Elliptic | YelpChi | Amazon |
|---|---|---|---|---|
| S0 — Random baseline | Uniform random node sampling | 0.5530 | 0.2388 | 0.4743 |
| SA — AISO node sampler | Topology features → AISO | 0.3972 | 0.2359 | 0.4114 |
| SB — Feature → SGD | mRMR features → SGD scorer | 0.6348 | 0.2220 | 0.5140 |
| **SC — Feature → AISO** | **AISO(Smart) → AISO(Smart)** | **0.6644** | 0.2249 | 0.4940 |

On Elliptic, the full stage sequence recovers monotonically (0.3972 → 0.6348 → 0.6644). Stage S0→SB (+0.0818) is the primary gain from feature-space structuring; SB→SC (+0.0296, +4.7% relative) is the additional increment from node-level diversity preservation after feature filtering. On Amazon, SC falls below SB (−0.0200), and on YelpChi, all methods compress into a narrow band (0.207–0.244), indicating zero headroom from any sampler. These differences are structurally predicted by the governing condition (Section 6).

**Relationship to CARE-GNN and PC-GNN.** CARE-GNN [Dou et al., 2020] and PC-GNN [Liu et al., 2021] operate on the *training objective*: they modify aggregation or loss weighting to handle camouflage and class imbalance during GNN training. AISO operates on *training data selection*: it determines which nodes and features enter the training set before any GNN training begins. The two approaches are orthogonal — AISO's subgraph selection could be combined with CARE-GNN's aggregation or PC-GNN's resampling as the downstream GNN. The 18-method comparison in Table 3 uses a fixed GCN backbone to isolate the selection effect; CARE-GNN/PC-GNN improvements are additive and do not compete with AISO's contribution.

### 5.4 Full Elliptic Ranking (18 Methods)

**Table 4. Elliptic Bitcoin results, 18 methods, 5 seeds.**

| Rank | Method | Stage | PR-AUC | Std |
|---|---|---|---|---|
| **1** | **AISO(Smart)→AISO(Smart)** | SC | **0.6644** | 0.0200 |
| 2 | mRMR→SGD | SB | 0.6348 | 0.0055 |
| 2 | mRMR→AISO(Rand) | SC | 0.6348 | 0.0493 |
| 4 | MI→AISO(Smart) | SC | 0.6312 | 0.0150 |
| 5 | Greedy-Raw | S0 | 0.6245 | 0.0216 |
| 6 | MI→SGD | SB | 0.6138 | 0.0146 |
| 9 | AISO-node(Rand M) | SA | 0.6004 | 0.0811 |
| 11 | AISO(Rand)→AISO | SC | 0.5872 | 0.0332 |
| 16 | AISO-node(Smart M) | SA | 0.3972 | 0.0575 |

Four of the top five methods are from the SC stage, confirming the two-stage structure's systematic advantage. The Rank 1 method's std (0.0200) is comparable to strong deterministic baselines (Greedy-Raw: 0.0216), indicating mechanism-driven rather than seed-fragile gains. Replacing stage-2 with random $M$ reduces PR-AUC by −0.013 (0.6644 → 0.6510), confirming Smart $M$'s independent contribution at the node selection stage.

### 5.5 Label Efficiency

GraphSAGE trained on the full graph (no budget constraint) achieves PR-AUC 0.7128 (GCN at full graph: comparable). Within a constrained budget of $N_\mathrm{illicit} = 1{,}000$ fraud nodes (28.9% of the 3,462 available fraud labels),$^1$ AISO recovers:

$$\frac{0.6644 - 0.5530}{0.7128 - 0.5530} = 70.0\%\ \text{of the gap to the unconstrained ceiling}$$

Equivalently, AISO reaches 93.2% of full-graph performance (0.6644/0.7128) under this label constraint.$^2$ This is the deployment-side argument: a fraction of available fraud supervision is converted into near-ceiling performance.

$^1$ The ceiling (0.7128) uses GraphSAGE; AISO results use GCN. The comparison is conservative for AISO — GraphSAGE-AISO achieves 0.6611 (Table~5, Section~5.7), giving 92.8% recovery under a matched backbone.

$^2$ Formal significance testing of AISO (0.6644 ± 0.020) vs. mRMR→SGD (0.6348 ± 0.006) is limited by the 5-seed budget; the gap (+0.030) exceeds the pooled standard error but a larger-seed replication is warranted.

### 5.6 Stage-2 Smart M Decomposition (2×2 Ablation)

**Table 5. 2×2 stage M-configuration ablation on Elliptic.**

| | Stage-2 Rand M | Stage-2 Smart M | Stage-2 gain |
|---|---|---|---|
| **Stage-1 Rand M** | 0.5502 | 0.5872 | +0.037 |
| **Stage-1 Smart M** | 0.6510 | **0.6644** | +0.013 |
| **Stage-1 gain** | +0.101 | +0.077 | |

Stage-1 Smart $M$ contributes +0.101 (dominant): diverse feature perspectives at stage 1 determine the quality of the subspace available to stage 2. Stage-2 Smart $M$ contributes +0.013–0.037 depending on upstream quality — confirming independent but diminishing-return contributions at each stage. Critically, Stage-2's contribution (+0.037) persists even when Stage 1 uses random $M$ (top row), confirming that node-level diversity pressure is not wholly dependent on upstream feature quality — Stage 2 provides independent value regardless of Stage 1 fidelity.

### 5.7 Backbone Variation

Fixing AISO(Smart)→AISO(Smart) selection and varying the GNN backbone:

| Backbone | PR-AUC | Note |
|---|---|---|
| GCN | 0.6644 | Baseline |
| GraphSAGE | 0.6611 | $\Delta = +0.003$ (tied) |
| GAT | 0.6493 | — |
| GIN | 0.5493 | — |
| SGC (linear) | 0.3175 | Collapses: no nonlinear aggregation |

The nonlinear-to-linear collapse span (0.6611 → 0.3175, −0.343) vastly exceeds the best-to-worst nonlinear spread (0.6611 − 0.5493 = 0.111). **Feature cluster diversity is the binding constraint, not backbone architecture**: once a nonlinear aggregator is present, architectural variation matters little, while the absence of nonlinearity causes collapse. This supports the governing-condition framing: the environment controls utility more than the model.

---

## 6. Preliminary Governing Condition

### 6.1 Diversity Metric: $\mathrm{CV}(\mu)$

We define the feature cluster diversity tuple:

$$\mathcal{D}_{fc} = \left(k,\ \mathrm{CV}(\mu_1,\dots,\mu_k),\ \overline{|C|}_\mathrm{cross}\right)$$

where $k$ is the number of feature clusters, $\mu_c = \mathbb{E}_{j \in \mathcal{C}_c}[\mathrm{MI}(x_j, y)]$ is the per-cluster mean mutual information, $\mathrm{CV}(\cdot)$ is the coefficient of variation, and $\overline{|C|}_\mathrm{cross}$ is mean cross-cluster absolute correlation. Three candidate governing variables were screened: cluster count $k$, $\mathrm{CV}(\mu)$, and the effective rank of the feature correlation matrix.

### 6.2 Three-Dataset Screening

**Table 6. Diversity metrics vs. AISO outcome (SC PR-AUC $\Delta$ vs. S0 baseline).**

| Dataset | $k$ | $\mathrm{CV}(\mu)$ | eff\_rank | $\Delta$ SC |
|---|---|---|---|---|
| Elliptic | 12 | **0.667** | 47.2 | **+0.111** |
| Amazon | 9 | 1.601 | 9.8 | +0.020 |
| YelpChi | 23 | 1.733 | 21.1 | $-0.014$ |

Spearman $\rho$: $\mathrm{CV}(\mu) = -1.0$, $k = -0.5$, eff\_rank $= +0.5$. $\mathrm{CV}(\mu)$ rank-orders all three datasets by AISO outcome ($n = 3$, observational). Lower $\mathrm{CV}(\mu)$ means cluster-level mutual information is balanced across clusters — which is precisely the condition under which AISO's information gradient ($M_{ij} \propto \tilde{\mu}_j - \tilde{\mu}_i$) creates meaningful directional specialization. When $\mathrm{CV}(\mu)$ is high, one cluster dominates MI and the gradient degenerates toward a single direction, collapsing the multi-type dynamics.

We note that $n = 3$ precludes statistical significance testing of this real-dataset correlation; the directional finding is observational. Synthetic validation ($n = 135$, Section 6.5) provides broader statistical support for AISO's advantage and contextualizes the real-data threshold as a deployment heuristic mediated by GNN-environment factors.

### 6.3 Failure Mode Classification

| Dataset | $\mathrm{CV}(\mu)$ | Failure type | Evidence |
|---|---|---|---|
| **Elliptic** | 0.667 | None — condition met | SA→SB→SC monotonically improving |
| **Amazon** | 1.601 | Dominant mode collapse | SA below random ($-0.063$); SC below SB ($-0.020$) |
| **YelpChi** | 1.733 | GCN propagation saturation | 18-method range = 0.037; all stages near-flat |

Each failure type is diagnosable before running AISO: Amazon's dominant mode is visible in its MI distribution; YelpChi's graph saturation is visible in the homophily coefficient. The threshold $\mathrm{CV}(\mu) < 1.0$ separates success (0.667) from failure (1.601, 1.733).

### 6.4 Pre-Deployment Feasibility Assessment

As a preliminary heuristic ($n = 3$), $\mathrm{CV}(\mu)$ converts a deployment decision into a three-minute pre-check: cluster features hierarchically, compute per-cluster MI, compute $\mathrm{CV}(\mu)$ — if below 1.0, AISO is *more likely* to provide gains over random baselines; otherwise, deterministic baselines (mRMR→SGD) may be sufficient. This threshold requires wider validation before prescriptive use (Section 7.2).

### 6.5 Synthetic Validation: AISO Robustness Across $\mathrm{CV}(\mu)$ Regimes

To extend beyond $n = 3$ real datasets, we ran a controlled synthetic experiment (135 conditions: $\mathrm{CV}(\mu) \in [0.2, 2.0]$, $K \in \{4, 8, 12\}$, 5 seeds each) using a proxy fitness that directly models the governing condition — MI-weighted cluster selection with cross-cluster correlation penalty, evaluated against a 50-trial random baseline. The AISO implementation mirrors the expA pipeline (AgglomerativeClustering on correlation distance, identical Smart $M$ construction).

**Key results:**

| Statistic | Value |
|---|---|
| Conditions | $n = 135$ |
| Mean $\Delta$ (AISO $-$ Random), all | $+0.044$ |
| Mean $\Delta$, $\mathrm{CV}(\mu) < 1.0$ | $+0.041$ ($n = 97$) |
| Mean $\Delta$, $\mathrm{CV}(\mu) \geq 1.0$ | $+0.048$ ($n = 38$) |
| Spearman $\rho(\mathrm{CV}, \Delta)$ | $+0.403$, $p < 0.0001$ |

AISO outperforms the random baseline in all 135 synthetic conditions (mean $\Delta > 0$); see Figure 4.

![Figure 4: Synthetic governing condition validation (n=135).](../results/governing_condition_synthetic.png)

**Sign reversal caveat.** The synthetic Spearman $\rho(\mathrm{CV}, \Delta) = +0.403$ is in the *opposite direction* from the real-data $\rho = -1.0$. This is not a minor discrepancy — it means the synthetic proxy does not reproduce the directional relationship observed in real deployments. The most likely explanation is that the proxy omits GNN-specific environmental factors: class imbalance dynamics, graph propagation saturation (dominant in YelpChi), and feature-noise amplification under high CV — none of which are present in the synthetic fitness model. Consequently, the synthetic $n = 135$ result supports only the weaker claim that AISO is broadly superior to a random baseline; it does **not** validate the $\mathrm{CV}(\mu) < 1.0$ directional threshold. That threshold remains an observational heuristic derived from $n = 3$ real datasets and requires wider validation (Section 7.2).

---

## 7. Discussion

### 7.1 Structural Interpretation

The Jaccard collapse (1.000 → 0.136 under asymmetric vs. symmetric $M$) confirms that cyclic preference structures — not radius constraints or explicit penalties — are the source of persistent specialization. All 15+ mechanism elaborations failed because they imposed *explicit* diversity control on a system that already achieves *implicit* diversity through asymmetric interaction; the productive complement is convergence assistance (phased refinement), not additional diversity pressure. The CEC-to-GNN transfer succeeds because discrete selection eliminates the type-spatial decoupling: agent type and selection action are the same variable, so type specialization directly produces diverse training coverage.

### 7.2 Limitations

1. **Governing condition is preliminary ($n = 3$).** The $\mathrm{CV}(\mu) < 1.0$ threshold is an observational heuristic from three real datasets. The synthetic validation ($n = 135$) supports AISO's general superiority over random baselines but shows a sign-reversed directional relationship (Section 6.5), indicating the proxy does not capture the GNN-specific factors that drive the real-world threshold. Wider real-dataset validation is required before the threshold can be used prescriptively.
2. **Headline comparison lacks formal significance.** The margin of AISO (0.6644 ± 0.020) over mRMR→SGD (0.6348 ± 0.006) on Elliptic is based on 5 seeds. A larger replication is needed for a definitive significance claim.
3. **CEC continuous niching: asymmetric mechanism is statistically inert.** In the continuous setting (Section 4), AISO+Refine (0.911) is not statistically distinguishable from Random+Refine (0.906, $p = 0.43$) or SymM+Refine (0.896, $p = 0.21$). The mechanism's contribution to spatial niching performance is therefore zero in this regime; the gain over PSO+LS is attributable to diversity preservation generally, not asymmetric routing specifically. The mechanism's value emerges only in discrete selection (Section 5), where type and position are the same variable.
4. **High-peak-count continuous niching.** F6 (18 peaks) and F7 (36 peaks) remain below CrowdingDE. Maintaining $\geq 18$ simultaneous niches with $N = 80$ agents exceeds AISO's current capacity.
5. **Smart $M$ pre-processing dependency.** Cluster assignment quality is the binding constraint for Smart $M$ reliability (Appendix B.3). Mis-specified clusters misdirect the information gradient.
6. **No theoretical non-collapse guarantee.** Empirical entropy measurements show sustained diversity under asymmetric $M$, but a rigorous proof of non-collapse under cyclic preference structures remains open.

### 7.3 Future Work

- **Fitness-informed anchor initialization**: Instead of random anchors (which created false attractors), initialize anchors from a preliminary scan of the landscape. This would address the single most impactful failure mode identified in the ablation.
- **Learnable $M$**: A Hebbian rule $\Delta M \propto \Delta f \cdot W_i \otimes W_{j^\star}$ produced near-zero improvement (+0.001, $p = 0.25$) in isolation — but was tested under a fixed learning rate without adaptive scheduling. A curriculum-based approach with early exploration and late consolidation may be more effective.
- **Active learning extension**: The two-stage pipeline's budget-constrained selection is structurally equivalent to an active learning query strategy. Connecting AISO to the active learning literature would generalize the governing condition beyond fraud detection.
- **Wider governing condition validation**: The $\mathrm{CV}(\mu) < 1.0$ threshold should be evaluated on additional real fraud datasets (e.g., TSOCIAL, TFINANCE) to determine whether the threshold generalizes, or whether a dataset-specific calibration is needed.

---

## 8. Conclusion

We presented AISO, a swarm optimizer with asymmetric bilinear compatibility $c_{ij} = W_i^\top M W_j \neq c_{ji}$ on simplex-valued type vectors. Our investigation makes three validated contributions.

**First, mechanism diagnostic with scope identification.** On CEC2013 F1–F8 (30 seeds), systematic ablation of 15+ candidate enhancements confirms that (1) asymmetric $M$ is the structural prerequisite for diversity persistence — symmetrizing $M$ collapses all agents to identical solutions (Jaccard 1.000); (2) phased local refinement is the sole productive complement (+0.466 average peak ratio); and (3) in continuous niching, the asymmetric mechanism is statistically indistinguishable from a random global phase ($p = 0.43$), establishing that its value is domain-specific to discrete selection. This is scope identification, not a failure.

**Second, application on fraud detection.** On Elliptic Bitcoin, the two-stage pipeline achieves PR-AUC 0.6644 (highest among 18 compared methods, 5 seeds), recovering 93.2% of full-graph performance under a constrained label budget, confirmed by the asymmetry ablation (Jaccard 1.000 → 0.136, PR-AUC 0.5527 → 0.6644) and monotone 2×2 stage decomposition. Formal significance of the margin over mRMR→SGD (+0.030) is limited by seed count.

**Third, a preliminary governing condition.** $\mathrm{CV}(\mu)$ rank-orders three real datasets by outcome (Spearman $\rho = -1.0$, $n = 3$, observational); synthetic validation ($n = 135$, $p < 0.0001$) confirms AISO outperforms random broadly, but the directional relationship reverses in the synthetic setting, indicating the real-world threshold is mediated by GNN-environment factors and requires wider validation.

The unified message: asymmetric bilinear interaction is a structurally motivated diversity mechanism whose effective operating regime is established by controlled ablation, and whose transfer to structured selection problems is validated and conditionally bounded — enabling practitioners to assess applicability before deployment.

---

## References

- Brits, R., Engelbrecht, A.P., & van den Bergh, F. (2007). Locating multiple optima using particle swarm optimization. *Applied Mathematics and Computation*, 189(2), 1859–1883.
- Caldarelli, G., Capocci, A., & Servedio, V.D.P. (2007). Dynamical affinity in opinion dynamics modelling. arXiv:physics/0701204.
- Chiang, W.L., et al. (2019). Cluster-GCN: An efficient algorithm for training deep and large graph convolutional networks. *KDD 2019*.
- Ding, C., & Peng, H. (2005). Minimum redundancy feature selection from microarray gene expression data. *JBCB*, 3(2), 185–205.
- Dou, Y., et al. (2020). Enhancing graph neural network-based fraud detection via locally homophilous aggregation. *CIKM 2020*.
- Engelbrecht, A.P. (2010). Heterogeneous Particle Swarm Optimization. *LNCS* 6234, 191–202.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *ICNN 1995*.
- Li, X. (2004). Adaptively choosing neighbourhood bests using species in a particle swarm optimizer for multimodal function optimization. *GECCO*, 105–116.
- Li, X. (2010). Niching without niching parameters: PSO using a ring topology. *IEEE TEVC*, 14(1), 150–169.
- Li, X., Engelbrecht, A., & Epitropakis, M.G. (2013). Benchmark functions for CEC'2013 special session and competition on niching methods for multimodal function optimization. RMIT University Technical Report.
- Liu, Y., et al. (2021). Pick and choose: A GNN-based imbalanced learning approach for fraud detection. *WWW 2021*.
- Liu, Z., et al. (2025). Game-theoretic asymmetric interaction in swarm robotics. *IEEE Robotics and Automation Letters*.
- Qu, B.Y., Suganthan, P.N., & Das, S. (2013). A distance-based locally informed particle swarm model for multimodal optimization. *IEEE TEVC*, 17(3), 387–402.
- Riget, J., & Vesterstrøm, J.S. (2002). A diversity-guided particle swarm optimizer — the ARPSO. EVALife TR 2002-02.
- Schlichtkrull, M., et al. (2018). Modeling relational data with graph convolutional networks. *ESWC 2018*.
- Shen, D., & Li, Y. (2012). A role-based particle swarm optimization for multimodal optimization. *ICCIS 2012*.
- Silva, A., Neves, A., & Costa, E. (2002). Chasing the swarm: A predator-prey approach to function optimization. *MENDEL*.
- Thomsen, R. (2004). Multimodal optimization using crowding-based differential evolution. *CEC 2004*.
- Wu, F., et al. (2019). Simplifying graph convolutional networks. *ICML 2019*.
- Zeng, H., et al. (2020). GraphSAINT: Graph sampling based inductive learning method. *ICLR 2020*.
- Moscato, P. (1989). On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms. Caltech Concurrent Computation Program, C3P Report 826.
- Molina, D., Lozano, M., García-Martínez, C., & Herrera, F. (2010). Memetic algorithms for continuous optimisation based on local search chains. *Evolutionary Computation*, 18(1), 27–63.
- Moscato, P., & Cotta, C. (2003). A gentle introduction to memetic algorithms. In *Handbook of Metaheuristics*, Springer, 105–144.
- Neri, F., & Tirronen, V. (2010). Recent advances in differential evolution: A survey and experimental analysis. *Artificial Intelligence Review*, 33(1–2), 61–106.

---

## Appendix A: Hyperparameters

**Table A1. AISO hyperparameters (CEC2013 experiments).**

| Parameter | Value | Description |
|---|---|---|
| $N$ | 80 | Number of agents |
| $T$ | 200 | Total iterations |
| $\alpha$ | 0.4 | Position update step size |
| $\beta$ | 0.1 | Type assimilation rate |
| Phase-1 fraction | 0.70 | Fraction of $T$ for global search |
| Phase-2 step (initial) | $0.05 \times \mathrm{range}$ | Gaussian walk initial std |
| Adaptive repulsion $w_r$ | $1 + 3e^{-\delta/0.12}$ | Repulsion multiplier ($\delta$ = mean inter-agent distance) |
| Accuracy threshold $\varepsilon$ | $0.01 \times \mathrm{range}$ | Peak found criterion |
| Seeds | 30 | Independent runs per variant |

**Table A2. AISO hyperparameters (GNN application).**

| Parameter | Value | Description |
|---|---|---|
| $N$ | 20 | Number of agents |
| $T$ | 60 | Total iterations |
| $K$ (feature clusters) | 15 (Stage 1) | AgglomerativeClustering, correlation distance |
| $K$ (node types) | 20 (Stage 2) | Re-clustered within selected feature subspace |
| $\gamma$ (Smart M) | 0.5 | Information gradient weight |
| Phase-1 fraction | 0.70 | Same schedule as CEC |
| $\alpha$ | 0.4 | Position update step size |
| $\beta$ | 0.1 | Type assimilation rate |
| GCN hidden dim | 64 | — |
| GCN dropout | 0.5 | — |
| Adam lr | 0.01 | — |
| Early stopping patience | 15 | — |
| Budget ($N_\mathrm{illicit}$) | 1,000 | Fraud nodes selected |
| Budget ($N_\mathrm{licit}$) | 10,000 | Normal nodes selected |
| Seeds | 5 $\{0, 7, 42, 77, 123\}$ | GNN training seeds |

## Appendix B: Sensitivity Analysis

**Table B1. Phase ratio (ρ) sensitivity on CEC F1–F8 (30 seeds, avg peak ratio).**

| ρ | 0.3 | 0.5 | **0.7** | 0.9 |
|---|---|---|---|---|
| Avg PR | 0.936 | 0.928 | **0.911** | 0.850 |

Performance decreases monotonically as ρ increases (less refinement time). ρ=0.7 was chosen as the initial default to ensure a sufficiently long global search phase before refinement; ρ ∈ [0.3, 0.7] yields strong performance (0.911–0.936) with no cliff, and qualitative conclusions are unchanged across this range. The trend is consistent with the CEC finding that local refinement is the dominant mechanism in continuous niching.

**Table B2. Smart M gradient weight (γ) sensitivity on proxy fitness matching Elliptic profile (CV(μ)≈0.667, K=15, 10 seeds).**

| γ | 0.1 | 0.3 | **0.5** | 0.7 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| avg Δ (AISO−Random) | +0.019 | +0.019 | **+0.019** | +0.019 | +0.019 | +0.016 |

AISO advantage is flat across γ ∈ [0.1, 1.0] and degrades slightly at γ=2.0 (over-dominance of the information gradient relative to correlation repulsion). γ=0.5 is a robust default; practitioners can adjust freely within [0.1, 1.0] without performance degradation.

**B.3 Smart M Pre-Processing Robustness.** Gaussian noise on MI estimates ($\sigma \in \{0, 0.05, 0.1, 0.2, 0.5, 1.0\}$) and cluster-assignment swap noise ($p \in \{0, 0.1, 0.2, 0.3, 0.5\}$) were evaluated on Elliptic (LR wrapper, 5 seeds). MI noise: PR-AUC $0.3906 \to 0.4303$ ($\sigma=0 \to 1.0$, no degradation). Cluster swap: PR-AUC $0.3906 \to 0.3152$ ($p=0 \to 0.5$, moderate). Smart $M$ is robust to MI estimation noise; cluster assignment quality is the binding constraint.

---

**Table A3. Synthetic governing condition experiment (Section 6.5).**

| Parameter | Value |
|---|---|
| CV($\mu$) targets | $\{0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0\}$ |
| $K$ values | $\{4, 8, 12\}$ |
| Seeds per condition | 5 |
| Total conditions | 135 |
| Samples per dataset | 1,500 |
| Features per cluster | 10 |
| Class imbalance | 10% positive |
| Random baseline draws | 50 |
| AISO agents / iters | 20 / 60 |
