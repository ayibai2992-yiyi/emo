# Algorithm Innovation Method Draft

This document turns the current CPEB + MEA + THEGN implementation into a
paper-ready algorithmic contribution. It is written as method material rather
than as final experimental claims; all numeric results must be filled only after
real experiments.

## Proposed Method Name

**RACF-ERC: Reliability-Aware Causal and Meta-Graph Fusion for Personalized
Emotion Recognition in Chinese Multi-turn Dialogues**

Short names that can be used in tables:

- **CPEB**: Causal Personalized Emotion Baseline
- **MEA**: Meta-Emotion Adapter
- **THEGN**: Temporal Heterogeneous Emotion Graph Network
- **RACF**: Reliability-Aware Contextual Fusion
- **LGF**: Learnable Graph Fallback

## Core Contributions

1. **Causal personalized emotion debiasing.** We explicitly separate stable
   user-specific expression habits from transient emotional states, reducing
   user-level confounding in personalized ERC.
2. **Few-shot user adaptation.** We use a meta-emotion adapter to support fast
   adaptation from a small user-specific support set.
3. **Temporal heterogeneous emotion graph modeling.** Dialogue history is
   represented as a temporal graph so the model can capture long-range emotional
   evolution and history-dependent crisis signals.
4. **Reliability-aware contextual fusion.** Instead of concatenating branch
   probabilities with fixed weights, we learn sample-level fusion weights from
   branch uncertainty, branch availability, and current text context.
5. **Learnable cold-start graph fallback.** When dialogue history is missing,
   the graph branch produces trainable text-conditioned evidence instead of a
   zero vector, enabling cold-start inference without contaminating fusion.

## Task Formulation

Given a user `u`, a current utterance `x_t`, and optional dialogue history
`H_t = {x_1, ..., x_{t-1}}`, the model predicts:

- an emotion label `y_t in {1, ..., K}`, where `K = 8`;
- optionally a crisis label `c_t in {0, 1}`, where crisis can be mapped from
  `label_id = 7` or an annotated `is_crisis` field.

The model estimates:

```text
p(y_t | x_t, u, H_t) = RACF(CPEB(x_t, u), MEA(x_t, S_u), THEGN(H_t, x_t))
```

where `S_u` is the user-specific support set for few-shot adaptation.

## Module 1: CPEB

CPEB models a user-specific emotional baseline and removes expression-habit
confounding from the raw emotion distribution.

Let `h_t` be the BERT representation of the current utterance:

```text
h_t = BERT([CLS], x_t)
```

The user baseline estimator predicts a variational baseline:

```text
mu_u, log_sigma_u^2 = f_base([e_u ; h_t])
b_u = sigmoid(mu_u + epsilon * exp(0.5 * log_sigma_u^2))
```

The raw emotion distribution is:

```text
p_raw = softmax(f_cls(h_t))
```

The causal intervention module estimates a debiased distribution:

```text
z_u = f_habit(b_u)
p_cpeb = f_debias([p_raw ; z_u])
```

The causal regularization term can be written as:

```text
L_causal = KL(p_raw || p_cpeb) +
           beta * KL(q(b_u | x_t, u) || N(0, I))
```

## Module 2: MEA

MEA provides user-level few-shot adaptation. Given support examples
`S_u = {(x_i, y_i)}`, support features are encoded as:

```text
h_i = BERT([CLS], x_i)
```

For the MAML variant, adapter parameters are updated by inner-loop gradient
steps:

```text
theta'_u = theta - alpha * grad_theta L_Su(f_theta(h_i), y_i)
p_mea = softmax(f_theta'_u(h_t))
```

For the prototypical variant, class prototypes are:

```text
proto_k = mean({g(h_i) | y_i = k})
p_mea(k | x_t) = softmax(-d(g(h_t), proto_k))
```

Recommended paper design: report both variants, then use validation Macro-F1 and
crisis PR-AUC to select the primary MEA implementation.

## Module 3: THEGN

THEGN represents dialogue history as a temporal heterogeneous graph:

```text
G_t = (V_t, E_t)
```

Possible node types:

- utterance node;
- speaker/user node;
- emotion-state node.

Possible edge types:

- temporal edge;
- reply edge;
- influence edge;
- optional same-speaker edge.

For edge `(i, j)` with edge type `r`, heterogeneous attention is:

```text
alpha_ij^r = softmax_j(LeakyReLU(a_r^T [W_r h_i ; W_r h_j]) * w_ij)
```

where temporal decay can be:

```text
w_ij = exp(-lambda * |time_i - time_j|)
```

Node update:

```text
h_j' = sigma(sum_r sum_{i in N_r(j)} alpha_ij^r W_r h_i)
```

The graph branch prediction for the current turn is:

```text
p_graph = softmax(f_graph(h_t^G))
```

## Learnable Graph Fallback

If dialogue history or graph edges are unavailable, the original zero-vector
path makes the graph branch untrainable and may mislead fusion. We replace it
with a trainable fallback:

```text
p_lgf = softmax(f_lgf(h_t))
```

The graph evidence used by fusion is:

```text
p_thegn = I_graph * p_graph + (1 - I_graph) * p_lgf
```

where `I_graph = 1` if a valid graph exists and `0` otherwise.

This lets the same model handle both cold-start users and users with rich
dialogue history.

## RACF: Reliability-Aware Contextual Fusion

The three branch outputs are first normalized:

```text
q_m = normalize(p_m), m in {cpeb, mea, thegn}
```

For each branch, reliability features include:

```text
r_m = [q_m ; max(q_m) ; 1 - H(q_m) / log(K) ; availability_m]
```

where:

- `max(q_m)` is branch confidence;
- `H(q_m)` is entropy;
- `availability_m` is `1` for CPEB/MEA and graph availability for THEGN.

Text context is projected as:

```text
c_t = f_ctx(h_t)
```

Fusion weights are learned sample-wise:

```text
gamma = softmax(f_gate([c_t ; r_cpeb ; r_mea ; r_thegn]))
```

Branch evidence is projected:

```text
v_m = f_m(q_m)
```

Final fused representation:

```text
v = sum_m gamma_m * v_m
```

Final emotion prediction:

```text
p_final = softmax(f_out(v + c_t))
```

The fusion weights `gamma` can be logged for interpretability, allowing analysis
of whether the model trusts causal debiasing, user adaptation, or graph history
for each sample.

## Optional Crisis Auxiliary Head

To align the model with psychological crisis warning, add an auxiliary crisis
head:

```text
p_crisis = sigmoid(f_crisis(v + c_t))
```

Recommended multi-task objective:

```text
L = L_emotion +
    lambda_1 * L_crisis +
    lambda_2 * L_causal +
    lambda_3 * L_meta +
    lambda_4 * L_graph
```

Suggested defaults for validation search, not final claims:

```text
lambda_1 in {0.3, 0.5, 1.0}
lambda_2 in {0.05, 0.1, 0.2}
lambda_3 in {0.05, 0.1}
lambda_4 in {0.01, 0.05}
```

## Algorithm Pseudocode

```text
Algorithm 1: RACF-ERC Inference
Input: current utterance x_t, user id u, optional history H_t, optional support set S_u
Output: emotion distribution p_final, optional crisis score p_crisis

1: h_t <- BERT(x_t)
2: p_cpeb, baseline_stats <- CPEB(h_t, u)
3: if S_u is available then
4:     p_mea <- MEA_Adapt(h_t, S_u)
5: else
6:     p_mea <- MEA_Default(h_t)
7: end if
8: if H_t can form a valid graph then
9:     G_t <- BuildDialogueGraph(H_t, x_t)
10:    p_thegn <- THEGN(G_t)
11:    I_graph <- 1
12: else
13:    p_thegn <- LGF(h_t)
14:    I_graph <- 0
15: end if
16: r <- ReliabilityFeatures(p_cpeb, p_mea, p_thegn, I_graph)
17: gamma <- softmax(Gate([Project(h_t); r]))
18: p_final <- Output(sum_m gamma_m * Project_m(p_m) + Project(h_t))
19: return p_final
```

## Training Protocol

1. Use user-level split so the same user does not appear across train,
   validation, and test sets.
2. Select crisis threshold only on validation set.
3. Report at least three random seeds where compute allows.
4. Save each run's config, seed, split, metrics JSON, and summary CSV.
5. Use class-weighted cross-entropy or focal loss for emotion imbalance.
6. For crisis warning, report PR-AUC, recall, FPR, and threshold policy.

## Ablation Design

### Module Ablation

| Setting | CPEB | MEA | THEGN | LGF | RACF |
| --- | --- | --- | --- | --- | --- |
| Single BERT | No | No | No | No | No |
| CPEB only | Yes | No | No | No | No |
| MEA only | No | Yes | No | No | No |
| THEGN only | No | No | Yes | Yes | No |
| CPEB + MEA | Yes | Yes | No | No | RACF optional |
| CPEB + THEGN | Yes | No | Yes | Yes | RACF optional |
| MEA + THEGN | No | Yes | Yes | Yes | RACF optional |
| Full model | Yes | Yes | Yes | Yes | Yes |

### Fusion Ablation

| Fusion Method | Description |
| --- | --- |
| Concatenation MLP | Original fixed concatenation baseline |
| Mean ensemble | Uniform average of three branches |
| Confidence weighted | Hand-crafted confidence weighting |
| RACF | Learned reliability-aware contextual fusion |

### Cold-start Ablation

| User Group | Definition | Expected Insight |
| --- | --- | --- |
| Cold-start | no history or no support set | Validates LGF and MEA fallback |
| Few-shot | 1/5/10 support examples | Validates MEA adaptation |
| History-rich | enough dialogue history | Validates THEGN contribution |

## Main Metrics

Emotion classification:

- Macro-F1;
- Micro-F1;
- Weighted-F1;
- per-class F1, especially despair.

Crisis warning:

- PR-AUC;
- ROC-AUC, if class distribution allows;
- recall;
- FPR;
- F1 at validation-selected threshold.

Efficiency:

- trainable parameters;
- inference latency P50/P95/P99;
- throughput;
- deep-analysis trigger ratio if used in the full monitoring pipeline.

## Suggested Paper Claims After Validation

Only make these claims if supported by real experiments:

1. RACF improves over fixed concatenation by down-weighting unavailable or
   uncertain branches.
2. LGF improves cold-start robustness compared with zero-vector graph evidence.
3. MEA improves few-shot personalized emotion recognition under user-level
   splits.
4. THEGN improves history-rich dialogue samples by modeling temporal emotion
   evolution.
5. The full model improves both Macro-F1 and crisis PR-AUC compared with a
   single BERT baseline under the same split and preprocessing.

## Implementation Mapping

Current code alignment:

- `src/models/unified_model.py`
  - `EvidenceAwareFusion`: implements RACF.
  - `graph_fallback`: implements LGF.
  - `fusion_weights`: exposes sample-level branch contribution.
  - `graph_availability`: marks whether THEGN used a real graph.
- `src/models/cpeb.py`
  - implements CPEB baseline estimation and causal intervention.
- `src/models/meta_learner.py`
  - implements MAML, Prototypical Network, and LoRA-style adapters.
- `src/models/thegn.py`
  - implements temporal heterogeneous graph attention.

## Next Implementation Steps

1. Add an optional crisis auxiliary head to `UnifiedEmotionModel`.
2. Extend training code to optimize emotion loss, crisis loss, causal loss, and
   graph regularization jointly.
3. Add validation-set threshold calibration for crisis warning.
4. Add experiment scripts for module ablation, fusion ablation, and cold-start
   grouped evaluation.
5. Install or declare PyTorch/Transformers dependencies so tests can run in a
   clean environment.
