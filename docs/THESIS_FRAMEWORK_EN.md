# Master's Thesis Framework - ALIGN Research

## 📋 Complete Thesis Structure Planning

---

## Chapter 1: Introduction
**Target Word Count: 3,000-4,000 words**

### 1.1 Research Background and Motivation
- Importance and real-world applications of tabular data
- Comparative strengths and weaknesses of existing deep learning models
  - Neural network models: Excellent at capturing nonlinearities, but require large labeled datasets
  - Tree-based models (XGBoost, CatBoost, LightGBM): Superior performance on small to medium-sized datasets
  - Pre-trained models (TabPFN): Leverage large-scale pre-training for rapid few-shot adaptation
- Potential and underexplored applications of Graph Neural Networks (GNNs) in tabular learning

### 1.2 Problem Formulation
- **Core Research Question 1**: How can we systematically evaluate the integration benefits of GNNs with different tabular model architectures?
- **Core Research Question 2**: How do GNN gains vary across different model architectures, data types, and sample scales?
- **Core Research Question 3**: At which processing stage is it most effective to inject GNN? Is there a universally optimal injection point?

### 1.3 Research Objectives and Main Contributions
- Definition of the ALIGN (Analyzing the Integration of Graph Networks in Tabular Learning) framework
- Expected research outputs
  - Phase 1: Comprehensive comparative analysis (13,920 experiments)
  - Phase 2: Three systematic ablation studies
- Academic innovations
  - First unified five-stage pipeline for fair cross-model GNN injection comparison
  - Large-scale systematic evaluation (116 datasets × 10 decomposable models × 6 stages)
  - Identification of key determinants and boundary conditions for GNN gains
- Practical application value
  - Decision guidelines for practitioners: "When should I use GNN?"
  - Open-source framework with fully reproducible experimental settings

### 1.4 Thesis Organization
- Content preview and logical relationship map across chapters
- Reading guide: Recommended chapter combinations for different audiences

---

## Chapter 2: Related Work
**Target Word Count: 4,000-5,000 words**

### 2.1 Progress in Tabular Deep Learning
- Evolution and development timeline of tabular data learning
- Comprehensive review of major SOTA models
  - Transformer family: FT-Transformer, TabTransformer, ExcelFormer
  - Attention mechanism models: TabNet
  - Self-supervised models: SCARF, VIME, SubTab
  - Other innovations: Trompt, TabM
- Design principles, applicable scenarios, and comparative strengths/weaknesses

### 2.2 GNN Theory and Applications
- GNN theoretical foundations
  - Message passing mechanism in Graph Convolutional Networks (GCNs)
  - Various GNN variants (GAT, GIN, GraphSAGE, etc.)
  - Graph construction strategies and similarity metrics
- GNN applications on structured/unstructured data
- Recent advances in dynamic graph construction (DGM) and differentiable neighborhood learning

### 2.3 Existing Integration Attempts of GNNs and Tabular Models
- Design philosophies of self-contained GNN tabular models
  - **TabGNN**: Graph-based feature interactions
  - **T2G-Former**: Table-to-Graph Transformer mapping
  - **DGM (Differentiable Graph Module)**: Dynamic graph structure learning
  - **LAN-GNN**: Adaptive neighborhood learning
- Strengths, limitations, and unresolved challenges
- Why systematic integration analysis is needed rather than isolated improvements

### 2.4 Innovation Position of This Research Relative to Existing Work
- **Systematicity**: Upgrade from single-model to unified framework for multi-model comparison
- **Scale**: Comprehensive evaluation across 116 datasets, multiple stages, and multiple models
- **Methodology**: Scientific comparison advantages from unified five-stage pipeline
- **Empirical basis**: Statistical significance and reproducibility of large-scale experimental results
- **Guidance**: Shift from "Does it work?" to "When does it work?" with mechanistic understanding

---

## Chapter 3: Methodology
**Target Word Count: 6,000-8,000 words**

### 3.1 TaBLEau Framework Design

#### 3.1.1 Framework Architecture and Core Features
- Framework positioning: Unified benchmark framework for tabular data deep learning
- Core features
  - Unified interface: All models follow identical input-output format
  - Diverse data: 116 datasets across multiple classification dimensions
  - Consistent environment: Standardized data splits and evaluation metrics
  - Extensibility: Support for dynamic model and dataset addition

#### 3.1.2 Dataset Organization Structure
- Quantity and classification dimensions
  - Dataset scale: Small (<5,000 rows) vs Large (>5,000 rows)
  - Task type: Binary classification (binclass), multi-class (multiclass), regression
  - Feature type proportion: Numerical-dominant, categorical-dominant, balanced
- Typical paths and storage locations
- Standard data split strategies
  - Few-shot setting (0.05:0.15:0.80): Simulates scarce annotation scenarios
  - Fully-supervised setting (0.80:0.15:0.05): Control group with sufficient training data
- Data characteristics and diversity assurance

#### 3.1.3 Model Family Classification
- Decomposable models (10 total)
  - PyTorch Frame family (6): ExcelFormer, FT-Transformer, ResNet, TabNet, TabTransformer, Trompt
  - Custom family (4): SCARF, SubTab, VIME, TabM
- Reference baseline models (8 total)
  - Tree models (3): XGBoost, CatBoost, LightGBM
  - Self-contained GNN models (4): TabGNN, T2G-Former, DGM, LAN-GNN
  - Pre-trained model (1): TabPFN
- Model overview table and expansion plans

### 3.2 Unified Five-Stage Pipeline Design

#### 3.2.1 Design Philosophy
- Design inspiration from PyTorch Frame paper
- Necessity for fair cross-model comparison
- Stage granularity selection (detailed yet universally applicable)

#### 3.2.2 Detailed Definition of Five Processing Stages

**Stage 0: start (Entry Point, Dummy Stage)**
- Function: No data processing; marks the position for "GNN injection at the very beginning"
- Implementation significance: Allows graph structure enhancement before main model pipeline

**Stage 1: materialize (Materialization Stage)**
- Functions:
  - Convert DataFrame to TensorFrame or DataLoader
  - Categorical encoding, missing value handling, standardization and normalization
  - Mutual information ranking and other preprocessing
- Output: Structured tensor representation

**Stage 2: encoding (Encoding Stage)**
- Functions:
  - Encode each field as token vectors
  - Add positional encodings
  - Generate initial representations
- Output: tokens `[Batch, Features, Channels]`

**Stage 3: columnwise (Column-wise Interaction Stage)**
- Functions:
  - Learning interactions across feature dimensions
  - Multi-head attention mechanisms and other interaction methods
  - Capture relationships between different fields
- Output: Tokens after interaction

**Stage 4: decoding (Decoding Stage)**
- Functions:
  - Decode feature representations to final predictions
  - Pooling, fully connected layers, etc.
  - Output logits or predicted values

#### 3.2.3 Pipeline Flow and Mapping Verification
- Stage flow diagram
- Model-specific stage mappings
  - Mapping documents for 10 models
  - Verification methods for mapping correctness

### 3.3 GNN Injection Strategy

#### 3.3.1 Core GNN Component Design

**DGM_d (Dynamic Graph Generation Module)**
- Function: Dynamically construct graph structure from input features
- Graph construction strategies: k-nearest neighbors (k-NN), fully connected, or hybrid
- Differentiable design: Use Gumbel softmax for differentiable neighbor selection

**SimpleGCN (Graph Convolutional Network)**
- Stack of multiple GCNConv layers
- Configurable hyperparameters: layer count, hidden dimensions, dropout, etc.
- Message passing and feature aggregation mechanism

**Residual Fusion Gate**
- Merge GNN output with original features
- Mechanism: `output = original + sigmoid(alpha) * gnn_output`
- Learnable parameters and training stability

**Auxiliary Modules**
- Field positional encodings, pooling operations, projection layers, etc.

#### 3.3.2 Six GNN Injection Strategies in Detail

**Strategy 0: none (No Injection, Baseline)**
- No GNN integration
- Performance baseline for comparison

**Strategy 1: start (Offline Feature Pre-injection)**
- Execution timing: Before data enters main pipeline
- Processing pipeline: Dimension expansion → self-attention → pooling → dynamic graph construction → GCN → write-back
- Characteristics: Offline processing, no model backbone changes, requires supervision signal

**Strategy 2: materialize (Post-Materialization Injection)**
- Execution timing: After materialization, before encoding
- Graph enhancement on more structured data
- Offline processing

**Strategy 3: encoding (Joint Training: Post-Encoding Injection)**
- Execution timing: After encoding, before convolution
- Token-level graph structure learning
- End-to-end joint training

**Strategy 4: columnwise (Joint Training: Post-Convolution Injection)**
- Execution timing: After column-wise interaction, before decoding
- Graph interaction on high-level features
- **Most frequently brings gains in experiments**

**Strategy 5: decoding (Joint Training: GNN Replaces Decoder)**
- Execution timing: Completely replace decoding stage
- End-to-end graph prediction
- Largest architectural modification

#### 3.3.3 Unification of Injection Styles and Key Observations
- Advantages of Dynamic-Graph + Attention (DGM + Self-Attn) design
- Comparison with Static kNN Graph design
- Potential conflicts between contrastive self-supervised objectives and GNN
- Model learning objective and GNN inductive bias compatibility analysis

#### 3.3.4 GNN Configuration Parameters
- gnn_hidden_dim, gnn_layers, gnn_dropout parameter settings
- Rationale for hyperparameter choices
- Stage selection guidance

### 3.4 Phase 1 Experimental Design (Comprehensive Comparative Analysis)

#### 3.4.1 Experimental Scale and Parameters
- Model quantity and coverage
- Dataset quantity: 116
- Split strategies: 2 types (few-shot vs fully-supervised)
- GNN stages: 6
- Total experiment count calculation: 10 models × 116 datasets × 2 splits × 6 stages = 13,920

#### 3.4.2 Baseline Configuration

**Baseline Category 1: Self-Baselines**
- few-shot non-GNN: Same model, same split, no GNN
- full-sample non-GNN: Control group

**Baseline Category 2: Tree Model Baselines**
- XGBoost, CatBoost, LightGBM
- Performance under two split settings

**Baseline Category 3: Self-contained GNN Baselines**
- TabGNN, T2G-Former, DGM, LAN-GNN
- Comparison between native GNN designs and GNN injection

**Baseline Category 4: Pre-trained Baseline**
- TabPFN: Strong baseline from large-scale pre-training

#### 3.4.3 Evaluation Metrics and Comparison Rules
- Primary metrics: AUC (classification), MAE (regression), average ranking
- Tolerance definition: 1e-3
- Application scenarios for strict vs tolerance-based comparison
- Per-category grouping analysis (18 combinations)

#### 3.4.4 Output Structure and Analysis Dimensions
- Per-model detailed report structure
- CSV tables, Markdown analysis, visualization plots

### 3.5 Phase 2 Experimental Design (Ablation Studies)

#### 3.5.1 Ablation Study 1: Impact of Training Sample Quantity on GNN Gains

**Research Hypothesis**
> GNN injection provides significant advantages with fewer training samples, but gains gradually diminish as sample quantity increases.

**Experimental Design**
- Dataset selection: small_datasets + binclass (20 datasets)
- Training ratio scanning: 16 points (0.05 to 0.80)
- Random seeds: 20 (complete) or 5 (quick verification)
- Fixed validation set: 15%

**Expected Result Characteristics**
- None baseline: Monotonic increase
- GNN variants: Outperform none at low ratios, decrease at high ratios
- Gain curve: columnwise reaches maximum gain, decreases with training ratio

**Statistical Aggregation**
- Aggregation logic: 20 datasets × N seeds × 6 stages
- Compute: mean, variance, std

#### 3.5.2 Ablation Study 2: Impact of Numerical Feature Proportion on GNN Gains

**Research Hypothesis**
> GNN injection works best when numerical feature proportion is high; gains decrease as categorical features increase.

**Experimental Design**
- Dataset selection: small_datasets + binclass (20 datasets with mixed features)
- Numerical proportion scanning: 6-11 points (100%, 80%, 60%, ..., 0%)
- Feature adjustment strategy: Dynamic feature selection or diverse dataset selection
- Fixed split: train=0.05, val=0.15, test=0.80 (few-shot)

**Expected Result Characteristics**
- Performance and gain decrease with numerical proportion
- Causal analysis: Relationship between Euclidean distance and feature similarity, categorical feature sparsity

#### 3.5.3 Ablation Study 3: Impact of Dataset Size on GNN Gains

**Research Hypothesis**
> GNN injection is most effective with smaller datasets; gains decrease as dataset size increases.

**Experimental Design**
- Dataset selection: large_datasets + binclass (20 datasets)
- Subsampling ratio scanning: 10 points (10%, 20%, ..., 100%)
- Apply few-shot split (0.05/0.15/0.80) to each subset
- Random seeds: 20 (complete) or 5

**Important Note**
- Regardless of sampling ratio, apply few-shot 5% training ratio to the subset
- This tests the relative impact of "total sample quantity" vs "training sample quantity"

**Expected Result Characteristics**
- Maximum GNN gains at 10% sampling
- Gains decrease with sampling ratio increase
- Gains approach zero or slightly negative at 100% sampling

#### 3.5.4 Statistical Methods and Error Handling
- Calculation and presentation of mean, variance, standard deviation
- t-test/ANOVA statistical testing
- Rendering of error bars or confidence intervals

---

## Chapter 4: Experimental Results and Analysis
**Target Word Count: 8,000-10,000 words**

### 4.1 Phase 1: Comprehensive Comparative Analysis

#### 4.1.1 Overall Findings and Rankings
- Average performance ranking of GNN stages
- Theoretical and empirical evidence for columnwise's consistent advantage
- Quantification of gains in low-sample scenarios

#### 4.1.2 Fine-grained Analysis by Dataset Categories
- Performance comparison across 18 category combinations
- Small vs large dataset comparison
- Performance variance across task types
  - Binary classification: Best GNN performance
  - Multi-class and regression: Diminishing trends
- Feature type impact analysis
  - Sharp contrast between numerical-dominant and categorical-dominant

#### 4.1.3 Cross-model Sensitivity Analysis
- Ranking of 10 models by GNN responsiveness
- Explanations for high sensitivity in FT-Transformer and ExcelFormer
- Medium sensitivity in TabNet and SCARF
- Low sensitivity and instability in ResNet and VIME
- Correlation between model architecture and GNN compatibility

#### 4.1.4 Key Statistical Indicators
- Average ranking tables for each model in each category
- Win/tie/loss statistics against various baseline categories

### 4.2 Comparative Analysis Against Multiple Baselines

#### 4.2.1 vs few-shot Non-GNN Baselines
- Definition, tolerance, comparison rules
- Data and conclusions

#### 4.2.2 vs Tree Models (XGBoost, CatBoost, LightGBM)
- Strong performance of tree models across different data types
- When GNN injection can overcome tree models

#### 4.2.3 vs Self-contained GNN Models (TabGNN, T2G-Former, DGM, LAN-GNN)
- Critical question: Native GNN design vs GNN injection strategy trade-offs
- Position alignment hypothesis: Map self-contained model GNNs to five-stage framework
- Experimental findings: When GNN injection matches or exceeds native design

#### 4.2.4 vs Pre-trained Baseline (TabPFN)
- TabPFN's few-shot advantages
- Can GNN injection supplement pre-training limitations?

### 4.3 Phase 2 Ablation Study Results

#### 4.3.1 Ablation Study 1: Impact of Training Sample Quantity
- Performance curve presentation (6 lines with error bars)
- Gain curve presentation and trend analysis
- Critical point identification (where GNN gains turn negative)
- Cross-model consistency (do all models follow same trend?)
- Statistical significance testing

#### 4.3.2 Ablation Study 2: Impact of Numerical Feature Proportion
- Relationship plots of performance vs feature type
- Quantitative analysis of gain decrease with numerical proportion
- Specific impact mechanisms of categorical features on graph construction
- Cross-model consistency verification

#### 4.3.3 Ablation Study 3: Impact of Dataset Size
- Plots of sampling ratio vs performance relationship
- Verification of small dataset inductive bias hypothesis
- Quantitative support for GNN redundancy in large datasets
- Interaction effects analysis between sample quantity and dataset size

#### 4.3.4 Comprehensive Comparison of Three Ablations
- Relative importance ranking of three independent variables
- Interaction effect analysis
- Implications for "when to use GNN" decision-making

### 4.4 Visualization and Intuitive Presentation
- Performance line plots (6 lines: all stages)
- Gain curve plots (5 lines: all non-none stages)
- Error bars/shaded confidence intervals
- Heatmaps: Model × dataset category × optimal stage
- Box plots: Performance distribution comparison
- Model sensitivity ranking plots

---

## Chapter 5: Discussion
**Target Word Count: 5,000-7,000 words**

### 5.1 Main Findings and Deep Mechanistic Analysis

#### 5.1.1 Why Is Columnwise the Optimal Stage?
- Timing appropriateness analysis
  - Sufficiently leverages high-level features from column-wise interaction
  - Preserves token dimensions for decoder layer use
- Feature quality analysis
  - Post-interaction features more conducive to meaningful graph construction
- Comparison with other stages
  - Why encoding is second-best, decoding worse
  - Why start/materialize perform poorly

#### 5.1.2 Why Is GNN More Effective with Small Data and Few Samples?
- Inductive bias (inductive bias) mechanism
  - Assumptions about graph structure provide regularization with insufficient samples
  - Intuitive understanding of overfitting prevention
- Information-theoretic analysis of inter-sample similarity structure
  - Sample similarity relationships more critical with few samples
- Contrast with sufficient samples
  - With sufficient data, models can self-learn similarity

#### 5.1.3 Why Is Performance Better with High Numerical Feature Proportion?
- Direct correlation between Euclidean distance and feature similarity
  - Numerical features provide continuous similarity signals
- Sparsity issues post-categorical encoding
  - One-hot encoding dimension explosion
  - Difficulty in similarity calculation on sparse representations
- Practical implications: Which data types are suitable for GNN enhancement

### 5.2 Boundary Conditions and Ineffective Scenarios for GNN Gains

#### 5.2.1 Failure with Large Datasets and Sufficient Samples
- Quantitative evidence: how gain varies with training ratio or dataset size
- Mechanistic analysis:
  - Large samples sufficient for model backbone learning
  - GNN structural information becomes redundant
- Resource-benefit analysis: computational overhead vs performance gain

#### 5.2.2 How Strong Baselines Diminish GNN Advantages
- Tree models' dominance with sufficient samples
  - Why harder to overcome tree models
  - Roots of advantage in specific data types (e.g., categorical-heavy)
- Pre-trained models (TabPFN) as alternative superiority
  - Pre-training vs GNN structural priors comparison
  - Future possibilities for combining pre-training + GNN

### 5.3 Model Dependency and Architecture Compatibility Analysis

#### 5.3.1 Why Are Some Models More GNN-Sensitive?
- FT-Transformer's high sensitivity
  - Transformer architecture characteristics
  - Token representation geometric properties
- SCARF's compatibility issues
  - Conflict between contrastive learning and GNN
  - Tension between instance discrimination and neighborhood aggregation

#### 5.3.2 Model Learning Objectives and GNN Inductive Bias Compatibility
- Reconstruction objectives vs message aggregation smoothing
- Classification objectives vs sample clustering trade-offs
- Future work: Adaptively select GNN configuration to match model objectives

### 5.4 Strategic Comparison: GNN Injection vs Self-contained GNN Design

#### 5.4.1 Verification of Position Alignment Hypothesis
- Map self-contained GNN model processing to five-stage framework
- Identify which stage DGM, LAN-GNN's GNNs correspond to
- Verification results: Does position alignment explain performance differences?

#### 5.4.2 Advantages of Hybrid Strategy (GNN Injection) Over Native Design
- Higher expressiveness: Leverage existing model feature extraction capabilities
- More flexible design space: Freedom to choose injection location
- Experimental evidence: When does GNN injection outperform native design?

#### 5.4.3 Practical Application Implications
- Strategy selection for different application scenarios

### 5.5 Research Limitations and Future Directions

#### 5.5.1 Explicit Research Limitations
- Dataset scale and diversity boundaries
  - Justification for 116 datasets (coverage explanation)
  - Potential representativeness bias
- Model type coverage
  - Whether 10 decomposable models sufficiently represent domain (coverage of trees, Transformers, attention, self-supervised, etc.)
  - Future expansion plans
- Simplified assumptions in graph construction strategies
  - Why k-NN chosen (vs alternative construction methods)
  - Potential impact on conclusion generalizability

#### 5.5.2 Future Research Opportunities

**Direction 1: Adaptive GNN Injection**
- Automatically select optimal stage based on data characteristics (size, feature type)
- Application of machine learning meta-learning

**Direction 2: Multi-stage Joint Injection**
- Synergistic effects of simultaneous GNN injection at multiple stages
- Design of inter-stage interactions

**Direction 3: Extended Application Scenarios**
- GNN enhancement for time-series tabular data
- Integration of multi-modal tables (text + numerical + images)
- Specialized analysis for imbalanced datasets

**Direction 4: Graph Construction Strategy Innovation**
- Learnable graph structure parameters
- Trade-offs between dynamic and static graphs
- Explicit integration of domain knowledge

---

## Chapter 6: Conclusion
**Target Word Count: 2,000-3,000 words**

### 6.1 Research Achievement Summary

#### 6.1.1 Core Contributions of ALIGN Framework
- Unified five-stage pipeline: Enables fair comparison across heterogeneous models
- Large-scale systematic evaluation: 116 datasets × 10 models × 6 stages = 13,920 experiments
- Quantified gain analysis: Clear decision basis for "when it works"

#### 6.1.2 Phase 1: Findings from Comprehensive Comparison
- Columnwise stage most consistently brings gains
- Most favorable: few-shot, small data, numerical features
- Boundary between tree models and GNN applicability

#### 6.1.3 Phase 2: Findings from Ablation Experiments
- Relative importance of training sample quantity, numerical feature proportion, dataset size
- Quantitative patterns of gain decrease
- Verification of cross-model consistency

### 6.2 Practical Guidelines and Decision Flow Chart

#### 6.2.1 When to Use GNN Injection (Decision Tree)
```
1. Dataset size < 5,000 rows?
   YES → 2
   NO → Abandon GNN
   
2. Training samples < 20% of total?
   YES → 3
   NO → Abandon GNN
   
3. Numerical feature proportion > 50%?
   YES → 4
   NO → Use GNN cautiously
   
4. Task type is binary classification?
   YES → Recommend columnwise stage GNN
   NO → Consider encoding/decoding stages
```

#### 6.2.2 Best Stage Selection Recommendations
- **First choice**: columnwise (highest stability)
- **Second choice**: encoding (token-level enhancement)
- **Backup**: decoding (end-to-end)
- **Avoid**: start/materialize (offline processing, unstable results)

#### 6.2.3 Resource-Benefit Trade-offs
- Computational overhead analysis (GNN layer count, hidden dimension impact on speed)
- Expected value of performance gains
- When additional cost is not worthwhile

### 6.3 Academic Impact and Open-source Contribution

#### 6.3.1 Implications for Tabular Deep Learning Community
- Challenge excessive optimism that "GNN is always beneficial"
- Systematic understanding of relationships between model architecture, data characteristics, and GNN gains
- Establish quantitative benchmarks for future research

#### 6.3.2 Open-source Resources and Reproducibility
- Public release of TaBLEau framework
- Unified preprocessing of 116 datasets
- Complete results from 13,920 experiments
- Scripts and results for three ablation studies

#### 6.3.3 Recommendations for Follow-up Work
- How community can leverage framework for new research
- Potential extensions and improvement directions

---

## Appendices

### Appendix A: Detailed Dataset List
- Names, sizes, task types, and feature distributions of 116 datasets
- Detailed statistical tables organized by three-dimensional classification

### Appendix B: Model Architecture Mapping Documents
- Detailed correspondence between 10 decomposable models and five-stage framework
- Implementation file locations and key code snippets for each model

### Appendix C: Complete Experimental Results Tables
- Detailed performance metrics (AUC/Accuracy/MAE) per model
- Win/tie/loss statistics against each baseline
- Results for all (model, dataset, stage) combinations

### Appendix D: Complete Visualization Sets for Ablation Experiments
- Performance and gain plots for all 10 models across three ablations
- Each plot includes: line chart + error bars + statistical annotations

### Appendix E: Hyperparameter Sensitivity Analysis
- Impact of GNN layer count on performance
- Trade-offs between hidden dimension and computational cost
- Effects of dropout ratio

### Appendix F: Implementation Details and Reference Code
- Pseudocode for GNN injection
- Python implementations of key functions
- Utility functions for result aggregation and visualization

### Appendix G: Ultra-large Scale Experiment Logs
- Execution statistics from Phase 1 (13,920 experiments)
- Time and resource usage reports from Phase 2 ablation studies

---

## 📊 Thesis Structure Statistics

| Chapter | Target Word Count | % of Total | Sections |
|---------|-------------------|------------|----------|
| Ch. 1 Introduction | 3,000-4,000 | 8-10% | 4 |
| Ch. 2 Related Work | 4,000-5,000 | 10-12% | 4 |
| Ch. 3 Methodology | 6,000-8,000 | 15-20% | 5 |
| Ch. 4 Results | 8,000-10,000 | 20-25% | 4 |
| Ch. 5 Discussion | 5,000-7,000 | 12-17% | 5 |
| Ch. 6 Conclusion | 2,000-3,000 | 5-8% | 3 |
| Appendices | 3,000-5,000 | 7-12% | 7 |
| **Total** | **31,000-42,000** | **100%** | **32** |

---

## 🎯 Thesis Writing Recommendations

### Logical Progression Advantages
- ✅ Clear progression from motivation → method → experiments → analysis → conclusion
- ✅ Combination of Phase 1 comprehensiveness with Phase 2 depth
- ✅ Unified framework, systematic analysis, new comparative dimensions

### Emphasis Strategy
- Showcase core findings with clear visualizations in Chapter 4
- Provide mechanistic analysis in Chapter 5, not just phenomenon description
- Offer actionable practical guidelines in Chapter 6

### Reproducibility Assurance
- Detailed methodology (Chapter 3)
- Open-source code and data (TaBLEau)
- Complete statistical reporting (Appendices)

---

**This thesis framework provides a systematic, comprehensive, and academically rigorous foundation for a high-quality master's thesis.**

---

*Document Generated: January 2026*
