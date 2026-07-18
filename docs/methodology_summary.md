# Methodology summary

The source domain is generated in OMNeT++/INET with AODV routing and three
operational states: Normal, Flooding, and Blackhole.

Continuous attributes are discretized into three quantile-based linguistic
states: Low, Medium, and High. The Bayesian Network is trained on the source
domain and evaluated internally using a stratified split.

For the external NS-3 assessment, packet-loss causes are mapped semantically to
the source operational states. Two discretization strategies are evaluated:

1. fixed source-domain thresholds;
2. unsupervised adaptive percentile thresholds calculated from the unlabeled
   target-domain attributes.

The Bayesian graph and source-domain conditional probability tables are
preserved during zero-shot external inference. Mutual Information is used as a
global explainability measure to compare attribute relevance between domains.
