[![build](https://github.com/arzwa/TwoTypeDLModel/actions/workflows/workflow.yaml/badge.svg)](https://github.com/arzwa/TwoTypeDLModel/actions/workflows/workflow.yaml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/TwoTypeDLModel/dev/index.html)

Arthur Zwaenepoel 2021 <arzwa@psb.vib-ugent.be>

# TwoTypeDLModel

Inference for a two-type branching process model of gene family evolution by
duplication and loss. The main aim is more realistic estimation of gene
duplication and loss rates from gene family data. The method for computing
transition probabilities is based on the work of Xu et al. (2015) on the
Birth-Death-Shift process.

The model is a two-type continuous-time branching process with stochastic
dynamics that could be schematically represented as

```
1 → 12   with rate λ
1 →  ∅   with rate μ₁
2 → 22   with rate λ
2 →  ∅   with rate μ₂ > μ₁ and μ₂ > λ + ν
2 →  1   with rate ν
```

A special case of this model that may be of interest is the `μ₁ = 0` case.
The case `μ₁ = ν = 0` corresponds to a linear birth-death immigration process
incremented by one with immigration and birth rate `λ` and death rate `μ₂`.

Here we can roughly think of the number of type 1 particles denoting the number
of groups of redundant genes in a family, or the number of more or less
essential genes per family, while the number of type 2 particles reflects the
number of excess genes per family. The most important aspect of the model is
that type 2 genes get lost at a higher rate than type 1 genes, and that type 2
genes can get established (capturing the processes of 'complete' sub- and
neofunctionalization) and become stably incorporated type 1 genes.

For more information, consult the
[documentation](https://arzwa.github.io/TwoTypeDLModel/dev/index.html)

## Reference

Zwaenepoel & Van de Peer *in preparation*

