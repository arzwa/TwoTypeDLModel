Data sets
=========

Orthogroups for all data sets were obtained using OrthoFinder, in all cases for
a data set that was larger than the one included here (see below). For all data
sets I obtained the subset of gene families from the `N0.tsv` file from
OrthoFinder where there was at least one copy in each clade stemming from the
root of the relevant subtree (see below). Families where some species had more
then 10 copies were filtered out.

1. Drosophila data
------------------
Sequence data was obtained from NCBI, where I took the longest isoform as a
representative for each gene. The complete data set consists of the 12
Drosophila species from Eisen et al. (also see Hahn et al. 2007). The dated
tree for the eight-taxon subset was obtained from TimeTree.

2. Primates data set
--------------------
Sequence data was obtained from Ensembl. The complete data set includes besides
the 11 primate species also rat and mouse. The rest is analogous to the
Drosophila data set. For the primates data we consider subsets based on the
human gene onotology annotations. We obtained the GO annotation for all human
genes in the data, and derive a subset of families associated with some term,
e.g. GO:0002736, by identifying all families that have a homolog in *H.
sapiens* that is annotated with a the relevant term or any daughter term in the
GO graph.

3. Yeast data set (YGOB)
------------------------
Sequence data was obtained from the Yeast Gene Order Browser (YGOB), including
all species included in YGOB. An eight species subset consisting of those
species that have *not* undergone the well-described yeast WGD was obtained. We
used r8s to obtain a dated tree from the OrthoFinder species tree with the root
of the complete YGOB data set (MRCA of *K. lactis* and *S.  cerevisiae*) fixed
to 112 My based on Beimforde et al. (2014) assuming the molecular clock
(cross-validation using the penalized likelihood method suggested large
smoothing factors, i.e. clock-like substitution rates, which could be suspected
from an inspection of the molecular distances in the OrthoFinder species tree
as well.)

