# Algorithmic seriation of the entire Voynich Manuscript

This repository contains code and datasets associated with the paper:

> Greshko, Michael A. (2026). Algorithmic seriation of the Voynich Manuscript
supports an all-singulion construction and yields a bifolium-level reordering
of the entire codex. Submitted to the Second International Conference on the
Voynich Manuscript 2026.
  
## Abstract

Adding to the enigmatic codex’s mysteries, the binding of the
15th-century Voynich Manuscript (VMS) probably does not reflect its
original structure. Recently, a computational analysis of the VMS’s
Stars section (ff. 103–116) suggested that the VMS may have been
structured originally as consecutive loose bifolia that each acted
as individual quires (singulions). In this work, I test this hypothesis
by running both the VMS and synthetic collations of known texts
through an algorithm that treats seriation of a manuscript’s
folia as a Hamiltonian path problem in latent topic space, based on
an ensemble of nonnegative matrix factorization topic modeling runs.
After establishing a baseline for the algorithm’s performance
by seriating synthetic collations of diverse medieval and classical
texts while varying quire structures, I seriate the VMS at the folio
level. The VMS’s observed rate of bifolium-conjoint folia becoming adjacent
pairs within a topic-distance seriation is most consistent with the
VMS being made of entirely singulions. To assist future codicological
studies, I also propose a bifolium-level reordering of the entire VMS
that replicates a previously proposed singulion-based
rearrangement of the Stars section.

# License and copyright

Unless otherwise indicated, the source code contained in
this repository is provided under the modified MIT license below.

---

Copyright (c) 2026, Michael A. Greshko.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software, datasets, and associated documentation files (the "Software
and Datasets"), to deal in the Software and Datasets without restriction,
including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software and Datasets, and to
permit persons to whom the Software is furnished to do so, subject to the
following conditions:

- The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software and Datasets.
- Any publications making use of the Software and Datasets, or any substantial
  portions thereof, shall cite the Software and Datasets's original publication:

> Greshko, Michael A. (2026). Algorithmic seriation of the Voynich Manuscript
supports an all-singulion construction and yields a bifolium-level reordering
of the entire codex. Submitted to the Second International Conference on the
Voynich Manuscript 2026.
  
THE SOFTWARE AND DATASETS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE AND DATASETS.
