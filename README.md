# Algorithmic seriation of the Voynich Manuscript

This repository contains code and datasets associated with the paper:

> Greshko, Michael A. (2026). Algorithmic seriation of the Voynich Manuscript
is most consistent with an all-singulion collation and suggests a bifolium-level reordering
of the codex. Accepted to the Second International Conference on the
Voynich Manuscript 2026.
  
## Abstract

Adding to the manuscript’s mysteries, the binding of the 
15th-century Voynich Manuscript (VMS) probably does not reflect its 
originally intended page ordering. Recently, 
a computational analysis of the VMS’s 
Biological (ff. 75–84) and Stars (ff. 103–116) sections suggested 
that the VMS may have been structured originally 
as consecutive loose bifolia that each acted 
as individual quires (singulions). In this work, I test this hypothesis 
by running both the VMS and synthetic collations of known texts 
through an algorithm that treats seriation of a manuscript’s 
folia as a Hamiltonian path optimization problem in latent topic space, based on 
an ensemble of nonnegative matrix factorization topic modeling runs. 
After baselining the algorithm’s performance by seriating 
synthetic collations of diverse Latin texts while varying 
quire structures, I seriate the VMS at the folio 
level. The VMS’s observed rate of conjoint folia becoming adjacent 
pairs within a topic-distance seriation is most consistent with the 
VMS being made of entirely singulions, although controls indicate this 
adjacency signal does not depend on whether 
the VMS is meaningful. Presupposing an all-singulion collation, I also propose 
a bifolium-level reordering of the VMS’s surviving bifolia that 
converges with previously proposed reorderings 
of the Biological and Stars sections.

You can view the proposed reordering here: https://vceditor.library.upenn.edu/project/6a6f5e5fef56250001215924/viewOnly

This repository also includes a JSON file of the reordering compatible with Voynich Viewer: https://voynichviewer.com/

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
is most consistent with an all-singulion collation and suggests a bifolium-level reordering
of the codex. Accepted to the Second International Conference on the
Voynich Manuscript 2026.
  
THE SOFTWARE AND DATASETS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE AND DATASETS.
