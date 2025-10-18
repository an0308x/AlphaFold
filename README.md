# AlphaFold Educational Components

This repository contains two PyTorch implementations of core AlphaFold modules
that are useful for experimentation and teaching:

- `IPA.py` provides the **Invariant Point Attention** layer.  The module mixes
  scalar features with point-based representations while remaining equivariant
  to rigid-body transformations.
- `MSATransformer.py` implements a compact version of the **MSA Transformer**
  stack used to process multiple sequence alignments and pair features.

Both modules have extensive documentation and follow the notation introduced in
*Highly accurate protein structure prediction with AlphaFold* (Jumper et al.,
2021).  They can serve as a reference when prototyping new ideas or porting the
model to different frameworks.
