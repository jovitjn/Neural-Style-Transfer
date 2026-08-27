# Neural Style Transfer with VGG19

A compact PyTorch implementation of **optimization-based neural style transfer**, inspired by Gatys et al., *A Neural Algorithm of Artistic Style*.

The script starts from the content image itself and optimizes its pixels so that deep VGG19 features preserve the content structure of one image while matching the style statistics of another.

## Method

A pretrained VGG19 network is used only as a feature extractor. Intermediate activations are collected from several convolutional layers.

The content objective compares generated and content features directly:

\[
\mathcal{L}_{content}=\sum_l \|F_l(x)-F_l(c)\|_2^2.
\]

Style is represented using Gram matrices:

\[
G_l(x)=F_l(x)F_l(x)^T,
\]

with style loss

\[
\mathcal{L}_{style}=\sum_l \|G_l(x)-G_l(s)\|_2^2.
\]

The generated image is optimized with

\[
\mathcal{L}=\alpha\mathcal{L}_{content}+\beta\mathcal{L}_{style}.
\]

## Current implementation

`code.py` includes:

- pretrained VGG19 feature extraction
- ImageNet normalization
- content loss
- Gram-matrix style loss
- direct optimization of image pixels with Adam
- GPU support when CUDA is available
- periodic saving of the generated image

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Place two images in the repository root:

```text
picture.png   # content image
style.jpg     # style reference
```

Then run:

```bash
python code.py
```

The script periodically writes `generated.png`.

## Scope

This is an **educational implementation**, not a novel style-transfer method. Its value is in demonstrating how perceptual feature losses and Gram-matrix statistics are constructed explicitly.

A stronger extension would add command-line arguments, configurable layer weights, total-variation regularization, better output logging, and a side-by-side result gallery.

## Reference

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, **A Neural Algorithm of Artistic Style**, 2015.
