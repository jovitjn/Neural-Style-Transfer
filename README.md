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

An optional total-variation term discourages abrupt neighboring-pixel changes. The
generated image is optimized with

\[
\mathcal{L}=\alpha\mathcal{L}_{content}+\beta\mathcal{L}_{style}
+\gamma\mathcal{L}_{TV}.
\]

## Current implementation

`code.py` includes:

- pretrained VGG19 feature extraction
- ImageNet normalization
- content loss
- Gram-matrix style loss
- direct optimization of image pixels with Adam
- optional total-variation regularization
- automatic CPU, CUDA, or Apple MPS device selection
- deterministic seeding, argument validation, and pixel-range projection
- configurable paths and optimization settings through a command-line interface
- periodic progress logging and image saving

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with your own content and style images:

```bash
python code.py \
  --content path/to/content.jpg \
  --style path/to/style.jpg \
  --output outputs/stylized.png \
  --steps 2000 \
  --image-size 356 \
  --tv-weight 1e-4
```

Run `python code.py --help` for all controls. With no arguments, the script reads
`picture.png` and `style.jpg`, then periodically overwrites `generated.png` with
the latest result. The first run downloads the pretrained VGG19 weights through
`torchvision`.

For a quick smoke test, use a smaller image and fewer steps:

```bash
python code.py --content picture.png --style style.jpg \
  --image-size 128 --steps 20 --save-every 10
```

No reference output is committed because results depend on user-supplied images.

## Scope

This is an **educational implementation**, not a novel style-transfer method. Its
value is in making perceptual feature losses and Gram-matrix statistics explicit
in a small, runnable program. It uses one fixed set of VGG19 feature layers and
does not claim real-time inference or a trained feed-forward stylization model.

## Reference

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge,
[**A Neural Algorithm of Artistic Style**](https://arxiv.org/abs/1508.06576), 2015.
