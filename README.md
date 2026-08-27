# Neural style transfer with VGG19

This is my implementation of the optimization-based style-transfer method from Gatys, Ecker, and Bethge. It starts from the content image and updates the image pixels directly: deeper VGG19 activations preserve the scene structure, while Gram matrices of shallower activations match the texture and colour statistics of the style image.

The loss is

\[
\mathcal{L}=\alpha\mathcal{L}_{content}
+\beta\mathcal{L}_{style}
+\gamma\mathcal{L}_{TV},
\]

where total variation is optional. VGG19 is used only as a fixed feature extractor; the script does not train a feed-forward stylization network.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python code.py \
  --content path/to/content.jpg \
  --style path/to/style.jpg \
  --output outputs/stylized.png \
  --steps 2000 \
  --image-size 356 \
  --tv-weight 1e-4
```

The first run downloads the pretrained VGG19 weights from PyTorch. The script automatically selects CUDA, Apple MPS, or CPU; use `--device` to override it.

For a short pipeline check:

```bash
python code.py \
  --content path/to/content.jpg \
  --style path/to/style.jpg \
  --output outputs/test.png \
  --image-size 128 \
  --steps 20 \
  --save-every 10
```

The historical defaults are `picture.png`, `style.jpg`, and `generated.png`. Supply those two input files or pass explicit paths as shown above.

## Implementation notes

- ImageNet normalization is applied before VGG feature extraction.
- Content loss compares intermediate activations directly.
- Style loss compares Gram matrices across the selected VGG layers.
- Adam optimizes only the generated image.
- Pixels are projected back to the valid ImageNet-normalized range after every step.
- `--seed` makes the run configuration repeatable, although device-level floating-point differences can remain.

This is an educational reproduction of a standard method, not a new style-transfer architecture. Output quality depends strongly on the image pair, weights, image size, and number of optimization steps.

## Reference

Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576), 2015.