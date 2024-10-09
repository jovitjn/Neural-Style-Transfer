# Neural-Style-Transfer
This repository implements a basic Neural Style Transfer (NST) using PyTorch and VGG19. The code applies the artistic style of one image (style image) to another image (content image) to create a new, stylized image. This technique is inspired by the original paper "A Neural Algorithm of Artistic Style" by Gatys et al.

Neural Style Transfer is a deep learning technique that uses convolutional neural networks (CNNs) to apply the artistic style of one image onto another image while preserving its original content structure. The model uses a pretrained VGG19 network to extract content and style features and then optimizes a new image to minimize both content and style loss.
