import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import VGG19_Weights, vgg19
from torchvision.utils import save_image


class VGGFeatures(nn.Module):
    """Extract the VGG19 activations used for content/style losses."""

    def __init__(self) -> None:
        super().__init__()
        self.chosen_features = {0, 5, 10, 19, 28}
        self.model = vgg19(weights=VGG19_Weights.DEFAULT).features[:29].eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if layer_num in self.chosen_features:
                features.append(x)
        return features


def build_loader(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_image(
    path: Path, loader: transforms.Compose, device: torch.device
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        return loader(image.convert("RGB")).unsqueeze(0).to(device)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def gram_matrix(feature: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = feature.shape
    flattened = feature.reshape(batch, channels, height * width)
    return torch.bmm(flattened, flattened.transpose(1, 2))


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """Penalize neighboring-pixel differences in the generated image."""

    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return horizontal + vertical


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def clamp_normalized_image(image: torch.Tensor) -> None:
    """Project optimized pixels back to the valid ImageNet-normalized range."""

    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    image.copy_(torch.maximum(torch.minimum(image, upper), lower))


def run_style_transfer(args: argparse.Namespace) -> None:
    validate_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    loader = build_loader(args.image_size)

    content = load_image(args.content, loader, device)
    style = load_image(args.style, loader, device)
    generated = content.clone().requires_grad_(True)

    model = VGGFeatures().to(device)
    optimizer = optim.Adam([generated], lr=args.learning_rate)

    with torch.no_grad():
        content_features = model(content)
        style_features = model(style)
        style_grams = [gram_matrix(feature) for feature in style_features]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        generated_features = model(generated)
        content_loss = torch.tensor(0.0, device=device)
        style_loss = torch.tensor(0.0, device=device)

        for gen_feature, content_feature, style_gram in zip(
            generated_features, content_features, style_grams
        ):
            content_loss = content_loss + torch.mean((gen_feature - content_feature) ** 2)
            generated_gram = gram_matrix(gen_feature)
            style_loss = style_loss + torch.mean((generated_gram - style_gram) ** 2)

        variation_loss = total_variation(generated)
        total_loss = (
            args.content_weight * content_loss
            + args.style_weight * style_loss
            + args.tv_weight * variation_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            clamp_normalized_image(generated)

        if step == 1 or step % args.save_every == 0 or step == args.steps:
            output = denormalize(generated.detach().squeeze(0)).clamp(0, 1)
            save_image(output, args.output)
            print(
                f"step={step:5d} total={total_loss.item():.4f} "
                f"content={content_loss.item():.4f} "
                f"style={style_loss.item():.4f} tv={variation_loss.item():.4f}"
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size < 32 or args.steps < 1 or args.save_every < 1:
        raise ValueError("image_size must be >= 32; steps and save_every must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if min(args.content_weight, args.style_weight, args.tv_weight) < 0:
        raise ValueError("loss weights cannot be negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimization-based neural style transfer")
    parser.add_argument("--content", type=Path, default=Path("picture.png"))
    parser.add_argument("--style", type=Path, default=Path("style.jpg"))
    parser.add_argument("--output", type=Path, default=Path("generated.png"))
    parser.add_argument("--image-size", type=int, default=356)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=0.01)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_style_transfer(parse_args())
