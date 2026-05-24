"""
Export sample images from CIFAR-10 dataset for visual review.
"""
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.utils import save_image


def export_samples(output_dir='dataset_samples', num_samples=10):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    classes = dataset.classes

    for i in range(min(num_samples, len(dataset))):
        image, label = dataset[i]
        filename = output_path / f"sample_{i+1}_{classes[label]}.png"
        save_image(image, filename)
        print(f"Saved {filename}")

    print(f"\nExported {min(num_samples, len(dataset))} sample images to '{output_path.resolve()}'")


if __name__ == '__main__':
    export_samples()
