import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .entity import SupportedArchitecture
from ..ae import AE


def imshow(
    idx: int,
    output_dir: str,
    x: torch.Tensor,
    x_hat: torch.Tensor | None = None,
):
    fig = plt.figure()
    if x_hat is not None:
        imgs = torch.stack([x, x_hat], dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(
            imgs, nrow=4, normalize=True, value_range=(-1, 1)
        )
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(7, 4.5))
        plt.title("Images Diff")
        plt.imshow(grid)
        plt.axis("off")
    else:
        fig.suptitle("batch_data", fontsize=14)
        ax1 = fig.add_axes((0.05, 0.1, 0.4, 0.7))
        ax1.set_title("org")
        img = torchvision.utils.make_grid(x.clone())
        img *= 255  # unnormalize
        npimg = img.to(torch.uint8).cpu().detach().permute(1, 2, 0).numpy()
        ax1.imshow(npimg)
    # plt.show()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"batch_{idx}.png"))
    plt.close(fig)


def visualize_imgs(
    architecture: SupportedArchitecture,
    output_dir: str,
    model: AE,
    dataloader: DataLoader,
):
    n = 10
    for idx, (x_org, _) in enumerate(dataloader):
        x = x_org.view(x_org.size(0), -1)
        match architecture:
            case SupportedArchitecture.FAA:
                h = model.encoder(x)
            case SupportedArchitecture.CA:
                h = model.encoder(x_org)
            case _:
                raise ValueError(f"{architecture} is not supported")
        x_hat = model.decoder(h)
        if SupportedArchitecture.CA:
            x = x_org.view(x_org.size(0), -1)
            x_hat = x_hat.view(x_hat.size(0), -1)
        if idx < n:
            imshow(
                idx,
                output_dir,
                x.reshape(x_org.shape)[0:10],
                x_hat.reshape(x_org.shape)[0:10],
            )
