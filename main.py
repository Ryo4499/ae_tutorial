from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
from importlib import import_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from src.ae_tutorial.ae import AE
from src.ae_tutorial.dataset import gen_loaders
from src.ae_tutorial.utils.entity import SupportedDataset, SupportedArchitecture
from src.ae_tutorial.utils.result import visualize_imgs
from src.ae_tutorial.utils.set_seed import set_seed


def main():
    parser = ArgumentParser(
        prog="AE tutorial",
    )
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="exec without train",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=SupportedArchitecture.__members__.values(),
        default=SupportedArchitecture.CA,
        help="using architecture",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=SupportedDataset.__members__.values(),
        default=SupportedDataset.MNIST,
        help="using dataset",
    )
    args = parser.parse_args()
    set_seed()
    data_dir = "data"
    output_dir = f"results/datasets/{args.dataset}/{args.architecture}/images/"
    batch_size = 128
    train_loader, val_loader, test_loader = gen_loaders(
        args.dataset, data_dir, batch_size
    )
    x, _ = next(iter(deepcopy(val_loader)))
    match args.architecture:
        case SupportedArchitecture.FAA:
            in_features = x.shape[1:].numel()
            out_features = 64
            hidden_size = in_features // 2
            m = import_module(".ae_faa", "src.ae_tutorial")
            encoder = m.AEEncoder(in_features, out_features, hidden_size)
            decoder = m.AEDecoder(in_features, out_features, hidden_size)
        case SupportedArchitecture.CA:
            in_features = x.shape[1]
            out_features = 32
            hidden_size = out_features // 2
            m = import_module(".ae_ca", "src.ae_tutorial")
            encoder = m.AEEncoder(in_features, out_features, hidden_size)
            decoder = m.AEDecoder(in_features, out_features, hidden_size)
        case _:
            ValueError(f"{args.architecture} is not supported")

    model = AE(args, encoder, decoder)

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=10,
                check_on_train_epoch_end=True,
                verbose=True,
            ),
            LearningRateMonitor("epoch"),
        ],
    )
    if args.eval:
        ck_dir = Path("lightning_logs")
        sorted_cks = sorted(
            ck_dir.glob("**/*.ckpt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        latest_ck = None
        for i in sorted_cks:
            model = AE.load_from_checkpoint(i)
            if (
                model.architecture == args.architecture
                and model.dataset == args.dataset
            ):
                latest_ck = i
                break
        if latest_ck is None:
            raise ValueError("checkpoint not found")
    else:
        trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    visualize_imgs(args.architecture, output_dir, model, test_loader)


if __name__ == "__main__":
    main()
