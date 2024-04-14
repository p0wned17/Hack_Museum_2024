import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import utils
from accelerate import Accelerator, DistributedDataParallelKwargs
from data_utils import get_dataloader
from loguru import logger
from models.model import RetrivealNet, Trunk
from omegaconf import OmegaConf
from pytorch_metric_learning import distances, losses
from tqdm import tqdm
from train import train, validation


def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """

    config = OmegaConf.load(args.cfg)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    outdir = osp.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Preparing train and val dataloaders...")
    train_loader, classes_count = get_dataloader.get_train_dataloader(config)

    config.dataset.num_of_classes = classes_count
    query_loader, gallery_loader = get_dataloader.get_query_gallery_dataloaders(config)

    trunk = Trunk(
        backbone=config.model.backbone,
        embedding_dim=config.model.embedding_dim,
        dropout=config.train.dropout,
        pretrained=config.model.pretrained,
    )

    model = RetrivealNet(trunk=trunk)

    checkpoint = torch.load(
        "/mnt/hack_museums/inference_code/laion_ft_wb.pt",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint)

    print("Load model...")

    model.to(device, memory_format=torch.channels_last)

    print("Prepare training params...")

    distance = distances.CosineSimilarity()

    class_loss = losses.TripletMarginLoss(
        margin=0.2,
        swap=True,
        smooth_loss=True,
        triplets_per_anchor="all",
        distance=distance,
    )

    category_loss = losses.TripletMarginLoss(
        margin=0.4,
        swap=True,
        smooth_loss=True,
        triplets_per_anchor="all",
        distance=distance,
    )
    
    optimizer = torch.optim.AdamW(
        trunk.parameters(),
        betas=(config.train.adamw_beta1, config.train.adamw_beta2),
        lr=config.train.trunk.lr,
        weight_decay=config.train.trunk.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.2, last_epoch=-1, verbose=False
    )

    (
        model,
        optimizer,
        train_loader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        scheduler,
    )

    print("Done.")

    train_epoch = tqdm(
        range(config.train.n_epoch), dynamic_ncols=True, desc="Epochs", position=0
    )

    best_acc = 0


    for epoch in train_epoch:
        train_loss = train(
            model,
            accelerator,
            train_loader,
            class_loss,
            category_loss,
            optimizer,
            config,
            epoch,
            scheduler,
        )

        if accelerator.is_main_process:
            epoch_avg_acc = validation(model, gallery_loader, query_loader, config)

            logger.info(
                f"""Epoch {epoch}
                public mAP: {epoch_avg_acc}
                Train loss: {train_loss}
                """
            )

            saved_model = accelerator.unwrap_model(model)
            if epoch_avg_acc >= best_acc:
                best_acc = epoch_avg_acc
                epoch_avg_acc = f"{epoch_avg_acc:.4f}"

                utils.save_checkpoint(
                    saved_model,
                    class_loss,
                    optimizer,
                    scheduler,
                    epoch,
                    outdir,
                    epoch_avg_acc,
                )

        scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to config file.")
    parser.add_argument(
        "--chkp", type=str, default=None, help="Path to checkpoint file."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
