import numpy as np
import torch
from accelerate import Accelerator
from mean_average_precision import calculate_map
from tqdm import tqdm
from utils import AverageMeter


def train(
    model: torch.nn.Module,
    accelerator: Accelerator,
    train_loader: torch.utils.data.DataLoader,
    class_loss_metric_fn: torch.nn.Module,
    category_loss_metric_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    epoch,
    scheduler,
) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch (int): epoch number
    :return: None
    """
    model.train()

    class_metric_loss_stat = AverageMeter("class_metric_loss")
    category_metric_loss_stat = AverageMeter("class_metric_second_loss")

    mean_loss_stat = AverageMeter("Mean Loss")

    train_iter = train_loader
    if accelerator.is_main_process:
        train_iter = tqdm(train_loader, desc="Train", dynamic_ncols=True, position=1)

    for step, (x, cat_labels, class_labels) in enumerate(train_iter, start=1):
        optimizer.zero_grad(set_to_none=True)

        embeddings = model(x.to(memory_format=torch.channels_last))

        class_metric_loss = class_loss_metric_fn(
            embeddings,
            class_labels,
        )
        category_metric_loss = category_loss_metric_fn(embeddings, cat_labels)

        num_of_samples = x.shape[0]
        loss = +0.5 * class_metric_loss + 0.5 * category_metric_loss
        accelerator.backward(loss)

        class_metric_loss_stat.update(
            class_metric_loss.detach().cpu().item(), num_of_samples
        )
        category_metric_loss_stat.update(
            category_metric_loss.detach().cpu().item(), num_of_samples
        )

        mean_loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        if accelerator.sync_gradients:
            accelerator.clip_grad_value_(model.parameters(), config.train.grad_clipping)
        optimizer.step()

        if step % config.train.freq_vis == 0 and not step == 0:
            _, class_metric_loss_avg = class_metric_loss_stat()
            _, category_metric_loss_avg = category_metric_loss_stat()
            _, avg_mean_loss = mean_loss_stat()

            if accelerator.is_main_process:
                print(
                    f"""Epoch {epoch}, step: {step}:
                        class_metric_loss: {class_metric_loss_avg},
                        category_metric_loss: {category_metric_loss_avg},
                        mean_loss: {avg_mean_loss}
                    """
                )

                model.train()

    _, class_metric_loss_avg = class_metric_loss_stat()
    _, category_metric_loss_avg = category_metric_loss_stat()
    _, avg_mean_loss = mean_loss_stat()
    if accelerator.is_main_process:
        print(
            f"""Train process of epoch {epoch} is done:
                        class_metric_loss: {class_metric_loss_avg},
                        category_metric_loss: {category_metric_loss_avg},
                        mean_loss: {avg_mean_loss}
                    """
        )
    return avg_mean_loss


def validation(
    model: torch.nn.Module,
    gallery_loader: torch.utils.data.DataLoader,
    query_loader: torch.utils.data.DataLoader,
    config,
) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
    """

    print("Calculating embeddings")
    gallery_embeddings = torch.zeros(
        (990, config.model.embedding_dim),
        device="cuda",
        requires_grad=False,
        dtype=torch.half,
    )
    query_embeddings = torch.zeros(
        (1005, config.model.embedding_dim),
        device="cuda",
        requires_grad=False,
        dtype=torch.half,
    )

    gallery_product_ids = np.zeros(990).astype(np.int32)
    query_product_ids = np.zeros(1005).astype(np.int32)

    model.eval()
    with torch.no_grad():
        gallery_ptr = 0
        for images, targets in tqdm(gallery_loader, total=len(gallery_loader)):
            outputs = model(images.half().cuda())
            batch_size = images.size(0)

            gallery_embeddings[gallery_ptr : gallery_ptr + batch_size, :] = outputs
            gallery_product_ids[gallery_ptr : gallery_ptr + batch_size] = (
                targets.cpu().numpy()
            )
            gallery_ptr += batch_size

        query_ptr = 0
        for images, targets in tqdm(query_loader, total=len(query_loader)):
            outputs = model(images.half().cuda())
            batch_size = images.size(0)

            query_embeddings[query_ptr : query_ptr + batch_size, :] = outputs
            query_product_ids[query_ptr : query_ptr + batch_size] = (
                targets.cpu().numpy()
            )
            query_ptr += batch_size

        print(query_product_ids)
        print(gallery_product_ids)
        print("Normalizing and calculating distances")

        import gc

        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.cuda.empty_cache()
            gc.collect()

        distances = torch.cdist(query_embeddings, gallery_embeddings)

        sorted_distances = torch.argsort(distances, dim=1)
        sorted_distances = sorted_distances.cpu().numpy()[:, :500]

        print(distances)
        print(sorted_distances)

        public_map = calculate_map(
            sorted_distances, query_product_ids, gallery_product_ids
        )

        with torch.no_grad():
            torch.cuda.empty_cache()

        return public_map
