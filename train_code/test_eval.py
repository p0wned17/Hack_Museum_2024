import torch
import numpy as np
from tqdm import tqdm
from mean_average_precision import calculate_map
from data_utils import dataset


def db_augmentation(query_vecs, reference_vecs, top_k=3):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    weights = torch.logspace(0, -2.0, top_k + 1).cuda()

    sim_mat = torch.cdist(query_vecs, reference_vecs)

    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = torch.tensordot(
        weights,
        torch.cat([torch.unsqueeze(query_vecs, 1), top_k_ref], dim=1),
        dims=([0], [1]),
    )

    sim_mat = torch.cdist(reference_vecs, reference_vecs)
    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref = reference_vecs[indices[:, : top_k + 1], :]
    reference_vecs = torch.tensordot(weights, top_k_ref, dims=([0], [1]))

    return query_vecs, reference_vecs


def get_wb_val_dataloaders():
    query_dataset = dataset.WBValDataset(
        root="/mnt/wb_products_dataset",
        annotation_file="/mnt/wb_products_dataset/testrerank2.csv",
        mode="query",
    )

    gallery_dataset = dataset.WBValDataset(
        root="/mnt/wb_products_dataset",
        annotation_file="/mnt/wb_products_dataset/testrerank2.csv",
        mode="gallery",
    )
    print("wb gallery, query", len(gallery_dataset), len(query_dataset))
    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=30,
    )

    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=30,
    )

    return gallery_loader, query_loader


def validation_wb(
    gallery_loader: torch.utils.data.DataLoader,
    query_loader: torch.utils.data.DataLoader,
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
        (9066, 2048), device="cuda", requires_grad=False, dtype=torch.half
    )
    query_embeddings = torch.zeros(
        (9066, 2048), device="cuda", requires_grad=False, dtype=torch.half
    )

    gallery_product_ids = np.zeros(9066).astype(np.int32)
    query_product_ids = np.zeros(9066).astype(np.int32)

    with torch.no_grad():
        for i, (outputs, targets) in tqdm(
            enumerate(gallery_loader), total=len(gallery_loader)
        ):

            outputs = outputs.half().cuda()

            gallery_embeddings[
                i * 128 : (i * 128 + 128),
                :,
            ] = outputs
            gallery_product_ids[i * 128 : (i * 128 + 128)] = targets

        for i, (outputs, targets) in tqdm(
            enumerate(query_loader), total=len(query_loader)
        ):

            outputs = outputs.half().cuda()

            query_embeddings[
                i * 128 : (i * 128 + 128),
                :,
            ] = outputs

            query_product_ids[i * 128 : (i * 128 + 128)] = targets

        query_embeddings, gallery_embeddings = (
            query_embeddings.float(),
            gallery_embeddings.float(),
        )
        concat = torch.cat((query_embeddings, gallery_embeddings), dim=0)
        center = torch.mean(concat, dim=0)
        query_embeddings = query_embeddings - center
        gallery_embeddings = gallery_embeddings - center
        gallery_embeddings = torch.nn.functional.normalize(
            gallery_embeddings, p=2.0, dim=1
        )
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2.0, dim=1)

        query_embeddings, gallery_embeddings = db_augmentation(
            query_embeddings, gallery_embeddings, top_k=6
        )

        distances = torch.cdist(query_embeddings, gallery_embeddings)

        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        class_ranks = sorted_indices

        first_gallery_idx = class_ranks[:, 0]
        first_gallery_dstx = sorted_distances[:, 0]

        rerank_embeddings1 = gallery_embeddings.index_select(0, first_gallery_idx)

        mask1 = first_gallery_dstx < 0.8322

        filter_rerank_embeddings1 = torch.where(
            mask1.view(-1, 1), rerank_embeddings1, query_embeddings
        )

        filter_rerank_embeddings = (
            0.5 * filter_rerank_embeddings1 + 0.5 * query_embeddings
        )
        distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)

        sorted_distances, sorted_indices = torch.sort(distances, dim=1)
        first_gallery_idx = class_ranks[:, 0]
        first_gallery_dstx = sorted_distances[:, 0]
        second_gallery_idx = class_ranks[:, 1]
        second_gallery_dstx = sorted_distances[:, 1]

        rerank_embeddings1 = gallery_embeddings.index_select(0, first_gallery_idx)
        rerank_embeddings2 = gallery_embeddings.index_select(0, second_gallery_idx)

        mask1 = first_gallery_dstx < 0.8322
        mask2 = second_gallery_dstx < 0.8322

        filter_rerank_embeddings1 = torch.where(
            mask1.view(-1, 1), rerank_embeddings1, query_embeddings
        )
        filter_rerank_embeddings2 = torch.where(
            mask2.view(-1, 1), rerank_embeddings2, query_embeddings
        )

        filter_rerank_embeddings = (
            0.275 * filter_rerank_embeddings1
            + 0.275 * filter_rerank_embeddings2
            + 0.45 * query_embeddings
        )

        distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)

        sorted_distances = torch.argsort(distances, dim=1)
        sorted_distances = sorted_distances.cpu().numpy()[:, :1000]
        class_ranks = sorted_distances

        public_map = calculate_map(
            sorted_distances, query_product_ids, gallery_product_ids
        )

        return public_map


def main():
    gallery_dataloader, query_dataloader = get_wb_val_dataloaders()

    map = validation_wb(gallery_dataloader, query_dataloader)
    print(map)


if __name__ == "__main__":
    main()
