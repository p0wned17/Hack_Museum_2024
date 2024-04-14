import torch
from pytorch_metric_learning import samplers

from . import augmentations, dataset


def get_train_dataloader(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")

    train_dataset = dataset.WBFlexDataset(
        root=config.dataset.root,
        annotation_file=config.dataset.train_list,
        transform=augmentations.get_train_aug(config),
    )

    sampler = samplers.MPerClassSampler(
        labels=train_dataset.labels[:, 0],
        m=config.train.mperclass,
        batch_size=config.dataset.batch_size,
        length_before_new_iter=len(train_dataset),
    )

    classes_count = train_dataset.classes_count

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        sampler=sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
    )
    print("Done.")

    return train_loader, classes_count


def get_query_gallery_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")

    query_dataset = dataset.WBFlexDataset(
        root=config.dataset.root,
        annotation_file=config.dataset.query_list,
        transform=augmentations.get_val_aug(config),
        val=True,
    )

    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    gallery_dataset = dataset.WBFlexDataset(
        root=config.dataset.root,
        annotation_file=config.dataset.gallery_list,
        transform=augmentations.get_val_aug(config),
        val=True,
    )

    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print("Done.")

    return query_loader, gallery_loader


def get_public_dataloaders(config):
    gallery_dataset = dataset.SubmissionDataset(
        root=config.dataset.public_dir,
        annotation_file=config.dataset.public_gallery_annotation,
        transforms=augmentations.get_val_aug(config),
    )

    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.dataset.num_workers,
    )

    query_dataset = dataset.SubmissionDataset(
        root=config.dataset.public_dir,
        annotation_file=config.dataset.public_query_annotation,
        transforms=augmentations.get_val_aug(config),
        with_bbox=True,
    )

    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.dataset.num_workers,
    )

    return gallery_loader, query_loader
