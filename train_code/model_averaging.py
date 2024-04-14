import argparse
import os
import random

import numpy as np
import torch

from omegaconf import OmegaConf
from data_utils import get_dataloader
from MCS2023_baseline.models.model import WBNet, Trunk

from eval import validation_public


def get_model(model_path, config):
    status = True
    trunk = Trunk(backbone=config.model.backbone, 
                        embedding_dim=config.model.embedding_dim,
                    pretrained=config.model.pretrained)
    model = WBNet(trunk=trunk)
    try:
        checkpoint = torch.load(model_path,
                                    map_location='cpu')['state_dict']
        model.load_state_dict(checkpoint)
    except:
        model, trunk, status = None, None, False
    return model, trunk, status


def get_score(path):
    '/home/cv_user/visual-product-recognition-2023-giga-flex/MCS2023_baseline/experiments/cleanwb_all_convnext_2048_256/model_0000_0.6173.pth'
    score = int(path.split('.')[-2])
    return score
    
def main() -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    
    config = OmegaConf.load('/home/cv_user/visual-product-recognition-2023-giga-flex/MCS2023_baseline/config/convnext.yml')
    device = torch.device('cuda:1')

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    print("Preparing train and val dataloaders...")

    gallery_loader, query_loader = get_dataloader.get_public_dataloaders(config)

    folders_with_models = [
        
        './experiments/ft_on_products10k',
        './experiments/train_on_wb',
        
    ]
    model_paths = []
    for folder_path in folders_with_models:
        model_paths.extend([os.path.join(folder_path, x) for x in os.listdir(folder_path)])

    #specify best_checkpoint!
    first_model_path = '/home/cv_user/visual-product-recognition-2023-giga-flex/MCS2023_baseline/experiments/cleanwb_20plus_convnext_2048_224_ft_product10k/model_0002_0.6261.pth' #           0.6388516909462032

    first_model, trunk, status = get_model(first_model_path, config)
    first_model.to(device)
    fisrt_model_map = validation_public(
            first_model.half(), config, gallery_loader, query_loader, 0
        )
    print('fisrt_model_map', fisrt_model_map)
    models_count = 2
    print("Load model...")
    lls = list(map(lambda x: get_score(x) > 6100, model_paths))
    models_names = []
    print('lls', len(lls), sum(lls))
    model_paths.sort(key=get_score, reverse=True)
    model_paths = model_paths[:sum(lls)]
    # random.shuffle(model_paths)
    for model_path in  model_paths:
        if not os.path.isfile(model_path):
            continue 
        additional_model, trunk, status = get_model(model_path, config)
        if not status:
            print('Cant load model')
            print(model_path)
            print('='*60)
            print('-'*60)
            continue
        additional_model.to(device)
        sdN = [m.state_dict() for m in [first_model, additional_model]]

        b = 1 / models_count
        a = 1 - b
        for key in sdN[0]:
            # sdN[0][key] = sum([sdi[key] for sdi in sdN]) / float(len(sdN))
            sdN[1][key] = a * sdN[0][key] + b * sdN[1][key]
            

        model = WBNet(trunk=trunk).to(device)
        model.load_state_dict(sdN[1])
        model.to(device)
        epoch_avg_map = validation_public(
            model.half(), config, gallery_loader, query_loader, 0
        )
        if epoch_avg_map >= fisrt_model_map:
            models_names.append(model_path)
            first_model = model
            fisrt_model_map = epoch_avg_map
            models_count += 1
            print('change model, now map:', epoch_avg_map)
        print('='*60)
        print('-'*60)
        # else:
            # fisrt_model_map2 = validation_public(
            #             first_model.half(), config, gallery_loader, query_loader, 0
            #             )
            # print('fisrt_model_map2', fisrt_model_map2)
            
    torch.save(first_model.state_dict(), f"/home/cv_user/visual-product-recognition-2023-giga-flex/MCS2023_baseline/experiments/Ave/Greedy_soup_{fisrt_model_map}.pt")
    print(models_names)
    print('map', fisrt_model_map)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to config file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
