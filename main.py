import os
import pprint
import random
import warnings
import torch
import numpy as np
from trainer import Trainer, Tester

from config import getConfig
warnings.filterwarnings('ignore')
cfg = getConfig()


def main(cfg):
    print('<---- Training Params ---->')
    pprint.pprint(cfg)

    # Random Seed
    seed = cfg.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.action =='train':
        save_path = os.path.join(cfg.model_path, cfg.dataset, f'TE{cfg.arch}_{str(cfg.exp_num)}')

        # Create model directory
        os.makedirs(save_path, exist_ok=True)
        Trainer(cfg, save_path)

    else:
        save_path = os.path.join(cfg.model_path, cfg.dataset, f'TE{cfg.arch}_{str(cfg.exp_num)}')

        datasets = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']
        for dataset in datasets:
            cfg.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = Tester(cfg, save_path).test()

            print(f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.4f} '
                  f'| AVG_F:{test_avgf:.4f} | MAE:{test_mae:.4f} | S_Measure:{test_s_m:.4f}')


if __name__ == '__main__':
    main(cfg)