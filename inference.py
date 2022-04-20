"""
author: Min Seok Lee and Wooseok Shin
"""
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import get_test_augmentation, get_loader
from model.TRACER import TRACER
from util.utils import load_pretrained

class Inference():
    def __init__(self, args, save_path):
        super(Inference, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_transform = get_test_augmentation(img_size=args.img_size)
        self.args = args
        self.save_path = save_path

        # Network
        self.model = TRACER(args).to(self.device)
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        path = load_pretrained(f'TE-{args.arch}')
        self.model.load_state_dict(path)
        print('###### pre-trained Model restored #####')

        te_img_folder = os.path.join(args.data_path, args.dataset)
        te_gt_folder = None

        self.test_loader = get_loader(te_img_folder, te_gt_folder, edge_folder=None, phase='test',
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, transform=self.test_transform)

        if args.save_map is not None:
            os.makedirs(os.path.join('pred_map', self.args.dataset), exist_ok=True)

    def test(self):
        self.model.eval()
        t = time.time()

        with torch.no_grad():
            for i, (images, original_size, image_name) in enumerate(tqdm(self.test_loader)):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)

                outputs, edge_mask, ds_map = self.model(images)
                H, W = original_size

                for i in range(images.size(0)):
                    h, w = H[i].item(), W[i].item()
                    output = F.interpolate(outputs[i].unsqueeze(0), size=(h, w), mode='bilinear')

                    # Save prediction map
                    if self.args.save_map is not None:
                        output = (output.squeeze().detach().cpu().numpy() * 255.0).astype \
                            (np.uint8)  # convert uint8 type
                        cv2.imwrite(os.path.join('pred_map', self.args.dataset, image_name[i] + '.png'), output)

        print(f'time: {time.time() - t:.3f}s')
