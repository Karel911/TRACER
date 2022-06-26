"""
author: Min Seok Lee and Wooseok Shin
"""
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataloader import get_test_augmentation
from model.TRACER import TRACER
from util.utils import load_pretrained
import torch.nn as nn
import urllib
from torchvision.transforms import transforms


class Inference():
    def __init__(self, args):
        super(Inference, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = get_test_augmentation(img_size=args.img_size)
        self.args = args

        self.invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                    ])

        # Network
        self.model = TRACER(args).to(self.device)
        self.model = nn.DataParallel(self.model).to(self.device)

        path = load_pretrained(f'TE-{args.arch}', self.device)
        self.model.load_state_dict(path)
        self.model.eval()
        print('###### pre-trained Model restored #####')


    def test(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        elif isinstance(image, str): # if path or URL
            if "http" in image or "https" in image:
                req = urllib.request.urlopen(image)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1) # 'Load it as it is'
            
            else: # if path in directory
                image = cv2.imread(image)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

        image = self.transform(image=image)['image']
       
        with torch.no_grad():
                image = torch.tensor(image.unsqueeze(0), device=self.device, dtype=torch.float32)

                output, edge_mask, ds_map = self.model(image)
                output = F.interpolate(output, size=(h, w), mode='bilinear')
                output = (output.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)  # convert uint8 type

                salient_object = self.post_processing(image, output, h, w)
                return output, salient_object

    
    def post_processing(self, original_image, output_image, height, width, threshold=200):
        
        original_image = self.invTrans(original_image)

        original_image = F.interpolate(original_image, size=(height, width), mode='bilinear')
        original_image = (original_image.squeeze().permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8)

        rgba_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
        output_rbga_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)

        output_rbga_image[:, :, 3] = output_image                     # Extract edges
        edge_y, edge_x, _ = np.where(output_rbga_image <= threshold)  # Edge coordinates
        
        rgba_image[edge_y, edge_x, 3] = 0
        return rgba_image

