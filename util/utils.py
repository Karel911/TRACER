import torch
from torch.utils import model_zoo

def to_array(feature_map):
    if feature_map.shape[0] == 1:
        feature_map = feature_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    else:
        feature_map = feature_map.permute(0, 2, 3, 1).detach().cpu().numpy()
    return feature_map

def to_tensor(feature_map):
    return torch.as_tensor(feature_map.transpose(0, 3, 1, 2), dtype=torch.float32)

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)


url_TRACER = {
    'TE-0': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-0.pth',
    'TE-1': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-1.pth',
    'TE-2': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-2.pth',
    'TE-3': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-3.pth',
    'TE-4': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-4.pth',
    'TE-5': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-5.pth',
    'TE-6': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-6.pth',
    'TE-7': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-7.pth',
}


def load_pretrained(model_name, device):
    state_dict = model_zoo.load_url(url_TRACER[model_name], map_location = device)

    return state_dict
