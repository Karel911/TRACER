import argparse

class DummyArgs():
    def __init__(self, arch = 7):
        d = {0:320, 1:320, 2:352, 3:384, 4:448, 5:512, 6:576, 7:640}
        self.arch = str(arch)
        self.channels = [24, 40, 112, 320]
        self.RFB_aggregated_channel = [32, 64, 128]
        self.frequency_radius = 16
        self.denoise = 0.93
        self.gamma = 0.1
        self.multi_gpu = False
        self.img_size = d[int(arch)] # image_size is based on architecture


def getConfig():
    with open ('./arch.txt') as f: arch = int(f.read())
    return DummyArgs(arch)


if __name__ == '__main__':
    cfg = getConfig()