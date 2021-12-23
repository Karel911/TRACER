import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Model Training or Testing options')
    parser.add_argument('--exp_num', default=0, type=str, help='experiment_number')
    parser.add_argument('--dataset', type=str, default='DUTS', help='DUTS')
    parser.add_argument('--data_path', type=str, default='data/')

    # Model parameter settings
    parser.add_argument('--arch', type=str, default='0', help='Backbone Architecture')
    parser.add_argument('--channels', type=list, default=[24, 40, 112, 320])
    parser.add_argument('--RFB_aggregated_channel', type=int, nargs='*', default=[32, 64, 128])
    parser.add_argument('--frequency_radius', type=int, default=16, help='Frequency radius r in FFT')
    parser.add_argument('--denoise', type=float, default=0.93, help='Denoising background ratio')
    parser.add_argument('--gamma', type=float, default=0.1, help='Confidence ratio')

    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--criterion', type=str, default='API', help='API or bce')
    parser.add_argument('--scheduler', type=str, default='Reduce', help='Reduce or Step')
    parser.add_argument('--aug_ver', type=int, default=2, help='1=Normal, 2=Hard')
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5, help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parser.add_argument('--model_path', type=str, default='results/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_map', type=bool, default=None, help='Save prediction map')


    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    cfg = parser.parse_args()

    return cfg


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)