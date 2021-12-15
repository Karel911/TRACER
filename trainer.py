"""
author: Min Seok Lee and Wooseok Shin
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import get_train_augmentation, get_test_augmentation, get_loader, gt_to_tensor
from util.utils import AvgMeter
from util.metrics import Evaluation_metrics
from util.losses import Optimizer, Scheduler, Criterion
from model.TRACER import TRACER


class Trainer():
    def __init__(self, args, save_path):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = args.img_size

        self.tr_img_folder = os.path.join(args.data_path, args.dataset, 'Train/images/')
        self.tr_gt_folder = os.path.join(args.data_path, args.dataset, 'Train/masks/')
        self.tr_edge_folder = os.path.join(args.data_path, args.dataset, 'Train/edges/')

        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=args.aug_ver)
        self.test_transform = get_test_augmentation(img_size=args.img_size)

        self.train_loader = get_loader(self.tr_img_folder, self.tr_gt_folder, self.tr_edge_folder, phase='train',
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       transform=self.train_transform, seed=args.seed)
        self.val_loader = get_loader(self.tr_img_folder, self.tr_gt_folder, self.tr_edge_folder, phase='val',
                                     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                     transform=self.test_transform, seed=args.seed)

        # Network
        self.model = TRACER(args).to(self.device)

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Loss and Optimizer
        self.criterion = Criterion(args)
        self.optimizer = Optimizer(args, self.model)
        self.scheduler = Scheduler(args, self.optimizer)

        # Train / Validate
        min_loss = 1000
        early_stopping = 0
        t = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            train_loss, train_mae = self.training(args)
            val_loss, val_mae = self.validate()

            if args.scheduler == 'Reduce':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Save models
            if val_loss < min_loss:
                early_stopping = 0
                best_epoch = epoch
                best_mae = val_mae
                min_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            if early_stopping == args.patience + 5:
                break

        print(f'\nBest Val Epoch:{best_epoch} | Val Loss:{min_loss:.3f} | Val MAE:{best_mae:.3f} '
              f'time: {(time.time() - t) / 60:.3f}M')

        # Test time
        datasets = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']
        for dataset in datasets:
            args.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = self.test(args, os.path.join(save_path))

            print(
                f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.3f} | AVG_F:{test_avgf:.3f} | MAE:{test_mae:.3f} '
                f'| S_Measure:{test_s_m:.3f}, time: {time.time() - t:.3f}s')

        end = time.time()
        print(f'Total Process time:{(end - t) / 60:.3f}Minute')

    def training(self, args):
        self.model.train()
        train_loss = AvgMeter()
        train_mae = AvgMeter()

        for images, masks, edges in tqdm(self.train_loader):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            masks = torch.tensor(masks, device=self.device, dtype=torch.float32)
            edges = torch.tensor(edges, device=self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            outputs, edge_mask, ds_map = self.model(images)
            loss1 = self.criterion(outputs, masks)
            loss2 = self.criterion(ds_map[0], masks)
            loss3 = self.criterion(ds_map[1], masks)
            loss4 = self.criterion(ds_map[2], masks)

            loss_mask = self.criterion(edge_mask, edges)
            loss = loss1 + loss2 + loss3 + loss4 + loss_mask

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
            self.optimizer.step()

            # Metric
            mae = torch.mean(torch.abs(outputs - masks))

            # log
            train_loss.update(loss.item(), n=images.size(0))
            train_mae.update(mae.item(), n=images.size(0))

        print(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        print(f'Train Loss:{train_loss.avg:.3f} | MAE:{train_mae.avg:.3f}')

        return train_loss.avg, train_mae.avg

    def validate(self):
        self.model.eval()
        val_loss = AvgMeter()
        val_mae = AvgMeter()

        with torch.no_grad():
            for images, masks, edges in tqdm(self.val_loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                masks = torch.tensor(masks, device=self.device, dtype=torch.float32)
                edges = torch.tensor(edges, device=self.device, dtype=torch.float32)

                outputs, edge_mask, ds_map = self.model(images)
                loss1 = self.criterion(outputs, masks)
                loss2 = self.criterion(ds_map[0], masks)
                loss3 = self.criterion(ds_map[1], masks)
                loss4 = self.criterion(ds_map[2], masks)

                loss_mask = self.criterion(edge_mask, edges)
                loss = loss1 + loss2 + loss3 + loss4 + loss_mask

                # Metric
                mae = torch.mean(torch.abs(outputs - masks))

                # log
                val_loss.update(loss.item(), n=images.size(0))
                val_mae.update(mae.item(), n=images.size(0))

        print(f'Valid Loss:{val_loss.avg:.3f} | MAE:{val_mae.avg:.3f}')
        return val_loss.avg, val_mae.avg

    def test(self, args, save_path):
        path = os.path.join(save_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(path))
        print('###### pre-trained Model restored #####')

        te_img_folder = os.path.join(args.data_path, args.dataset, 'Test/images/')
        te_gt_folder = os.path.join(args.data_path, args.dataset, 'Test/masks/')
        test_loader = get_loader(te_img_folder, te_gt_folder, edge_folder=None, phase='test',
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, transform=self.test_transform)

        self.model.eval()
        test_loss = AvgMeter()
        test_mae = AvgMeter()
        test_maxf = AvgMeter()
        test_avgf = AvgMeter()
        test_s_m = AvgMeter()

        Eval_tool = Evaluation_metrics(args.dataset, self.device)

        with torch.no_grad():
            for i, (images, masks, original_size, image_name) in enumerate(tqdm(test_loader)):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)

                outputs, edge_mask, ds_map = self.model(images)
                H, W = original_size

                for i in range(images.size(0)):
                    mask = gt_to_tensor(masks[i])

                    h, w = H[i].item(), W[i].item()

                    output = F.interpolate(outputs[i].unsqueeze(0), size=(h, w), mode='bilinear')

                    loss = self.criterion(output, mask)

                    # Metric
                    mae, max_f, avg_f, s_score = Eval_tool.cal_total_metrics(output, mask)

                    # log
                    test_loss.update(loss.item(), n=1)
                    test_mae.update(mae, n=1)
                    test_maxf.update(max_f, n=1)
                    test_avgf.update(avg_f, n=1)
                    test_s_m.update(s_score, n=1)

            test_loss = test_loss.avg
            test_mae = test_mae.avg
            test_maxf = test_maxf.avg
            test_avgf = test_avgf.avg
            test_s_m = test_s_m.avg

        return test_loss, test_mae, test_maxf, test_avgf, test_s_m


class Tester():
    def __init__(self, args, save_path):
        super(Tester, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_transform = get_test_augmentation(img_size=args.img_size)
        self.args = args
        self.save_path = save_path

        # Network
        self.model = self.model = TRACER(args).to(self.device)
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        path = os.path.join(save_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(path))
        print('###### pre-trained Model restored #####')

        self.criterion = Criterion(args)

        te_img_folder = os.path.join(args.data_path, args.dataset, 'Test/images/')
        te_gt_folder = os.path.join(args.data_path, args.dataset, 'Test/masks/')
        self.test_loader = get_loader(te_img_folder, te_gt_folder, edge_folder=None, phase='test',
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, transform=self.test_transform)

    def test(self):
        self.model.eval()
        test_loss = AvgMeter()
        test_mae = AvgMeter()
        test_maxf = AvgMeter()
        test_avgf = AvgMeter()
        test_s_m = AvgMeter()
        t = time.time()

        Eval_tool = Evaluation_metrics(self.args.dataset, self.device)

        with torch.no_grad():
            for i, (images, masks, original_size, image_name) in enumerate(tqdm(self.test_loader)):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)

                outputs, edge_mask, ds_map = self.model(images)
                H, W = original_size

                for i in range(images.size(0)):
                    mask = gt_to_tensor(masks[i])
                    h, w = H[i].item(), W[i].item()

                    output = F.interpolate(outputs[i].unsqueeze(0), size=(h, w), mode='bilinear')
                    loss = self.criterion(output, mask)

                    # Metric
                    mae, max_f, avg_f, s_score = Eval_tool.cal_total_metrics(output, mask)

                    # log
                    test_loss.update(loss.item(), n=1)
                    test_mae.update(mae, n=1)
                    test_maxf.update(max_f, n=1)
                    test_avgf.update(avg_f, n=1)
                    test_s_m.update(s_score, n=1)

            test_loss = test_loss.avg
            test_mae = test_mae.avg
            test_maxf = test_maxf.avg
            test_avgf = test_avgf.avg
            test_s_m = test_s_m.avg

        print(f'Test Loss:{test_loss:.4f} | MAX_F:{test_maxf:.4f} | MAE:{test_mae:.4f} '
              f'| S_Measure:{test_s_m:.4f}, time: {time.time() - t:.3f}s')

        return test_loss, test_mae, test_maxf, test_avgf, test_s_m