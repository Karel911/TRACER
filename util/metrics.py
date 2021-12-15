import torch
import numpy as np


class Evaluation_metrics():
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

        print(f'Dataset:{self.dataset}')

    def cal_total_metrics(self, pred, mask):
        # MAE
        mae = torch.mean(torch.abs(pred - mask)).item()
        # MaxF measure
        beta2 = 0.3
        prec, recall = self._eval_pr(pred, mask, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        max_f = f_score.max().item()
        # AvgF measure
        avg_f = f_score.mean().item()
        # S measure
        alpha = 0.5
        y = mask.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            Q = alpha * self._S_object(pred, mask) + (1 - alpha) * self._S_region(pred, mask)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        s_score = Q.item()

        return mae, max_f, avg_f, s_score

    def _eval_pr(self, y_pred, y, num):
        if self.device:
            prec, recall = torch.zeros(num).to(self.device), torch.zeros(num).to(self.device)
            thlist = torch.linspace(0, 1 - 1e-10, num).to(self.device)
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _S_object(self, pred, mask):
        fg = torch.where(mask == 0, torch.zeros_like(pred), pred)
        bg = torch.where(mask == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, mask)
        o_bg = self._object(bg, 1 - mask)
        u = mask.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, mask):
        temp = pred[mask == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, mask):
        X, Y = self._centroid(mask)
        mask1, mask2, mask3, mask4, w1, w2, w3, w4 = self._divideGT(mask, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, mask1)
        Q2 = self._ssim(p2, mask2)
        Q3 = self._ssim(p3, mask3)
        Q4 = self._ssim(p4, mask4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, mask):
        rows, cols = mask.size()[-2:]
        mask = mask.view(rows, cols)
        if mask.sum() == 0:
            if self.device:
                X = torch.eye(1).to(self.device) * round(cols / 2)
                Y = torch.eye(1).to(self.device) * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = mask.sum()
            if self.device:
                i = torch.from_numpy(np.arange(0, cols)).to(self.device).float()
                j = torch.from_numpy(np.arange(0, rows)).to(self.device).float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((mask.sum(dim=0) * i).sum() / total)
            Y = torch.round((mask.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, mask, X, Y):
        h, w = mask.size()[-2:]
        area = h * w
        mask = mask.view(h, w)
        LT = mask[:Y, :X]
        RT = mask[:Y, X:w]
        LB = mask[Y:h, :X]
        RB = mask[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, mask):
        mask = mask.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = mask.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((mask - y) * (mask - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (mask - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q
