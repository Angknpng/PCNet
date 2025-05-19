import os

# set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from model import TMSOD
from Code.utils.data import SalObjDataset, test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
import torch.nn as nn
from smooth_loss import get_saliency_smoothness
import argparse
import Code.utils.misc as utils
from Code.utils.default import _C as cfg
from torch.utils.data import DataLoader
import Code.utils.samplers as samplers
cudnn.benchmark = True
from torchvision import transforms
from PIL import Image

import random
from Code.lib.IHN.utils import *
import cv2
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

cfg = cfg
utils.init_distributed_mode(cfg.TRAIN)
device = torch.device(cfg.TRAIN.device)
seed = cfg.TRAIN.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# build the model
model = TMSOD()
if (opt.load is not None):
    model.load_pre(opt.load, opt.IHNmodel)
    print('load model from ', opt.load, opt.IHNmodel)

#check
for name, param in model.IHN.named_parameters():
    # if param.requires_grad:
    print(f"Model parameter {name} - mean: {param.data.mean()}, std: {param.data.std()}")
pretrained_weights = torch.load(opt.IHNmodel, map_location='cuda:0')
for name, param in pretrained_weights.items():
    print(f"Pretrained weight {name} - mean: {param.mean()}, std: {param.std()}")

model.to(device)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

params = model.parameters()
# optimizer = torch.optim.Adam(params, opt.lr)
optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=cfg.TRAIN.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.lr_drop)
# set the path
train_image_root = opt.rgb_label_root
train_gt_root = opt.gt_label_root
train_depth_root = opt.depth_label_root

val_image_root = opt.val_rgb_root
val_gt_root = opt.val_gt_root
val_depth_root = opt.val_depth_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
dataset_train = SalObjDataset(train_image_root, train_gt_root, train_depth_root,
                          trainsize=opt.trainsize)
dataset_test = test_dataset(val_image_root, val_gt_root, val_depth_root, opt.trainsize)

if cfg.TRAIN.distributed:
    sampler_train = samplers.DistributedSampler(dataset_train)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, opt.batchsize, drop_last=True)

train_loader = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               num_workers=16,
                               pin_memory=True)

total_step = len(train_loader)

if cfg.TRAIN.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.TRAIN.gpu], find_unused_parameters=True)



logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet_unif-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

CE = torch.nn.BCEWithLogitsLoss()

step = 0
best_mae = 1
best_epoch = 0

print(len(train_loader))


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
dice = SoftDiceLoss()


def draw_four_point_disp(image, four_point_disp):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    original_corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    four_point_disp_np = four_point_disp.cpu().detach().numpy()
    displaced_corners = original_corners + four_point_disp_np[0].reshape(4, 2)

    for point in original_corners:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 255, 0), -1)
    for point in displaced_corners:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    for i in range(4):
        start_point = tuple(original_corners[i].astype(int))
        end_point = tuple(displaced_corners[i].astype(int))
        cv2.line(image, start_point, end_point, (255, 0, 0), 2)

    return image


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    if cfg.TRAIN.distributed:
        sampler_train.set_epoch(epoch)

    try:
        for i, (images, gts, depths, img1, img2, depth_name) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.to(device)
            gts = gts.to(device)
            depths = depths.to(device)
            img2 = img2.to(device)
            img1 = img1.to(device)
            out = model(images, depths, img1, img2)
            sml = get_saliency_smoothness(torch.sigmoid(out), gts)
            loss1_fusion = F.binary_cross_entropy_with_logits(out, gts)
            dice_loss = dice(out, gts)
            loss_seg = loss1_fusion + sml + dice_loss
            loss = loss_seg
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 50 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f} sml: {:.4f} loss1_fusion: {:0.4f} dice: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, sml.data, loss1_fusion.data,
                           dice_loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f} sml: {:.4f} loss1_fusion: {:0.4f} dice: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, sml.data, loss1_fusion.data, dice_loss.data))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        # writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if (epoch) % 10 == 0 and epoch >= 40:
            torch.save(model.module.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.module.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post, img1, img2 = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            img1 = img1.cuda()
            img2 = img2.cuda()
            out = model(image, depth, img1, img2)
            res = out
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SPNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, save_path)
        if epoch >= 60 or epoch ==1:
            val(dataset_test, model, epoch, save_path)
