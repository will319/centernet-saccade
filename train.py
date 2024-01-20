import os
import sys
import time
import argparse
import torch.nn.functional
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from datasets.coco import COCO, COCO_eval
from datasets.pascal import PascalVOC, PascalVOC_eval

from nets.hg import get_hg_base
from nets.hg_attn import get_hg_attn
from nets.hg_light import get_hg_light
from nets.hg_attn_light import get_hg_attn_light
from nets.resnet import get_pose_net as get_resnet

from utils.utils import _tranpose_and_gather_feature, load_model
from utils.image import transform_preds
from utils.losses import _neg_loss, _reg_loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctdet_decode
import warnings

warnings.filterwarnings("ignore")
# Training settings
parser = argparse.ArgumentParser(description='simple_centernet_saccade')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')
parser.add_argument('--resume', type=str, default=None)

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='log')
parser.add_argument('--pretrain_name', type=str, default='pretrain')

parser.add_argument('--dataset', type=str, default='pascal', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='small'+'hg_attn_light')    # or 'hg_base' or 'hg_attn' or 'hg_light'

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_step', type=str, default='80, 110')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=150)

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.root_dir, cfg.pretrain_name, './.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    seed = 317
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # disable this if OOM at beginning of training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    num_gpus = torch.cuda.device_count()

    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=num_gpus, rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')

    print('Setting up data...')
    Dataset = COCO if cfg.dataset == 'coco' else PascalVOC
    train_dataset = Dataset(cfg.data_dir, 'train', split_ratio=cfg.split_ratio, img_size=cfg.img_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=num_gpus,
                                                                    rank=cfg.local_rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size // num_gpus
                                               if cfg.dist else cfg.batch_size,
                                               shuffle=not cfg.dist,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               sampler=train_sampler if cfg.dist else None)

    Dataset_eval = COCO_eval if cfg.dataset == 'coco' else PascalVOC_eval

    val_dataset = Dataset_eval(cfg.data_dir, 'val',
                               test_scales=[1.],
                               test_flip=False,
                               img_size=cfg.img_size,
                               fix_size=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             shuffle=False, num_workers=1, pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    # NEW for mix computing
    # scaler = GradScaler()
    print('Creating model...')
    if 'hg_base' in cfg.arch:
        model = get_hg_base[cfg.arch]
    elif 'hg_attn' in cfg.arch:
        model = get_hg_attn[cfg.arch]
    elif 'hg_light' in cfg.arch:
        model = get_hg_light[cfg.arch]
    elif 'hg_attn_light' in cfg.arch:
        model = get_hg_attn_light[cfg.arch]
    elif 'resnet' in cfg.arch:
        model = get_resnet(num_layers=int(cfg.arch.split('_')[-1]),
                           num_classes=train_dataset.num_classes)
    else:
        raise NotImplementedError

    if os.path.isfile(cfg.pretrain_dir):
        model = load_model(model, cfg.pretrain_dir)

    if cfg.dist:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cfg.device)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[cfg.local_rank, ],
                                                    output_device=cfg.local_rank)
    else:
        model = nn.DataParallel(model).to(cfg.device)
        # model = model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

    def train(epoch: int) -> None:
        print('\n Epoch: %d' % epoch)
        model.train()
        tic = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            # mask=batch['w_h_']!=0
            # ps+=np.average(batch['w_h_'][mask])
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

            # with autocast():    # auto-chose which computer can use mix_precision
            outputs = model(batch['image'])
            hmap, regs, w_h_ = zip(*outputs)
            regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
            w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

            hmap_loss = _neg_loss(hmap, batch['hmap'], overthr=0.85)
            reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
            w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])

            # grad accumulation
            loss = (hmap_loss + 1 * reg_loss + 1 * w_h_loss) / cfg.accumulation_steps
            loss.backward()
            if ((batch_idx + 1) % cfg.accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            # scaler.scale(loss).backward()   # New
            # scaler.step(optimizer)          # New
            # scaler.update()                 # New

            if batch_idx % cfg.log_interval == 0:
                # print(ps/(batch_idx+1))
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                      ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
                      (hmap_loss.item(), reg_loss.item(), w_h_loss.item()) +
                      ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

                step = len(train_loader) * epoch + batch_idx
                summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
                summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
                summary_writer.add_scalar('w_h_loss', w_h_loss.item(), step)

    def val_map(epoch):
        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()
        max_per_image = 100

        results = {}
        with torch.no_grad():
            for inputs in val_loader:
                img_id, inputs = inputs[0]
                detections = []
                for scale in inputs:
                    inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
                    output = model(inputs[scale]['image'])[-1]

                    dets = ctdet_decode(*output, K=cfg.test_topk)
                    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                    top_preds = {}
                    dets[:, :2] = transform_preds(dets[:, 0:2],
                                                  inputs[scale]['center'], inputs[scale]['scale'],
                                                  (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                                   inputs[scale]['center'], inputs[scale]['scale'],
                                                   (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    clses = dets[:, -1]
                    for j in range(val_dataset.num_classes):
                        inds = (clses == j)
                        top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                        top_preds[j + 1][:, :4] /= scale

                    detections.append(top_preds)

                bbox_and_scores = {j: np.concatenate([d[j] for d in detections], axis=0)
                                   for j in range(1, val_dataset.num_classes + 1)}
                scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, val_dataset.num_classes + 1)])   # np.hstack():在水平方向上平铺
                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, val_dataset.num_classes + 1):
                        keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                        bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

                results[img_id] = bbox_and_scores

        eval_results = val_dataset.run_eval(results)
        print(eval_results)
        summary_writer.add_scalar('val_mAP/mAP', eval_results[0], epoch)
        return eval_results

    print('Starting training...')
    best_score = 0
    stop_ = 0

    for epoch in range(1, cfg.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            eval_results = val_map(epoch)

            if cfg.dataset == 'pascal':
                if eval_results[0] > best_score:
                    best_score = eval_results[0]
                    best_epoch = epoch
                    print("Now the best score:%f.4 best epoch: %d" % (best_score, best_epoch))
                    stop_ = 0

                else:
                    if epoch > cfg.lr_step[-1]:
                        stop_ += 1
                        print("No the best score \n best score:%f.4" % best_score)
                        if stop_ > 10:
                            print("The accuracy was not improved for 5 consecutive rounds, exit the training")
                            break
                    if str(epoch) in cfg.lr_step:
                        print(saver.save(model.module.state_dict(), cfg.arch + "_epoch" + str(epoch)))
            elif cfg.dataset == 'coco':
                pass
            print(saver.save(model.module.state_dict(), cfg.arch + "_epoch" + str(epoch)))
        # lr_scheduler.step(epoch)  # move to here after pytorch1.1.0
    summary_writer.close()


if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
