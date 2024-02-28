import os
import sys
import argparse
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
import torch
import torch.utils.data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

from datasets.pascal import PascalVOC_eval
from datasets.coco import COCO_eval
from Visualize_Attn.hg_attn_light import exkp
from utils.utils import load_model
from utils.image import transform_preds
from utils.summary import create_logger
from utils.post_process import ctdet_decode


# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')

parser.add_argument('--dataset', type=str, default='pascal', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='small_hg_attn_light')

parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=1)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, '/pretrain_path.7z')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]


def plot_attention(img, attention, img_size, img_name):
    save_dir = '/save_hmaps_path'
    attention = nn.functional.interpolate(attention.unsqueeze(0),
                                          scale_factor=img_size // attention.shape[-1],
                                          mode="nearest").cpu().numpy()
    # print('attention size=', attention.shape)
    n_heads = attention.shape[1]

    # plt.figure()
    # fig = np.asarray(np.mean(attention.copy(), 1))
    # fig = fig.transpose((1, 2, 0))
    # plt.imshow(fig, cmap='inferno')
    # plt.savefig(os.path.join(save_dir, 'Head_Mean_%s' % str(img_name).replace('bmp', 'jpg')))

    # plt.figure(figsize=(10, 10))
    # text = ["Original_Image", "Head_Mean"]
    # for i, fig in enumerate([img, np.mean(attention, 1)]):
    #     # print('fig shape=', fig.shape)
    #     # plt.subplot(1, 2, i + 1)
    #     fig = np.asarray(fig.copy())
    #     fig = fig.transpose((1, 2, 0))
    #     plt.imshow(fig, cmap='inferno')
    #     # plt.title(text[i])
    #     plt.savefig(os.path.join(save_dir, '%s_%s' % (img_name, text[i])))
    #     # plt.show()

    plt.figure(figsize=(10, 10))
    attention=attention.squeeze(0)
    for i in range(n_heads):
        plt.subplot(n_heads // 4, 4, i + 1)
        attn = np.asarray(attention[i])
        print(attn.shape)
        plt.imshow(attn.transpose((1, 0)), cmap='inferno')
        plt.title(f"Head n: {i + 1}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'Hmaps_%s' % str(img_name).replace('bmp', 'jpg')))
    print('attn_maps of all heads have been saved')


def main():
    logger = create_logger(save_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    max_per_image = 100

    Dataset_eval = COCO_eval if cfg.dataset == 'coco' else PascalVOC_eval
    dataset = Dataset_eval(cfg.data_dir, split='val', img_size=cfg.img_size,
                           test_scales=cfg.test_scales, test_flip=cfg.test_flip, fix_size=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=True,
                                              collate_fn=dataset.collate_fn)

    print('Creating model...')
    if 'hourglass' in cfg.arch:
        model = exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4],
                     vit_hmap=True)
    else:
        raise NotImplementedError

    model = load_model(model, cfg.pretrain_dir)
    model = model.to(cfg.device)
    model.eval()

    results = {}
    with torch.no_grad():
        for inputs in tqdm(data_loader):
            img_id, inputs = inputs[0]

            detections = []
            for scale in inputs:
                inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)

                output = model(inputs[scale]['image'])[-1]

                plot_attention(inputs[scale]['image'].squeeze(0).cpu().numpy(),
                               model.attn2hmap_org_2,
                               cfg.img_size,
                               inputs[scale]['img_name']
                               )

                dets = ctdet_decode(*output, K=cfg.test_topk)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                top_preds = {}
                dets[:, :2] = transform_preds(dets[:, 0:2],
                                              inputs[scale]['center'],
                                              inputs[scale]['scale'],
                                              (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                               inputs[scale]['center'],
                                               inputs[scale]['scale'],
                                               (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                cls = dets[:, -1]
                for j in range(dataset.num_classes):
                    inds = (cls == j)
                    top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                    top_preds[j + 1][:, :4] /= scale

                detections.append(top_preds)

            bbox_and_scores = {}
            for j in range(1, dataset.num_classes + 1):
                bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
                # if len(dataset.test_scales) > 1:
                # soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
            scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, dataset.num_classes + 1)])

            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, dataset.num_classes + 1):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            results[img_id] = bbox_and_scores

    eval_results = dataset.run_eval(results, cfg.ckpt_dir)
    print(eval_results)


if __name__ == '__main__':
    main()
