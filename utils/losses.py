import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss(preds, targets, overthr=1):
    '''
    Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      preds (B x c x h x w)
      gt_regr (B x c x h x w)
    '''
    pos_inds = targets.ge(overthr).float()
    pos_weights = torch.pow(targets, 4)
    neg_inds = targets.lt(overthr).float()
    neg_weights = torch.pow(1 - targets, 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_weights * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def _reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum')
               / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


def get_boxes(regs, w_h_):
    batch, cat, height, width = regs.size()
    xs = torch.arange(0, width)
    ys = torch.arange(0, height)
    xs = regs[:, :, :, 0:1] + xs
    ys = regs[:, :, :, 1:2] + ys
    # _w = w_h_[:, :, :, 0]
    # _h = w_h_[:, :, :, 1]
    bboxes = torch.cat([xs - w_h_[:, :, :, 0:1] / 2,
                        ys - w_h_[:, :, :, 1:2] / 2,
                        xs + w_h_[:, :, :, 0:1] / 2,
                        ys + w_h_[:, :, :, 1:2] / 2], dim=2)
    return bboxes
