import math
import torch
import torch.nn.functional as F

def local_scan(x, w=2, H=8, W=8, flip=False, column_first=False):
    """
    Windowed scan (dynamic block size).
    Input:
        x: [B, L, C], where L == H*W
        w: block size
        H, W: original height and width before any padding
        column_first: 若为 True，则先按列再按行遍历块
        flip: 若为 True，则在最后反转序列
    Return:
        y: [B, C, H*W]  （恰好原始元素个数，不多也不少）
    """
    B, L, C = x.shape
    assert L == H * W, "输入长度必须等于 H*W"
    # 恢复空间形状
    x = x.view(B, H, W, C)

    # 决定遍历块的顺序
    ys = []
    # 先确定行列块索引序列
    row_idxs = list(range(0, H, w))
    col_idxs = list(range(0, W, w))
    if column_first:
        outer, inner = col_idxs, row_idxs
        swap = True
    else:
        outer, inner = row_idxs, col_idxs
        swap = False

    for i in outer:
        for j in inner:
            # 当前块的实际大小
            h_block = min(w, H - (j if swap else i))
            w_block = min(w, W - (i if swap else j))
            # 根据是否 column_first 选索引顺序
            if swap:
                block = x[:, j:j+h_block, i:i+w_block, :]
            else:
                block = x[:, i:i+h_block, j:j+w_block, :]
            # 扫描：B × (h_block*w_block) × C → B × C × (h_block*w_block)
            block = block.permute(0, 3, 1, 2).reshape(B, C, -1)
            ys.append(block)

    # 串联并可选反转
    y = torch.cat(ys, dim=2)  # [B, C, H*W]
    if flip:
        y = y.flip(-1)
    return y


def local_reverse(y, w=2, H=8, W=8, flip=False, column_first=False):
    """
    反扫描：将 local_scan 的输出还原回 [B, C, H*W]
    Input:
        y: [B, C, H*W]
        其它参数同 local_scan
    Return:
        x_rec: [B, C, H*W]
    """
    B, C, L = y.shape
    assert L == H * W, "输入长度必须等于 H*W"
    # 可选先反转
    if flip:
        y = y.flip(-1)

    # 逐块写回
    x_rec = torch.zeros(B, C, H, W, device=y.device, dtype=y.dtype)
    ptr = 0

    row_idxs = list(range(0, H, w))
    col_idxs = list(range(0, W, w))
    if column_first:
        outer, inner = col_idxs, row_idxs
        swap = True
    else:
        outer, inner = row_idxs, col_idxs
        swap = False

    for i in outer:
        for j in inner:
            h_block = min(w, H - (j if swap else i))
            w_block = min(w, W - (i if swap else j))
            size = h_block * w_block

            # 从 y 中切片并 reshape
            blk = y[:, :, ptr:ptr+size]            # [B, C, size]
            blk = blk.view(B, C, h_block, w_block)
            ptr += size

            # 写回到正确位置
            if swap:
                x_rec[:, :, j:j+h_block, i:i+w_block] = blk
            else:
                x_rec[:, :, i:i+h_block, j:j+w_block] = blk

    # 拉回 [B, C, H*W]
    return x_rec.reshape(B, C, -1)