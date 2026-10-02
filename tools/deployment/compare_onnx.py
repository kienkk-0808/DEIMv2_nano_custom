"""
So sánh đầu ra PyTorch (checkpoint gốc) với ONNX đã export trên cùng 1 ảnh.

    python tools/deployment/compare_onnx.py -w best_stg1.pth -m best_stg1.onnx -i test.jpg --num-classes 10
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from engine.core import YAMLConfig


def main(args):
    cfg = YAMLConfig(args.config, num_classes=args.num_classes)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    ckpt = torch.load(args.weights, map_location='cpu')
    cfg.model.load_state_dict(ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model'])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, sizes):
            return self.postprocessor(self.model(images), sizes)

    model = Model().eval()
    size = tuple(cfg.yaml_cfg['eval_spatial_size'])

    im = Image.open(args.image).convert('RGB')
    w, h = im.size
    x = T.Compose([T.Resize(size), T.ToTensor()])(im).unsqueeze(0)
    orig = torch.tensor([[w, h]])

    with torch.no_grad():
        t_labels, t_boxes, t_scores = [o.numpy() for o in model(x, orig)]

    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    o_labels, o_boxes, o_scores = sess.run(
        None, {'images': x.numpy(), 'orig_target_sizes': orig.numpy()})

    print(f'Shape  torch: {t_labels.shape} {t_boxes.shape} {t_scores.shape}')
    print(f'Shape  onnx : {o_labels.shape} {o_boxes.shape} {o_scores.shape}')
    # Thứ tự query ở các box score thấp không ổn định → ghép từng box torch với box onnx gần nhất
    idx = np.where(t_scores[0] > args.min_score)[0]
    score_diff = box_diff = 0.0
    label_diff = 0
    for i in idx:
        dist = np.abs(o_boxes[0] - t_boxes[0][i]).max(axis=1) + (o_labels[0] != t_labels[0][i]) * 1e6
        j = int(dist.argmin())
        if dist[j] >= 1e6:  # không có box nào cùng lớp
            label_diff += 1
            continue
        box_diff = max(box_diff, float(dist[j]))
        score_diff = max(score_diff, abs(float(t_scores[0][i] - o_scores[0][j])))
    print(f'So sánh trên {len(idx)} box torch có score > {args.min_score} (ghép theo vị trí + lớp):')
    print(f'Max |diff| scores: {score_diff:.2e}')
    print(f'Max |diff| boxes : {box_diff:.2e} px')
    print(f'Box không tìm thấy bên ONNX: {label_diff} / {len(idx)}')

    def dets(l, b, s):
        k = s[0] > args.conf
        order = np.argsort(-s[0][k])
        return l[0][k][order], b[0][k][order], s[0][k][order]

    tl, tb, ts = dets(t_labels, t_boxes, t_scores)
    ol, ob, os_ = dets(o_labels, o_boxes, o_scores)
    print(f'\nDetection (conf>{args.conf})  torch: {len(tl)}  onnx: {len(ol)}')
    for i in range(max(len(tl), len(ol))):
        a = f'{tl[i]} {ts[i]:.3f} {np.round(tb[i]).astype(int).tolist()}' if i < len(tl) else '-'
        b = f'{ol[i]} {os_[i]:.3f} {np.round(ob[i]).astype(int).tolist()}' if i < len(ol) else '-'
        print(f'  torch: {a:<45} onnx: {b}')

    ok = score_diff < args.tol and box_diff < 0.5 and label_diff == 0 and len(tl) == len(ol)
    print('\nKẾT QUẢ:', 'ONNX KHỚP với PyTorch' if ok else 'CÓ LỆCH — cần kiểm tra lại')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-w', '--weights', required=True)
    p.add_argument('-m', '--onnx', required=True)
    p.add_argument('-i', '--image', required=True)
    p.add_argument('--num-classes', type=int, required=True)
    p.add_argument('--conf', type=float, default=0.4)
    p.add_argument('--tol', type=float, default=1e-3, help='sai số cho phép trên score')
    p.add_argument('--min-score', type=float, default=0.4, help='chỉ so box có score trên ngưỡng này (box score thấp có nhiều bản sao gần giống nhau nên so không ổn định)')
    p.add_argument('-c', '--config', default='configs/deimv2_nano_custom.yml')
    main(p.parse_args())
