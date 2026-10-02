"""
Chạy thử DEIMv2 Nano trên 1 ảnh từ checkpoint đã train.

Ví dụ:
    python infer.py --weights outputs/deimv2_nano_custom/best_stg1.pth \
                    --data data/data.yaml --image test.jpg
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.core import YAMLConfig
from train import load_yolo_data, DEFAULT_CONFIG

PALETTE = ['#F05922', '#251C53', '#2E9E5B', '#1F77B4', '#9467BD',
           '#D4A017', '#17BECF', '#E377C2', '#8C564B', '#7F7F7F']


def load_model(args, num_classes, device):
    cfg = YAMLConfig(args.config, num_classes=num_classes)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    ckpt = torch.load(args.weights, map_location='cpu')
    state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    return Model().to(device).eval(), tuple(cfg.yaml_cfg['eval_spatial_size'])


@torch.no_grad()
def detect(model, device, image, size, conf):
    w, h = image.size
    x = T.Compose([T.Resize(size), T.ToTensor()])(image).unsqueeze(0).to(device)
    labels, boxes, scores = model(x, torch.tensor([[w, h]], device=device))
    keep = scores[0] > conf
    boxes = boxes[0][keep].cpu()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)  # kẹp box vào trong ảnh
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return labels[0][keep].cpu(), boxes, scores[0][keep].cpu()


def draw(image, labels, boxes, scores, names):
    d = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    drawn = []  # box đã vẽ, để xử lý các box trùng nhau (nhiều lớp cùng 1 vị trí)
    for lb, bx, sc in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        color = PALETTE[lb % len(PALETTE)]
        name = names[lb] if lb < len(names) else str(lb)
        text = f'{name} {sc:.2f}'
        k = sum(1 for o in drawn if max(abs(a - b) for a, b in zip(o, bx)) < 8)
        drawn.append(bx)
        x1, y1, x2, y2 = [v + (k * 4 if i < 2 else -k * 4) for i, v in enumerate(bx)]  # thụt vào để không bị che
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        ty = y1 - 12 if y1 >= 12 else y1
        tw = d.textlength(text, font=font)
        d.rectangle([x1, ty, x1 + tw + 4, ty + 12], fill=color)
        d.text((x1 + 2, ty), text, fill='white', font=font)
    return image


def main(args):
    names = None
    if args.data:
        names = load_yolo_data(args.data)['names']
    num_classes = len(names) if names else args.num_classes
    if not num_classes:
        sys.exit('[LỖI] Cần --data (data.yaml) hoặc --num-classes')
    names = names or [str(i) for i in range(num_classes)]

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, size = load_model(args, num_classes, device)

    image = Image.open(args.image).convert('RGB')
    labels, boxes, scores = detect(model, device, image, size, args.conf)

    print(f'Phát hiện {len(labels)} vật thể (conf > {args.conf}):')
    for lb, bx, sc in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        print(f'  {names[lb]:<14} {sc:.2f}  box=[{", ".join(f"{v:.0f}" for v in bx)}]')

    out = args.output or os.path.splitext(args.image)[0] + '_pred.jpg'
    draw(image, labels, boxes, scores, names).save(out)
    print(f'Đã lưu: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMv2 Nano — chạy thử 1 ảnh')
    parser.add_argument('-w', '--weights', required=True, help='checkpoint (best_stg1.pth / last.pth)')
    parser.add_argument('-i', '--image', required=True, help='đường dẫn ảnh')
    parser.add_argument('--data', help='data.yaml kiểu YOLO (lấy tên lớp + số lớp)')
    parser.add_argument('--num-classes', type=int, help='dùng khi không có --data')
    parser.add_argument('--conf', type=float, default=0.4, help='ngưỡng confidence')
    parser.add_argument('-o', '--output', help='ảnh kết quả (mặc định: <ảnh>_pred.jpg)')
    parser.add_argument('-d', '--device', help='cuda:0 / cpu (mặc định tự chọn)')
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG)
    main(parser.parse_args())
