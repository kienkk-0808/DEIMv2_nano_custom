"""
Chạy thử DEIMv2 Nano trên 1 ảnh, từ checkpoint (.pth) hoặc file ONNX đã export.

Ví dụ:
    python infer.py --weights outputs/deimv2_nano_custom/best_stg1.pth --data data/data.yaml --image test.jpg
    python infer.py --onnx best_stg1.onnx --data data/data.yaml --image test.jpg
"""

import os
import sys
import time
import argparse

import numpy as np
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


def load_onnx(path, device):
    import onnxruntime as ort
    providers = ['CPUExecutionProvider']
    if device.type == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    return ort.InferenceSession(path, providers=providers)


def detect_onnx(sess, image, conf):
    h_in, w_in = sess.get_inputs()[0].shape[2:]  # kích thước ảnh cố định khi export
    w, h = image.size
    x = np.asarray(image.resize((w_in, h_in), Image.BILINEAR), dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    labels, boxes, scores = sess.run(None, {'images': x, 'orig_target_sizes': np.array([[w, h]], dtype=np.int64)})
    keep = scores[0] > conf
    boxes = torch.from_numpy(boxes[0][keep]).clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return torch.from_numpy(labels[0][keep]), boxes, torch.from_numpy(scores[0][keep])


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
    image = Image.open(args.image).convert('RGB')

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    if args.onnx:
        sess = load_onnx(args.onnx, device)
        run = lambda: detect_onnx(sess, image, args.conf)
    else:
        model, size = load_model(args, num_classes, device)
        run = lambda: detect(model, device, image, size, args.conf)
    print(f'Nạp model     : {(time.perf_counter() - t0) * 1000:8.1f} ms  ({"ONNX" if args.onnx else "PyTorch"}, {device})')

    # Warm-up: các lần chạy đầu chậm (khởi tạo kernel, cấp phát bộ nhớ) nên chạy bỏ qua trước khi đo
    t0 = time.perf_counter()
    for _ in range(args.warmup):
        run()
    sync()
    print(f'Warm-up       : {(time.perf_counter() - t0) * 1000:8.1f} ms  ({args.warmup} lần)')

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        labels, boxes, scores = run()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    print(f'Suy luận      : {np.mean(times):8.1f} ms  (trung bình {args.runs} lần, min {min(times):.1f}, max {max(times):.1f}; '
          f'~{1000 / np.mean(times):.1f} FPS, gồm tiền xử lý + hậu xử lý)')

    print(f'Phát hiện {len(labels)} vật thể (conf > {args.conf}):')
    for lb, bx, sc in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        print(f'  {names[lb]:<14} {sc:.2f}  box=[{", ".join(f"{v:.0f}" for v in bx)}]')

    out = args.output or os.path.splitext(args.image)[0] + '_pred.jpg'
    draw(image, labels, boxes, scores, names).save(out)
    print(f'Đã lưu: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMv2 Nano — chạy thử 1 ảnh')
    parser.add_argument('-w', '--weights', help='checkpoint (best_stg1.pth / last.pth)')
    parser.add_argument('--onnx', help='file ONNX đã export — dùng thay cho --weights')
    parser.add_argument('-i', '--image', required=True, help='đường dẫn ảnh')
    parser.add_argument('--data', help='data.yaml kiểu YOLO (lấy tên lớp + số lớp)')
    parser.add_argument('--num-classes', type=int, help='dùng khi không có --data')
    parser.add_argument('--warmup', type=int, default=3, help='số lần chạy khởi động model trước khi đo')
    parser.add_argument('--runs', type=int, default=10, help='số lần chạy để đo thời gian trung bình')
    parser.add_argument('--conf', type=float, default=0.4, help='ngưỡng confidence')
    parser.add_argument('-o', '--output', help='ảnh kết quả (mặc định: <ảnh>_pred.jpg)')
    parser.add_argument('-d', '--device', help='cuda:0 / cpu (mặc định tự chọn)')
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not (args.weights or args.onnx):
        parser.error('cần --weights hoặc --onnx')
    main(args)
