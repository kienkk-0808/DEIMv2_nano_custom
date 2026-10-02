"""
DEIMv2 Nano (HGNetv2-B0) — fine-tune từ pretrained COCO trên dataset định dạng YOLO.

Ví dụ:
    python train.py --data data/data.yaml

Dựa trên DEIMv2 (c) 2025 The DEIMv2 Authors, DEIM (c) 2024 The DEIM Authors,
RT-DETR (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import argparse
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

DEFAULT_CONFIG = os.path.join(ROOT, 'configs', 'deimv2_nano_custom.yml')
DEFAULT_PRETRAINED = os.path.join(ROOT, 'checkpoint', 'pretrained', 'deimv2_hgnetv2_n_coco.pth')

# Số epoch cuối tắt augmentation (mosaic/mixup/copyblend)
NO_AUG_EPOCHS = 10


def load_yolo_data(path: str) -> dict:
    """Đọc data.yaml kiểu ultralytics (path/train/val/names) → đường dẫn ảnh + tên lớp."""
    with open(path, encoding='utf-8') as f:
        d = yaml.safe_load(f)
    base = d.get('path') or os.path.dirname(os.path.abspath(path))
    if not os.path.isabs(base):
        base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), base))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    names = d['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return dict(train=resolve(d['train']), val=resolve(d.get('val') or d['test']), names=list(names))


def build_updates(args, data) -> dict:
    """Chuyển tham số CLI thành dict ghi đè lên config YAML."""
    E = args.epochs
    no_aug = min(NO_AUG_EPOCHS, max(1, E // 10))
    stop = E - no_aug          # epoch dừng augmentation mạnh
    half = max(5, E // 2 - 2)  # kết thúc giai đoạn mixup/copyblend
    warm = min(4, half - 1)

    # LR backbone = 1/2 LR chung (giữ tỉ lệ của cấu hình Nano gốc)
    lr, blr = args.lr, args.lr * 0.5
    upd = {
        'num_classes': len(data['names']),
        'epoches': E,
        'no_aug_epoch': no_aug,
        'output_dir': args.output_dir,
        'print_freq': args.print_freq,
        'checkpoint_freq': args.save_freq,
        'use_amp': not args.no_amp,
        'train_dataloader': {
            'total_batch_size': args.batch_size,
            'num_workers': args.workers,
            'dataset': {
                'img_folder': data['train'],
                'class_names': data['names'],
                'transforms': {'policy': {'epoch': [warm, half, stop]}},
            },
            'collate_fn': {
                'mixup_epochs': [warm, half],
                'stop_epoch': stop,
                'copyblend_epochs': [warm, half],
            },
        },
        'val_dataloader': {
            'total_batch_size': args.batch_size * 2,
            'num_workers': args.workers,
            'dataset': {'img_folder': data['val'], 'class_names': data['names']},
        },
        'optimizer': {
            'lr': lr,
            'params': [
                {'params': '^(?=.*backbone)(?!.*norm|bn).*$', 'lr': blr},
                {'params': '^(?=.*backbone)(?=.*norm|bn).*$', 'lr': blr, 'weight_decay': 0.},
                {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.},
            ],
        },
        'DEIMCriterion': {'matcher': {'matcher_change_epoch': max(half + 1, E - 2 * no_aug)}},
    }
    return upd


def main(args) -> None:
    if args.resume and args.pretrained:
        args.pretrained = None  # resume đã chứa đầy đủ trọng số

    if not os.path.isfile(args.data):
        sys.exit(f'[LỖI] Không thấy data.yaml: {args.data}')
    data = load_yolo_data(args.data)
    for key in ('train', 'val'):
        if not os.path.isdir(data[key]):
            sys.exit(f'[LỖI] Thư mục ảnh {key} không tồn tại: {data[key]}')
    print(f"Dataset: {len(data['names'])} lớp {data['names']}")
    if args.pretrained and not os.path.isfile(args.pretrained):
        sys.exit(f'[LỖI] Không thấy pretrained: {args.pretrained}')

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = build_updates(args, data)
    update_dict = yaml_utils.merge_dict(update_dict, yaml_utils.parse_cli(args.update))
    update_dict.update({
        'tuning': args.pretrained,
        'resume': args.resume,
        'device': args.device,
        'seed': args.seed,
        'test_only': args.test_only,
    })
    update_dict = {k: v for k, v in update_dict.items() if v is not None}

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.pretrained:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune DEIMv2 Nano từ pretrained COCO')

    # dataset (YOLO format)
    parser.add_argument('--data', type=str, required=True,
                        help='data.yaml kiểu YOLO (path, train, val, names)')

    # huấn luyện
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4, help='LR chung (backbone = lr/2)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-amp', action='store_true', help='tắt mixed precision')
    parser.add_argument('--output-dir', type=str, default=os.path.join('outputs', 'deimv2_nano_custom'))
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=10, help='lưu checkpoint mỗi N epoch')

    # checkpoint
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED,
                        help='checkpoint pretrained để fine-tune (head sai số lớp sẽ tự bỏ qua)')
    parser.add_argument('-r', '--resume', type=str, help='resume từ checkpoint của lần train trước')
    parser.add_argument('--test-only', action='store_true', help='chỉ đánh giá trên tập val')

    # khác
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('-d', '--device', type=str, help='vd: cuda:0 / cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-u', '--update', nargs='+', help='ghi đè YAML, vd: -u train_dataloader.shuffle=False')
    parser.add_argument('--print-method', type=str, default='builtin')
    parser.add_argument('--print-rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int)

    main(parser.parse_args())
