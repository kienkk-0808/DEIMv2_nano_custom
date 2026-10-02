"""
Dataset định dạng YOLO (ultralytics): ảnh + file .txt cùng tên, mỗi dòng `cls cx cy w h` (chuẩn hóa 0-1).
Nội bộ dựng sẵn một COCO API trong bộ nhớ nên dùng chung CocoEvaluator / pipeline với CocoDetection.
"""

import os

import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO

from .coco_dataset import CocoDetection
from ...core import register

__all__ = ['YoloDetection']

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def default_label_dir(img_folder: str) -> str:
    """Quy ước ultralytics: thay thư mục `images` bằng `labels`."""
    norm = os.path.normpath(img_folder)
    parts = norm.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'
            return os.sep.join(parts)
    return norm  # label nằm cạnh ảnh


@register()
class YoloDetection(CocoDetection):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category', 'num_classes']

    def __init__(self, img_folder, transforms, label_folder=None, class_names=None,
                 num_classes=None, return_masks=False, remap_mscoco_category=False):
        # Không gọi CocoDetection.__init__ (nó đọc file json) — tự dựng COCO API.
        torchvision.datasets.VisionDataset.__init__(self, img_folder)
        from .coco_dataset import ConvertCocoPolysToMask
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.label_folder = label_folder or default_label_dir(img_folder)
        self.ann_file = self.label_folder
        self.return_masks = return_masks
        self.remap_mscoco_category = False

        if class_names is None:
            class_names = [str(i) for i in range(num_classes or 0)]
        self.coco = self._build_coco(list(class_names))
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _build_coco(self, class_names):
        files = []
        for dp, _, fns in os.walk(self.img_folder):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    files.append(os.path.relpath(os.path.join(dp, fn), self.img_folder))
        files.sort()
        if not files:
            raise FileNotFoundError(f'Không có ảnh nào trong {self.img_folder}')

        images, annotations = [], []
        ann_id = 1
        for img_id, rel in enumerate(files, start=1):
            with Image.open(os.path.join(self.img_folder, rel)) as im:
                w, h = im.size
            images.append(dict(id=img_id, file_name=rel, width=w, height=h))

            label_path = os.path.join(self.label_folder, os.path.splitext(rel)[0] + '.txt')
            if not os.path.isfile(label_path):
                continue  # ảnh nền (không có object)
            with open(label_path, encoding='utf-8') as f:
                for line in f:
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cls = int(float(p[0]))
                    v = [float(x) for x in p[1:]]
                    if len(v) == 4:
                        cx, cy, bw, bh = v
                    else:  # dòng polygon (segmentation) → lấy bbox bao
                        xs, ys = v[0::2], v[1::2]
                        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                    x1 = max(0.0, (cx - bw / 2) * w)
                    y1 = max(0.0, (cy - bh / 2) * h)
                    x2 = min(float(w), (cx + bw / 2) * w)
                    y2 = min(float(h), (cy + bh / 2) * h)
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue
                    if cls < 0 or cls >= len(class_names):
                        raise ValueError(f'class id {cls} ngoài khoảng [0, {len(class_names) - 1}] trong {label_path}')
                    annotations.append(dict(id=ann_id, image_id=img_id, category_id=cls,
                                            bbox=[x1, y1, x2 - x1, y2 - y1],
                                            area=(x2 - x1) * (y2 - y1), iscrowd=0))
                    ann_id += 1

        coco = COCO()
        coco.dataset = dict(images=images, annotations=annotations,
                            categories=[dict(id=i, name=n) for i, n in enumerate(class_names)])
        coco.createIndex()
        return coco

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n label_folder: {self.label_folder}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s
