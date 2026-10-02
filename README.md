# DEIMv2 Nano — Fine-tune dataset YOLO

Bản rút gọn của [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), chỉ giữ **DEIMv2 Nano (HGNetv2-B0)**,
fine-tune từ pretrained COCO `checkpoint/pretrained/deimv2_hgnetv2_n_coco.pth` trên dataset **định dạng YOLO**.

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu: Python 3.9+, PyTorch 2.x (GPU CUDA khuyến nghị; CPU chỉ đủ để chạy thử).

## 2. Cấu trúc thư mục

```
├── train.py                          # script train/eval chính
├── configs/
│   ├── deimv2_nano_custom.yml        # config Nano dùng để fine-tune
│   ├── deimv2/deimv2_hgnetv2_n_coco.yml  # config gốc Nano trên COCO (tham chiếu)
│   ├── base/ dataset/ runtime.yml    # config nền
├── checkpoint/pretrained/deimv2_hgnetv2_n_coco.pth
├── engine/                           # mã nguồn model/solver/dataset
└── tools/                            # inference, deployment (ONNX/TensorRT), benchmark
```

## 3. Chuẩn bị dataset (YOLO)

```
mydata/
├── data.yaml
├── images/{train,val}/*.jpg
└── labels/{train,val}/*.txt          # cùng tên với ảnh
```

Mỗi dòng label: `class_id cx cy w h` (chuẩn hóa 0–1, `class_id` bắt đầu từ 0).
Dòng polygon (segmentation) sẽ được chuyển thành bbox bao. Ảnh không có file label được coi là ảnh nền.

`data.yaml` (kiểu ultralytics):

```yaml
path: .            # gốc, tương đối so với vị trí data.yaml
train: images/train
val: images/val
names:
  0: cat
  1: dog
```

Thư mục label được suy ra bằng cách thay `images` → `labels` trong đường dẫn ảnh. Số lớp lấy tự động từ `names`.

## 4. Train

```bash
python train.py --data mydata/data.yaml
```

Mặc định: nạp pretrained COCO (head khác số lớp tự bỏ qua), 100 epoch, batch 32, lr 2e-4, AMP bật.

Tuỳ chọn thường dùng:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--data` | (bắt buộc) | `data.yaml` kiểu YOLO |
| `--epochs` | 100 | số epoch (lịch augmentation tự co giãn theo) |
| `--batch-size` | 32 | giảm nếu thiếu VRAM |
| `--lr` | 2e-4 | LR chung, backbone = lr/2 |
| `--workers` | 4 | số worker dataloader |
| `--pretrained` | checkpoint COCO Nano | đổi checkpoint khởi tạo |
| `--output-dir` | `outputs/deimv2_nano_custom` | nơi lưu kết quả |
| `--save-freq` | 10 | lưu checkpoint mỗi N epoch |
| `--no-amp` | tắt | tắt mixed precision |
| `-d` | tự chọn | thiết bị, vd `cuda:0`, `cpu` |
| `-u key=value` | | ghi đè bất kỳ giá trị YAML, vd `-u train_dataloader.shuffle=False` |

Ví dụ:

```bash
python train.py --data mydata/data.yaml --epochs 150 --batch-size 16 --lr 1e-4 -d cuda:0
```

Kết quả trong `--output-dir`: `best_stg1.pth` (tốt nhất), `last.pth`, `log.txt`, `summary/` (TensorBoard).

### Resume / đánh giá

```bash
# Tiếp tục train bị gián đoạn
python train.py --data mydata/data.yaml -r outputs/deimv2_nano_custom/last.pth

# Chỉ đánh giá (COCO mAP) trên tập val
python train.py --data mydata/data.yaml -r outputs/deimv2_nano_custom/best_stg1.pth --test-only
```

> Khi dùng `-r` hoặc `--test-only`, hãy truyền cùng `--data` / `--epochs` với lúc train.

## 5. Inference & export

Trước khi chạy, đặt `num_classes` trong `configs/dataset/custom_detection.yml` bằng số lớp của bạn.

```bash
python tools/inference/torch_inf.py -c configs/deimv2_nano_custom.yml \
    -r outputs/deimv2_nano_custom/best_stg1.pth -i image.jpg -d cuda:0

python tools/deployment/export_onnx.py -c configs/deimv2_nano_custom.yml \
    -r outputs/deimv2_nano_custom/best_stg1.pth --check --simplify
```

## 6. Ghi chú

- Ảnh được resize về 640×640; nếu dataset có vật thể nhỏ, tăng `eval_spatial_size` và `Resize` trong config.
- Dataset nhỏ (< vài trăm ảnh): giảm `--lr` (1e-4) và/hoặc `--epochs`.
- Nguồn: DEIMv2 (Apache-2.0), DEIM, RT-DETR.
