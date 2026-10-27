# CIFAR-100: Custom ResNet + MixUp và pretrained Swin-T

Code chạy chính nằm trong `src/cifar100/`; các notebook giữ nguyên để thử nghiệm, giải thích và lưu kết quả gốc. Notebook không chỉ là nháp: có thể dùng làm tài liệu thực nghiệm, nhưng không cần chạy tuần tự các cell để sử dụng pipeline Python này.

## Cấu trúc

```text
scripts/
  train.py                    # Huấn luyện theo JSON config
  evaluate.py                 # Đánh giá checkpoint, xuất metrics và hình
  predict.py                  # Dự đoán một ảnh
  inspect_notebooks.py        # Kiểm tra cả training/ và analysis/
src/cifar100/
  models/{resnet,swin}.py      # Kiến trúc và optimizer Swin
  config.py, data.py          # Cấu hình, preprocessing và split
  engine.py, pipeline.py      # Train / evaluate và điều phối
  checkpoint.py, reporting.py # Checkpoint và xuất kết quả
  utils.py, cli.py            # Tiện ích và tham số dòng lệnh
notebooks/
  training/
    cifar100-cnnbuild_mixup.ipynb
    swin transformer.ipynb
  analysis/
    cnnrebuild-feature-visualization.ipynb
    swin_t_feature.ipynb
configs/                      # Cấu hình của scripts/train.py
data/{raw,processed,splits}/
checkpoints/{resnet_mixup,swin_t}/
results/
  provenance.json             # Nguồn và hash các notebook
  resnet_mixup/                # Kết quả train cũ, các run Python
    feature_analysis/         # Kết quả feature CNN tải từ Kaggle
  swin_t/
    feature_analysis/         # Kết quả feature Swin tải từ Kaggle
reports/{slides,latex}/
tests/test_pipeline.py
docs/
requirements.txt
```

## Cài đặt

Dùng Python 3.10+ và môi trường riêng:

```bash
cd /home/shin-bkb/code/deep_learning_miniprojrect
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch cần bản tương thích GPU/CUDA của máy nếu muốn train GPU. Đã kiểm tra offline trên CPU với PyTorch 2.3.1 và torchvision 0.18.1; requirements chưa phải lockfile môi trường. Không cần GPU để chạy `--help`.

## Huấn luyện

```bash
python scripts/train.py --config configs/resnet_mixup.json --download --run-name resnet_run01
python scripts/train.py --config configs/swin_t.json --download --run-name swin_run01
```

`--device auto` mặc định chọn CUDA nếu có, nếu không dùng CPU. Có thể chỉ định `--device cuda:0` hoặc `--device cpu`. CIFAR-100 tải vào `data/raw/`; bỏ `--download` khi đã có dữ liệu. Lần đầu train Swin-T cần Internet để tải trọng số ImageNet nếu chưa được cache, độc lập với cờ tải dataset. Train đầy đủ trên CPU có thể rất lâu.

- ResNet: 200 epoch, SGD + cosine, MixUp alpha 0.2, label smoothing 0.1, ảnh 32×32.
- Swin-T: train head 3 epoch, nạp head checkpoint tốt nhất rồi fine-tune toàn bộ 8 epoch, ảnh 224×224.
- Hai model dùng chung stratified split 45.000 train / 5.000 validation, seed 42, được lưu và kiểm tra trong `data/splits/`.
- Thay tham số trực tiếp trong JSON trước khi chạy; khi đổi seed hoặc cỡ validation hãy đổi `split_path` để tạo split mới.
- Checkpoint tốt nhất được chọn theo validation accuracy; đồng thời lưu checkpoint có validation loss thấp nhất cho mỗi giai đoạn.
- Sau train, tự đo clean train accuracy từ checkpoint tốt nhất. Không dùng test set để chọn model.

Mỗi run có config, history CSV, learning curves, log và training summary riêng. Tên run đã tồn tại sẽ bị từ chối để tránh ghi đè. Checkpoint phục vụ đánh giá/dự đoán, chưa hỗ trợ tiếp tục huấn luyện từ optimizer state.

## Đánh giá và xuất hình

Sau khi chốt cấu hình, đánh giá test:

```bash
python scripts/evaluate.py --checkpoint checkpoints/resnet_mixup/resnet_run01/train_best_accuracy.pt --split test
python scripts/evaluate.py --checkpoint checkpoints/swin_t/swin_run01/finetune_best_accuracy.pt --split test
```

Có thể dùng `--split val` (mặc định) hoặc `--split train` để đánh giá dữ liệu sạch; `--data-dir`, `--batch-size`, `--num-workers` cho phép đổi môi trường đánh giá. Kết quả nằm trong `results/<model>/<run>/evaluations/<split_timestamp>/`:

- Accuracy, macro F1, loss, báo cáo precision/recall/F1 từng lớp.
- Confusion matrix dạng số đếm và chuẩn hóa: CSV và PNG.
- Biểu đồ các cặp lớp nhầm nhiều nhất, ảnh dự đoán mẫu và predictions CSV.

Không so sánh trực tiếp train accuracy có MixUp với validation accuracy. Dùng clean train accuracy để tính khoảng cách train–validation. Loss giữa hai model cũng khác do ResNet dùng label smoothing. Swin có lợi thế pretraining và độ phân giải; xem thêm `docs/experiment_protocol.md`.

## Dự đoán ảnh

```bash
python scripts/predict.py --checkpoint checkpoints/resnet_mixup/resnet_run01/train_best_accuracy.pt --image /duong/dan/anh.jpg --top-k 5
```

Kết quả JSON gồm tên lớp và softmax probability; đây không phải bảo đảm độ tin cậy trên ảnh ngoài phân phối CIFAR-100.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Tests dùng dữ liệu tổng hợp, không tải dataset/trọng số và không tái tạo các điểm số báo cáo. Pipeline `.py` chưa được train lại đầy đủ; kết quả lịch sử trong notebook và `results/<model>/metrics/` vẫn là kết quả notebook gốc. Checkpoint notebook `.pth` có định dạng khác, không nạp trực tiếp bằng các CLI mới. Xem `docs/paths.md`.

## Notebook phân tích đặc trưng

- `notebooks/training/`: hai notebook huấn luyện gốc và kết quả thực nghiệm.
- `notebooks/analysis/`: hai notebook nạp checkpoint đã train, trích feature và vẽ PCA/t-SNE; không train lại.
- Upload notebook analysis và checkpoint tương ứng lên Kaggle, thêm dataset CIFAR-100 rồi chạy các cell. Các đường dẫn trong notebook vẫn dành cho Kaggle.
- Tải ZIP kết quả feature từ Kaggle, đưa **nội dung bên trong thư mục `feature_analysis`** vào `results/resnet_mixup/feature_analysis/` hoặc `results/swin_t/feature_analysis/`. Hiện đây là thư mục đích, chưa có dữ liệu feature được tải về trong lần sắp xếp này.
- Dùng cùng test indices và nhãn khi so sánh hai model. `sample_test_indices.npy` và `feature_sample_manifest.csv` giúp kiểm tra điều đó.

Kiểm tra cú pháp cả bốn notebook mà không chạy huấn luyện:

```bash
python scripts/inspect_notebooks.py
```
