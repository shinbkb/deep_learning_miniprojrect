# Đường dẫn và dữ liệu

Các đường dẫn tương đối trong config và CLI được tính từ thư mục project, không phụ thuộc thư mục terminal hiện tại. Đường dẫn tuyệt đối cũng được chấp nhận.

## Pipeline Python

- Dataset torchvision: `data/raw/cifar-100-python/{train,test,meta}`. Dùng `scripts/train.py --download` để tải đúng cấu trúc; nếu đã có pickle Kaggle, đặt ba file vào thư mục này.
- Split: `data/splits/stratified_seed42.npz`; có indices, seed và hash labels. Hai model dùng chung file này. Không thay file giữa các run dùng để so sánh.
- Checkpoint: `checkpoints/<model>/<run>/<stage>_best_accuracy.pt` và `_best_loss.pt`.
- Train outputs: `results/<model>/<run>/{metrics,figures,logs}/`, cùng config JSON.
- Evaluation outputs: `results/<model>/<run>/evaluations/<split_timestamp>/`.

Checkpoint chứa config, class order, train/val indices, stage/epoch và state_dict. Có thể chuyển checkpoint sang máy khác, nhưng cần giữ cấu trúc dataset hoặc truyền `scripts/evaluate.py --data-dir /path/to/data`. Chỉ nạp checkpoint từ nguồn tin cậy; pipeline sử dụng `weights_only=True`.

## Kết quả cũ

Notebook gốc vẫn giữ đường dẫn Kaggle và không đọc config Python. Các PNG/metrics/log đã trích từ notebook nằm trực tiếp trong `results/<model>/{figures,metrics,logs}/`; nguồn được ghi trong `results/provenance.json`. Chúng không phải output của pipeline Python mới.

Các checkpoint notebook như `best_cifar100_resnet_mixup.pth`, `best_finetune_by_acc.pth` không tương thích định dạng CLI mới. Không đổi đuôi file để nạp; hãy dùng notebook gốc hoặc viết bộ chuyển đổi riêng khi có checkpoint thực tế.

## Phân loại notebook và lệnh chạy

- Huấn luyện gốc: `notebooks/training/`.
- Phân tích feature sau train: `notebooks/analysis/`.
- Lệnh Python: `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`.
- Phần triển khai dùng chung: `src/cifar100/`.
- Kết quả feature tải từ Kaggle: `results/<model>/feature_analysis/`.

Di chuyển notebook không sửa các đường dẫn Kaggle bên trong. Giữ nguyên toàn bộ cell và outputs đã lưu.
