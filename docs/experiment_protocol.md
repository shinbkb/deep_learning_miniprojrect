# Quy tắc thực nghiệm

- Train 45.000, validation 5.000, test 10.000 ảnh CIFAR-100.
- Seed 42 và stratified split. Cần đối chiếu split indices để xác nhận hai lần chạy dùng đúng cùng tập con.
- Chọn checkpoint bằng validation accuracy; test để đánh giá sau khi chốt cấu hình.
- Train accuracy của MixUp không so trực tiếp với accuracy ảnh sạch.
- Không so trực tiếp loss của ResNet có label smoothing với CE thường của Swin.
- So sánh này gồm cả pretraining, kích thước ảnh và cấu hình train khác nhau.
- Chưa có clean train của Swin hoặc macro-F1 của ResNet trong output cung cấp.
- Các thư mục data và checkpoints chưa chứa dataset hoặc trọng số. Notebook không nhúng các file này.
