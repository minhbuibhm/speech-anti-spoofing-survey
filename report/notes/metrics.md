# Notes: Evaluation metrics cho Speech Deepfake Detection

Mục đích: tổng hợp các metric được dùng để đánh giá. Sẽ chắt lọc vào §2.4 của `report.md`.

## Equal Error Rate (EER)

- **Định nghĩa**: ngưỡng quyết định mà tại đó tỉ lệ False Acceptance Rate (FAR) bằng tỉ lệ False Rejection Rate (FRR). Giá trị này thường được báo cáo dưới dạng phần trăm.
- **Công thức**: với hàm FAR(τ) và FRR(τ) theo ngưỡng τ, EER là giá trị FAR(τ*) = FRR(τ*) tại ngưỡng τ*.
- **Ý nghĩa**: phản ánh hiệu năng tổng hợp ở điểm cân bằng giữa hai loại lỗi; càng thấp càng tốt.
- **Ưu điểm**: không cần chọn ngưỡng cố định, dễ so sánh giữa các hệ thống.
- **Nhược điểm**: không phản ánh được chi phí thực tế khi hai loại lỗi có cost khác nhau.

## Minimum tandem Detection Cost Function (min t-DCF)

- **Định nghĩa**: metric do ASVspoof 2019 đề xuất để đánh giá CM khi tích hợp vào hệ thống ASV (automatic speaker verification).
- **Khi nào dùng**: khi đánh giá CM trong ngữ cảnh hệ thống ASV thực; cần thông tin về ASV system kèm theo.
- **Lưu ý**: không phải lúc nào cũng tính được nếu chỉ có CM scores đơn lẻ — báo cáo này dùng EER là chính.

## ROC / DET curve

- **ROC**: True Positive Rate vs False Positive Rate.
- **DET (Detection Error Tradeoff)**: FAR vs FRR trên thang xác suất (Gaussian-warped). Chuẩn trong cộng đồng speaker/spoofing detection.

## Calibration

- **Mô tả ngắn**: model có thể có EER tốt nhưng score không calibrated (không phản ánh xác suất thật) — quan trọng khi cần ngưỡng cố định trong production.
- **Trong báo cáo này**: chỉ đề cập ở mức cao, không đi sâu (do scope Internship 1).

## Tham chiếu

*[Liệt kê các paper được dùng.]*
