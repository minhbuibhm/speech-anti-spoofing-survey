# Notes: Các loại tấn công giọng nói giả mạo

Mục đích: tổng hợp các attack types liên quan đến speech deepfake. Sẽ chắt lọc vào §2.1 của `report.md`.

## Phân loại

- **Logical Access (LA)**: tấn công bằng cách tiêm tín hiệu giả vào hệ thống ở mức số (digital injection).
  - **Text-to-Speech (TTS)**: sinh giọng từ văn bản (Tacotron, FastSpeech, VITS, neural codec TTS).
  - **Voice Conversion (VC)**: chuyển giọng nguồn thành giọng đích.
- **Physical Access (PA) / Replay**: phát lại bản ghi qua loa và thu lại bằng micro — không cần mô hình sinh.
- **Adversarial attacks**: tạo perturbation nhỏ vào audio thật để vượt qua hệ thống detection (gradient-based, black-box).
- **Neural codec attacks (mới 2024-2026)**: dùng neural codec làm bottleneck vừa để nén vừa để tạo audio giả khó phát hiện.

## Bảng nhanh

| Loại | Mô tả ngắn | Đặc điểm dấu vết | Dataset đại diện |
|------|------------|------------------|------------------|
| TTS | Sinh từ text | Spectral artifact của vocoder | ASVspoof 2019 LA |
| VC | Chuyển giọng | Mismatch giữa nội dung và phổ giọng | ASVspoof 2019 LA |
| Replay | Phát lại | Phản hồi phòng, phổ thiết bị | ASVspoof 2019 PA, EchoFake |
| Adversarial | Perturbation | Không nhìn thấy bằng tai | ASVspoof 5 |
| Neural codec | Nén bằng neural | Mất artifact truyền thống | ASVspoof 5 (2024) |

*[Điền chi tiết khi research thêm.]*

## Tham chiếu

*[Liệt kê các paper được dùng.]*
