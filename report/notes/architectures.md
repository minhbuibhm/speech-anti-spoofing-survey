# Notes: Architectures cho Speech Deepfake Detection

Mục đích: nơi lưu kết quả research chi tiết về các approach SDD. Sẽ chắt lọc vào §2.3 của `report.md`.

---

## 1. Phân loại Approach theo Dạng Cây

Các hướng tiếp cận trong speech deepfake detection (SDD) có thể tổ chức theo cấu trúc cây từ high-level xuống low-level như sau:

```
Speech Deepfake Detection
├── Hand-crafted features + Classifier
│   ├── Features: LFCC, CQCC, MFCC, LPC, IMFCC
│   └── Classifiers: GMM, SVM, LCNN, ResNet, TDNN
│
├── End-to-end DNN trên Raw Audio
│   ├── RawNet / RawNet2 (SincNet + residual blocks)
│   └── AASIST family
│       ├── AASIST (heterogeneous graph attention)
│       ├── AASIST-L (lightweight variant)
│       └── AASIST3 (SSL + KAN + AASIST)
│
├── SSL Front-end + Back-end
│   ├── Front-end (feature extractor được pre-train):
│   │   ├── Wav2Vec2 (Baevski et al., 2020)
│   │   ├── XLS-R 300M / 1B / 2B (Babu et al., 2022)
│   │   ├── WavLM Base / Large (Chen et al., 2022)
│   │   └── HuBERT Base / Large (Hsu et al., 2021)
│   └── Back-end (classifier/aggregator):
│       ├── AASIST back-end
│       ├── Nes2Net / Nes2Net-X (Liu et al., 2025)
│       ├── MFA (Multi-Fusion Attention)
│       ├── SLS (Sensitive Layer Selection)
│       └── Linear / MLP đơn giản
│
├── Foundation Model Approach (xu hướng 2024–2026)
│   ├── Whisper-based (Whisper + AASIST/classifier)
│   ├── Multi-task learning với speech task khác
│   └── Codec-aware training / knowledge distillation
│
└── Ensemble Methods
    ├── Score-level fusion
    └── Feature-level fusion (multi-frontend)
```

---

### 1.1 Hand-crafted Features + Classifier

**Cách hoạt động:** Bước feature extraction được thiết kế bằng tay dựa trên hiểu biết về âm học. LFCC (Linear Frequency Cepstral Coefficients) trích xuất đặc trưng phổ tuyến tính; CQCC (Constant-Q Cepstral Coefficients) dùng biến đổi Constant-Q để bắt được các artifact ở cả tần số thấp lẫn cao. Các vector đặc trưng 60 chiều (tĩnh + delta + delta-delta) được đưa vào classifier như GMM hoặc LCNN.

**Ưu điểm:**
- Số lượng tham số rất nhỏ (~60K cho LFCC+LCNN), huấn luyện nhanh.
- Không đòi hỏi GPU lớn; phù hợp cho edge deployment.
- Dễ giải thích về mặt tín hiệu học.

**Nhược điểm:**
- Phụ thuộc vào spectral artifacts cụ thể của từng hệ thống TTS/VC. Khi audio bị nén qua codec (MP3, AAC, Opus), các artifact này bị xóa, dẫn đến EER tăng vọt trên ASVspoof 2021 DF.
- Khả năng generalize kém trên các attack không thấy trong training (unseen attacks).
- Bottleneck ở thiết kế feature: không tự thích ứng với dữ liệu mới.

---

### 1.2 End-to-end DNN trên Raw Audio

**Cách hoạt động:** Mô hình học trực tiếp từ raw waveform, bỏ qua bước feature engineering bằng tay. RawNet2 dùng SincNet layer + residual blocks + GRU. AASIST mở rộng bằng cách mô hình hóa artifact trên cả hai chiều spectral và temporal thông qua graph attention network dị thể (heterogeneous graph attention).

**Ưu điểm:**
- Có thể học được các cues mà feature engineering truyền thống bỏ qua.
- Số tham số khiêm tốn (AASIST: khoảng 297K; AASIST-L: 85K) so với SSL front-end.
- AASIST đạt EER 0.83% trên ASVspoof 2019 LA — rất cạnh tranh với các hệ thống phức tạp hơn nhiều.

**Nhược điểm:**
- Vẫn phụ thuộc vào acoustic artifact cụ thể trong tập train; chưa mang được prior ngôn ngữ/ngữ âm từ large-scale pretraining.
- Generalization sang codec-compressed audio còn hạn chế do không có pre-train trên đa dạng dữ liệu.

---

### 1.3 SSL Front-end + Back-end

**Cách hoạt động:** Dùng một mô hình self-supervised learning (SSL) lớn (đã pre-train trên hàng chục nghìn giờ audio) làm front-end để trích xuất biểu diễn đặc trưng phong phú. Các biểu diễn này sau đó được đưa vào một back-end nhẹ (AASIST, Nes2Net, SLS, ...) để phân loại bonafide/spoof. Trong quá trình fine-tune, SSL front-end thường được fine-tune cùng back-end trên tập train anti-spoofing.

**Ưu điểm:**
- SSL front-end mang prior về cấu trúc âm thanh/ngôn ngữ học từ pre-training, giúp generalize tốt hơn nhiều trên unseen attacks và codec-compressed audio.
- Dominate leaderboard ASVspoof 2021, 2024 và các challenge gần đây.
- Linh hoạt: có thể swap front-end hoặc back-end.

**Nhược điểm:**
- Chi phí tính toán và bộ nhớ rất lớn: XLS-R 300M có ~300 triệu tham số, WavLM Large ~316M. Huấn luyện và inference trên GPU cao cấp.
- Fine-tuning toàn bộ SSL front-end đòi hỏi nhiều VRAM; nhiều nhóm chỉ fine-tune một phần hoặc dùng layer-wise learning rate.
- Hiệu suất phụ thuộc chất lượng tập training; nếu tập quá nhỏ, có thể overfit.

**Trade-off params/generalization/codec:** WavLM Large (~316M params) nhờ pre-training kết hợp masked prediction và denoising, được báo cáo robust hơn với noisy/compressed audio so với Wav2Vec2 hay XLS-R. Tuy nhiên chi phí inference cao hơn đáng kể so với AASIST (297K params).

---

### 1.4 Foundation Model Approach (2024–2026)

Xu hướng mới nhất tận dụng các mô hình ngôn ngữ/âm thanh có quy mô cực lớn như Whisper (OpenAI), AudioPaLM, hoặc kết hợp multi-modal LLM. Hướng tiếp cận bao gồm: dùng Whisper encoder làm front-end thay cho SSL models; multi-task learning để cùng lúc học nhận dạng tiếng nói và phát hiện deepfake; codec-aware training bằng cách augment dữ liệu với nhiều loại codec và mức bitrate khác nhau trong quá trình huấn luyện. Một hướng khác là knowledge distillation từ mô hình SSL lớn sang mô hình nhỏ hơn, giúp cân bằng giữa chi phí và hiệu suất.

---

### 1.5 Ensemble Methods

Kết hợp nhiều hệ thống thông qua score-level fusion (trung bình/trọng số điểm phân loại) hoặc feature-level fusion (ghép đặc trưng từ nhiều front-end). Phổ biến trong các hệ thống top-tier tại ASVspoof challenges. Nhược điểm chính là chi phí inference nhân lên theo số lượng hệ thống con.

---

## 2. Mô tả Chi tiết 6 Model Được Reproduce

### 2.1 LFCC + LCNN

**Kiến trúc:** LFCC (Linear Frequency Cepstral Coefficients) trích xuất vector 60 chiều (20 chiều tĩnh + delta + delta-delta) với frame 20ms, bước 10ms, FFT 512 điểm, 20 filter bank tuyến tính. Back-end là LCNN (Light Convolutional Neural Network) với Max Feature Map (MFM) activation, kiến trúc lấy cảm hứng từ VGG. **Paper gốc:** Sử dụng làm baseline trong ASVspoof 2019 challenge (Wang et al., 2020). **EER báo cáo trên ASV19 LA:** Baseline đạt khoảng 8.0% (B1, GMM) đến khoảng 4–5% với LCNN back-end; hệ thống kết hợp LFCC + LCNN-LSTM có thể đạt ~1.92% với P2SGrad loss. **Params:** khoảng 60K (LCNN). **Đặc điểm nổi bật:** Nhẹ nhất trong 6 model; là baseline chuẩn của ASVspoof challenges; dễ overfit vào spectral artifact của training attacks; EER tăng mạnh trên DF datasets do codec làm mất artifact.

---

### 2.2 AASIST

**Kiến trúc:** End-to-end model nhận raw waveform. Dùng RawNet2-style encoder để trích xuất biểu diễn, sau đó xây dựng heterogeneous graph gồm hai nhánh: spectral graph và temporal graph. Một heterogeneous stacking graph attention layer (HS-GAL) kết hợp thông tin từ cả hai miền qua attention mechanism dị thể với stack node. **Paper gốc:** Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022. **EER báo cáo trên ASV19 LA:** 0.83% (min t-DCF: 0.0275). **Params:** khoảng 297K. **Đặc điểm nổi bật:** Compact nhưng hiệu suất cao; single-system (không cần ensemble) đã vượt SOTA tại thời điểm 2022; graph attention cho phép mô hình hóa mối quan hệ dài hạn trong cả spectral lẫn temporal domain.

---

### 2.3 AASIST-L

**Kiến trúc:** Phiên bản lightweight của AASIST, thu nhỏ số kênh và độ phức tạp của các graph attention layer. Kiến trúc tổng thể giữ nguyên thiết kế heterogeneous graph từ AASIST nhưng với capacity thấp hơn đáng kể. **Paper gốc:** Jung et al., ICASSP 2022 (cùng paper với AASIST). **EER báo cáo trên ASV19 LA:** 0.99% (min t-DCF: 0.0309). **Params:** 85,306 (~85K). **Đặc điểm nổi bật:** Cực kỳ nhỏ gọn — nhỏ hơn AASIST khoảng 3.5 lần nhưng chỉ mất khoảng 0.16% EER tuyệt đối; phù hợp cho edge deployment hoặc khi tài nguyên tính toán hạn chế.

---

### 2.4 AASIST3

**Kiến trúc:** Mở rộng AASIST bằng cách tích hợp SSL front-end (Wav2Vec2) và back-end tăng cường bởi KAN (Kolmogorov-Arnold Networks). Pipeline gồm: (1) Wav2Vec2 encoder trích xuất SSL features từ raw audio; (2) KAN Bridge biến đổi và nén SSL features thông qua learnable activation functions; (3) Residual encoder + AASIST back-end phân loại. Ngoài ra, AASIST3 tích hợp pre-emphasis filtering và thêm regularization để tránh overfitting. **Paper gốc:** Borodin et al., "AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization for the ASVspoof 2024 Challenge," arXiv 2408.17352, 2024. **Kết quả ASVspoof 2024:** minDCF 0.5357 (closed condition), 0.1414 (open condition). **Params:** khoảng 300M (dominated by Wav2Vec2 encoder). **Đặc điểm nổi bật:** Đưa KAN vào pipeline anti-spoofing; báo cáo cải thiện hơn 2 lần so với AASIST baseline trong điều kiện open của ASVspoof 2024; available trên HuggingFace (MTUCI/AASIST3).

---

### 2.5 XLS-R + AASIST (SSL-AASIST)

**Kiến trúc:** Front-end là XLS-R 300M — phiên bản cross-lingual của wav2vec 2.0, được pre-train trên ~436K giờ audio đa ngôn ngữ (128 ngôn ngữ) theo Babu et al. (2022). Back-end là AASIST (cùng kiến trúc graph attention như §2.2). Trong quá trình fine-tune, XLS-R được tối ưu chung với AASIST back-end qua back-propagation trên ASVspoof 2019 LA training set. **Paper gốc:** Tak et al., "Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation," Odyssey 2022 (arXiv:2202.12233). **EER báo cáo:** Trên ASVspoof 2021 LA, hệ thống XLS-R+AASIST đạt EER ~0.82% — giảm ~82% tương đối so với baseline; trên ASVspoof 2021 DF, đạt khoảng 2.85% EER. **Params:** khoảng 300M (XLS-R 300M chiếm phần lớn). **Đặc điểm nổi bật:** Đây là một trong những hệ thống đầu tiên chứng minh giá trị của SSL front-end cho anti-spoofing; XLS-R được pre-train trên dữ liệu đa dạng giúp generalize tốt hơn AASIST thuần; vẫn gặp khó khăn trên heavily codec-compressed audio.

---

### 2.6 XLS-R + Nes2Net (Nes2Net-X)

**Kiến trúc:** Front-end là XLS-R 300M (tương tự §2.5). Back-end là Nes2Net-X — kiến trúc nested và lightweight, thiết kế để xử lý trực tiếp high-dimensional SSL features mà không cần dimensionality reduction layer. Nes2Net-X dùng concatenation + learnable weighted summation thay vì additive combination, cho phép mô hình ưu tiên các feature layers thông tin hơn. **Paper gốc:** Liu et al., "Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing," IEEE Transactions on Information Forensics and Security, 2025 (arXiv:2504.05657). **EER báo cáo:** ASVspoof 2021 LA: ~1.66% (best run), ~1.87% average; ASVspoof 2021 DF: ~1.49%; In-the-Wild: 5.52% (best run), 6.60% average. **Params:** Nes2Net-X back-end: khoảng 511K (chỉ tính back-end); tổng với XLS-R ~300M. **Đặc điểm nổi bật:** Nes2Net-X giảm 87% chi phí tính toán back-end so với baseline trong khi cải thiện 22% hiệu suất; đạt SOTA trên In-the-Wild dataset thực tế; kiến trúc nested multi-scale giúp tận dụng đa dạng layer của SSL front-end.

---

## 3. Nhóm Approach Tiềm Năng Hiện Nay

### 3.1 Tại sao SSL + Back-end đang Dominate

Kể từ ASVspoof 2021, các hệ thống dựa trên SSL front-end liên tục dẫn đầu leaderboard. Nguyên nhân chủ yếu:

1. **Rich representation từ pre-training quy mô lớn:** XLS-R 300M được pre-train trên 436K giờ audio đa ngôn ngữ; WavLM Large trên 94K giờ với mục tiêu denoising. Những biểu diễn này mang thông tin về cấu trúc âm thanh sâu rộng mà feature engineering hoặc end-to-end model nhỏ không thể đạt được.

2. **Kết quả cụ thể từ challenges:** Tại ASVspoof 2021 DF (codec-heavy), các hệ thống top-tier đều dùng SSL front-end: XLS-R+AASIST đạt ~2.85% EER, trong khi LFCC+GMM baseline lên tới hàng chục phần trăm. Tại ASVspoof 2024 (open track), AASIST3 với Wav2Vec2 đạt minDCF 0.1414, cải thiện hơn 2 lần so với AASIST thuần.

3. **Generalization trên unseen attacks:** SSL front-end học general speech representation, không overfit vào artifact của một attack cụ thể, giúp generalize tốt hơn khi gặp attack mới hoặc audio qua codec.

### 3.2 Các Hướng Mới 2024–2026

**Foundation model approach:** Một số nhóm nghiên cứu đã khám phá Whisper encoder làm front-end cho anti-spoofing (Whisper+AASIST, Qian et al., 2024), tận dụng khả năng multi-task của Whisper (ASR + speaker understanding). Hướng này mở ra khả năng học joint representation giữa nhận dạng nội dung và phát hiện artifact tổng hợp.

**Multi-task learning:** Kết hợp objective phát hiện deepfake với các task liên quan (speaker verification, speech enhancement) để tăng cường robustness và giảm phụ thuộc vào labeled anti-spoofing data.

**Codec-aware training:** Augment dữ liệu train bằng nhiều codec (MP3, AAC, Opus, Codec2) và mức bitrate khác nhau trong quá trình huấn luyện. Kỹ thuật FTDKD (Frequency-Time Domain Knowledge Distillation) cho phép mô hình nhẹ học từ mô hình lớn trên audio chất lượng cao.

**Explainability và fairness:** Nghiên cứu 2024–2025 chỉ ra rằng nhiều mô hình SDD có bias với người nói cao tuổi, người nói có giọng không chuẩn, hoặc giới tính nam; đây là hướng nghiên cứu mới bên cạnh cải thiện accuracy.

### 3.3 Trade-off Computational Cost vs. Generalization

| Nhóm | Params (typical) | Inference cost | Generalization |
|------|-----------------|---------------|----------------|
| Hand-crafted + LCNN | ~60K | Thấp | Kém (codec-sensitive) |
| End-to-end (AASIST) | ~297K | Rất thấp | Trung bình |
| SSL + AASIST | ~300M | Cao (GPU) | Tốt |
| SSL + Nes2Net-X | ~300M | Cao, back-end nhẹ | Tốt–Rất tốt |
| Ensemble | N×system | Rất cao | Tốt nhất |

---

## 4. Bảng So sánh 6 Model

| Model | Front-end | Back-end | Params (approx.) | EER báo cáo (ASV19 LA) | Điểm mạnh | Điểm yếu |
|-------|-----------|----------|------------------|----------------------|-----------|-----------|
| LFCC + LCNN | LFCC (hand-crafted) | LCNN | ~60K | ~4–5% (LCNN); baseline | Nhẹ, huấn luyện nhanh | Yếu với codec, không generalize |
| AASIST | Raw waveform | Het. Graph Attention | ~297K | **0.83%** | Compact, mạnh, không cần ensemble | Yếu hơn SSL trên DF/codec |
| AASIST-L | Raw waveform | GAT (lightweight) | ~85K | **0.99%** | Cực nhẹ (~85K) | EER cao hơn AASIST, codec sensitivity |
| AASIST3 | Wav2Vec2 | KAN + AASIST | ~300M | — (minDCF 0.1414 ASV24 open) | SSL prior + KAN flexibility | Rất nặng, phụ thuộc pretrain |
| XLS-R + AASIST | XLS-R 300M | AASIST | ~300M | ~0.82% (ASV21 LA) | Generalize tốt, SOTA 2022 | Nặng, cần fine-tune |
| XLS-R + Nes2Net-X | XLS-R 300M | Nes2Net-X | ~300M (+511K back-end) | ~1.66% (ASV21 LA) | Back-end nhẹ, SOTA In-the-Wild | SSL front-end vẫn tốn kém |

*Lưu ý: EER cho LFCC+LCNN trên ASV19 LA phụ thuộc vào cấu hình; LFCC+GMM baseline (B1) đạt ~8%; LCNN-LSTM với P2SGrad loss có thể đạt ~1.92%. Số liệu AASIST, AASIST-L lấy từ Jung et al. (ICASSP 2022). Số liệu XLS-R+AASIST trên ASVspoof 2021 LA từ Tak et al. (Odyssey 2022). Số liệu Nes2Net-X từ Liu et al. (IEEE TIFS 2025).*

---

## 5. Tham chiếu

1. **Jung, J.-w., Heo, H.-S., Tak, H., Shim, H.-j., Chung, J. S., Lee, B.-J., Yu, H.-J., & Evans, N.** (2022). AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 6367–6371.

2. **Tak, H., Todisco, M., Wang, X., Jung, J.-w., Yamagishi, J., & Evans, N.** (2022). Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation. *Proceedings of the Speaker Odyssey Workshop*, arXiv:2202.12233.

3. **Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y., Pino, J., Baevski, A., Conneau, A., & Auli, M.** (2022). XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale. *Interspeech 2022*, arXiv:2111.09296.

4. **Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Wu, J., Zeng, M., Yu, X., & Wei, F.** (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE Journal on Selected Topics in Signal Processing*, 16(6), 1505–1518.

5. **Liu, T., Truong, D.-T., Das, R. K., Lee, K. A., & Li, H.** (2025). Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing. *IEEE Transactions on Information Forensics and Security*, 20, 12005–12018, arXiv:2504.05657.

6. **Borodin, K., Kudryavtsev, V., Korzh, D., Efimenko, A., Mkrtchian, G., Gorodnichev, M., & Rogov, O. Y.** (2024). AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization for the ASVspoof 2024 Challenge. arXiv:2408.17352.

7. **Li, J., et al.** (2024). A Survey on Speech Deepfake Detection. *ACM Computing Surveys*, doi:10.1145/3714458. arXiv:2404.13914.

8. **Wang, X., Yamagishi, J., Todisco, M., Delgado, H., Nautsch, A., Evans, N., Schwarz, A., Valentin Pla, A., & Galindo, F.** (2020). ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech. *Computer Speech & Language*, 64, 101114.

9. **Tak, H., Patino, J., Todisco, M., Nautsch, A., Evans, N., & Larcher, A.** (2021). End-to-End Anti-Spoofing with RawNet2. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 6369–6373, arXiv:2011.01108.

10. **PMC Review (2025).** Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead. *PMC*, PMCID: PMC11991371.
