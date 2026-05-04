# Notes: Architectures cho Speech Deepfake Detection

Mục đích: nơi lưu kết quả research chi tiết về các approach SDD. Sẽ chắt lọc vào §2.3 của `report.md`.

---

## 1. Phân loại Approach theo Dạng Cây

Để tránh biến literature review thành danh sách model rời rạc, phần architecture nên đi từ ít nhóm lớn trước, sau đó mới chia nhỏ theo cách representation/back-end/training strategy. Một taxonomy gọn cho SDD hiện nay:

```
Speech Deepfake Detection
├── Signal/task-specific supervised detectors
│   ├── Hand-crafted acoustic features: LFCC, CQCC, MFCC, IMFCC
│   └── Learned time-frequency back-ends: LCNN, ResNet, TDNN, Transformer/Conformer
├── Raw waveform / end-to-end detectors
│   ├── RawNet/RawNet2: raw waveform encoder + temporal modeling
│   └── AASIST family: integrated spectro-temporal graph attention
├── SSL/foundation front-end + anti-spoofing back-end
│   ├── SSL front-end: Wav2Vec2, XLS-R, WavLM, HuBERT
│   ├── Back-end: AASIST, MFA, Nes2Net, pooling/MLP, SLS/layer selection
│   └── Emerging foundation front-end: Whisper encoder, other speech FMs
└── System-level robustness strategies
    ├── Codec/noise/reverb augmentation, CodecFake-aware training
    ├── Score calibration and thresholding
    └── Score-level / feature-level fusion and ensemble
```

---

### 1.1 Signal/task-specific supervised detectors

**Cách hoạt động:** Nhóm này bắt đầu từ biểu diễn tín hiệu được thiết kế hoặc chuẩn hoá theo kinh nghiệm âm học, rồi dùng classifier/back-end supervised để phân loại. Nhánh cổ điển dùng LFCC/CQCC/MFCC/IMFCC với GMM, SVM hoặc LCNN. Nhánh hiện đại hơn dùng spectrogram/CQT/LFCC làm input cho LCNN, ResNet, TDNN, Transformer hoặc Conformer. Điểm chung là mô hình học trực tiếp từ dữ liệu anti-spoofing, không có pre-training speech-scale lớn.

**Ưu điểm:**
- Nhẹ và dễ triển khai; LFCC/CQCC + GMM/LCNN vẫn là baseline chuẩn trong ASVspoof.
- Có thể phân tích failure mode theo phổ/tần số tương đối trực quan.
- Learned time-frequency back-end như LCNN/ResNet mạnh hơn GMM nhưng vẫn rẻ hơn SSL front-end.

**Nhược điểm:**
- Phụ thuộc vào spectral/phase artifacts của dataset train. Codec lossy, replay channel hoặc neural codec generation có thể làm mờ hoặc thay đổi artifact này.
- Khi generator chuyển từ vocoder-era sang codec-based/diffusion-era, cue mà mô hình học được có thể không còn ổn định.
- Nếu train trên benchmark sạch, cross-domain generalization thường là điểm yếu.

---

### 1.2 End-to-end DNN trên Raw Audio

**Cách hoạt động:** Mô hình nhận waveform trực tiếp và tự học filter/representation. RawNet2 dùng SincConv/residual blocks/GRU để học temporal cues. AASIST đi xa hơn bằng cách mô hình hóa quan hệ artifact trong cả spectral và temporal branches qua heterogeneous graph attention. Nhóm này giảm phụ thuộc vào feature engineering thủ công nhưng vẫn học chủ yếu từ dữ liệu anti-spoofing có nhãn.

**Ưu điểm:**
- Có thể học được các cues mà feature engineering truyền thống bỏ qua.
- Số tham số khiêm tốn (AASIST: khoảng 297K; AASIST-L: 85K) so với SSL front-end.
- AASIST/AASIST-L là mốc quan trọng vì đạt hiệu năng tốt trên ASVspoof 2019 LA với single compact system.

**Nhược điểm:**
- Vẫn phụ thuộc vào acoustic artifact cụ thể trong tập train; chưa mang được prior ngôn ngữ/ngữ âm từ large-scale pretraining.
- Generalization sang ASVspoof 5, codec-compressed audio hoặc in-the-wild audio có thể hạn chế nếu không có augmentation/pre-training phù hợp.

---

### 1.3 SSL/foundation front-end + anti-spoofing back-end

**Cách hoạt động:** Dùng một mô hình self-supervised/foundation speech model lớn làm front-end, sau đó gắn anti-spoofing back-end. Wav2Vec2/XLS-R/WavLM/HuBERT học representation từ pre-training quy mô lớn; back-end có thể là AASIST, MFA, Nes2Net, pooling/MLP hoặc layer-selection module. Các paper 2022-2026 cho thấy sự khác biệt không chỉ nằm ở front-end, mà còn ở cách chọn layer, fusion theo thời gian/layer, và cách xử lý high-dimensional SSL features.

**Ưu điểm:**
- SSL front-end mang prior về cấu trúc âm thanh/ngôn ngữ học từ pre-training, giúp generalize tốt hơn nhiều trên unseen attacks và codec-compressed audio.
- Nhiều hệ thống mạnh trong ASVspoof 2021/2024/2025 dùng Wav2Vec2, XLS-R hoặc WavLM front-end; WavLM+MFA đạt kết quả rất mạnh trên ASVspoof 2021 DF, còn WavLM back-end/fusion là hướng phổ biến trong ASVspoof 5.
- Linh hoạt: có thể swap front-end hoặc back-end.

**Nhược điểm:**
- Chi phí tính toán và bộ nhớ rất lớn: XLS-R 300M có ~300 triệu tham số, WavLM Large ~316M. Huấn luyện và inference trên GPU cao cấp.
- Fine-tuning toàn bộ SSL front-end đòi hỏi nhiều VRAM; nhiều nhóm chỉ fine-tune một phần hoặc dùng layer-wise learning rate.
- Hiệu suất phụ thuộc checkpoint, layer selection, augmentation và calibration; không nên kết luận SSL front-end tự động robust trong mọi threat model.

**Ví dụ đại diện:** Tak et al. (2022) chứng minh Wav2Vec2/XLS-R front-end cải thiện ASVspoof 2021 LA/DF; Guo et al. (2024) kết hợp WavLM với Multi-Fusion Attentive classifier; Borodin et al. (2024) đề xuất AASIST3; Liu et al. (2025) đề xuất Nes2Net để xử lý SSL features mà không cần dimensionality reduction; các hệ thống ASVspoof 5 dùng WavLM back-ends, augmentation và fusion để cải thiện EER/minDCF.

---

### 1.4 System-level robustness strategies

**Cách hoạt động:** Đây không phải một architecture đơn lẻ, mà là tầng thực hành giúp hệ thống generalize tốt hơn: codec/noise/reverb augmentation, training trên CodecFake/CodecFake+ hoặc codec-resynthesized speech, score calibration, và score-level/feature-level fusion. ASVspoof 5 evaluation 2026 cho thấy nhiều hệ thống vẫn suy giảm dưới adversarial attacks và neural encoding/compression, nên robustness không thể chỉ giải quyết bằng việc đổi backbone.

**Vai trò:** Nhóm này nên được trình bày sau ba nhóm model chính. Nó giải thích vì sao các hệ thống challenge thường là tổ hợp của SSL front-end, augmentation, calibration và fusion, thay vì một single backbone thuần. Trong report Internship 1, nhóm này chưa reproduce đầy đủ; nó là hướng trực tiếp cho Chương 4/Future work.

---

### 1.5 Emerging foundation-model approaches

Whisper-based detector, multi-task learning với ASR/speaker tasks và distillation từ speech foundation model là các hướng mới hơn. Whisper+AASIST là ví dụ 2024 cho việc dùng Whisper encoder như front-end. Tuy nhiên, nhóm này chưa có vai trò canonical bằng Wav2Vec2/XLS-R/WavLM trong anti-spoofing benchmark, nên nên viết ở mức emerging thay vì đặt ngang hàng với các nhóm chính trong phần đánh giá.

---

## 2. Mô tả Chi tiết 6 Model Được Reproduce

### 2.1 LFCC + LCNN

**Kiến trúc:** LFCC (Linear Frequency Cepstral Coefficients) trích xuất vector 60 chiều (20 chiều tĩnh + delta + delta-delta) với frame 20ms, bước 10ms, FFT 512 điểm, 20 filter bank tuyến tính. Back-end là LCNN (Light Convolutional Neural Network) với Max Feature Map (MFM) activation, kiến trúc lấy cảm hứng từ VGG. **Paper gốc:** Sử dụng làm baseline trong ASVspoof 2019 challenge (Wang et al., 2020). **EER báo cáo trên ASV19 LA:** Baseline đạt khoảng 8.0% (B1, GMM) đến khoảng 4–5% với LCNN back-end; hệ thống kết hợp LFCC + LCNN-LSTM có thể đạt ~1.92% với P2SGrad loss. **Params:** khoảng 60K (LCNN). **Đặc điểm nổi bật:** Nhẹ nhất trong 6 model; là baseline chuẩn của ASVspoof challenges; dễ overfit vào spectral artifact của training attacks; EER tăng mạnh trên DF datasets do codec làm mất artifact.

---

### 2.2 AASIST

**Kiến trúc:** End-to-end model nhận raw waveform. Dùng RawNet2-style encoder để trích xuất biểu diễn, sau đó xây dựng heterogeneous graph gồm hai nhánh: spectral graph và temporal graph. Một heterogeneous stacking graph attention layer (HS-GAL) kết hợp thông tin từ cả hai miền qua attention mechanism dị thể với stack node. **Paper gốc:** Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022. **EER báo cáo trên ASV19 LA:** 0.83% (min t-DCF: 0.0275). **Params:** khoảng 297K. **Đặc điểm nổi bật:** Compact nhưng hiệu suất cao; single-system (không cần ensemble) đạt kết quả rất cạnh tranh tại thời điểm 2022; graph attention cho phép mô hình hóa mối quan hệ dài hạn trong cả spectral lẫn temporal domain.

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

**Kiến trúc:** Front-end là XLS-R 300M (tương tự §2.5). Back-end là Nes2Net-X — kiến trúc nested và lightweight, thiết kế để xử lý trực tiếp high-dimensional SSL features mà không cần dimensionality reduction layer. Nes2Net-X dùng concatenation + learnable weighted summation thay vì additive combination, cho phép mô hình ưu tiên các feature layers thông tin hơn. **Paper gốc:** Liu et al., "Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing," IEEE Transactions on Information Forensics and Security, 2025 (arXiv:2504.05657). **EER báo cáo:** ASVspoof 2021 LA: ~1.66% (best run), ~1.87% average; ASVspoof 2021 DF: ~1.49%; In-the-Wild: 5.52% (best run), 6.60% average. **Params:** Nes2Net-X back-end: khoảng 511K (chỉ tính back-end); tổng với XLS-R ~300M. **Đặc điểm nổi bật:** Nes2Net-X giảm 87% chi phí tính toán back-end so với baseline trong khi cải thiện 22% hiệu suất trong paper; kiến trúc nested multi-scale giúp tận dụng đa dạng layer của SSL front-end.

---

## 3. Nhóm Approach Tiềm Năng Hiện Nay

### 3.1 Tại sao SSL + Back-end là hướng nổi bật

Kể từ ASVspoof 2021, nhiều hệ thống dựa trên SSL front-end đạt kết quả rất cạnh tranh trên các benchmark LA/DF và được dùng rộng rãi trong các challenge gần đây. Nguyên nhân chủ yếu:

1. **Rich representation từ pre-training quy mô lớn:** XLS-R 300M được pre-train trên 436K giờ audio đa ngôn ngữ; WavLM Large trên 94K giờ với mục tiêu denoising. Những biểu diễn này mang thông tin về cấu trúc âm thanh sâu rộng mà feature engineering hoặc end-to-end model nhỏ không thể đạt được.

2. **Kết quả cụ thể từ challenges:** Tại ASVspoof 2021 DF (codec-heavy), các hệ thống SSL như XLS-R+AASIST đạt EER thấp hơn nhiều baseline truyền thống. Tại ASVspoof 2024/5, nhiều hệ thống tiếp tục dùng Wav2Vec2/WavLM/XLS-R kết hợp augmentation và fusion; AASIST3, WavLM back-ends và WavLM ensemble là các ví dụ đáng chú ý.

3. **Generalization trên unseen attacks:** SSL front-end học general speech representation, không overfit vào artifact của một attack cụ thể, giúp generalize tốt hơn khi gặp attack mới hoặc audio qua codec.

### 3.2 Các Hướng Mới 2024–2026

**WavLM + fusion/back-end design:** Guo et al. (2024) dùng WavLM với Multi-Fusion Attentive classifier để khai thác thông tin theo cả time-level và layer-level. Một số hệ thống ASVspoof 5 cũng dùng WavLM front-end với pooling/back-end khác nhau, codec/noise/reverb augmentation, calibration và score fusion.

**Foundation model approach:** Một số nhóm nghiên cứu đã khám phá Whisper encoder làm front-end cho anti-spoofing (Whisper+AASIST, Qian et al., 2024), tận dụng khả năng multi-task của Whisper (ASR + speaker understanding). Hướng này mở ra khả năng học joint representation giữa nhận dạng nội dung và phát hiện artifact tổng hợp, nhưng hiện nên xem là emerging so với Wav2Vec2/XLS-R/WavLM.

**Multi-task learning:** Kết hợp objective phát hiện deepfake với các task liên quan (speaker verification, speech enhancement) để tăng cường robustness và giảm phụ thuộc vào labeled anti-spoofing data.

**Codec-aware / CodecFake-aware training:** Augment dữ liệu train bằng nhiều codec (MP3, AAC, Opus, Codec2) và mức bitrate khác nhau trong quá trình huấn luyện. Các dataset CodecFake/CodecFake+ còn đề xuất hướng train trên codec-resynthesized speech để detector nhận diện deepfake từ codec-based speech generation, thay vì chỉ học vocoder artifacts.

**Explainability và fairness:** Nghiên cứu 2024–2025 chỉ ra rằng nhiều mô hình SDD có bias với người nói cao tuổi, người nói có giọng không chuẩn, hoặc giới tính nam; đây là hướng nghiên cứu mới bên cạnh cải thiện accuracy.

### 3.3 Trade-off Computational Cost vs. Generalization

| Nhóm | Params (typical) | Inference cost | Generalization |
|------|-----------------|---------------|----------------|
| Hand-crafted + LCNN | ~60K | Thấp | Kém (codec-sensitive) |
| End-to-end (AASIST) | ~297K | Rất thấp | Trung bình |
| SSL + AASIST | ~300M | Cao (GPU) | Tốt |
| SSL + Nes2Net-X | ~300M | Cao, back-end nhẹ | Tốt–Rất tốt |
| Ensemble/fusion | N×system | Rất cao | Thường tốt trong challenge, nhưng tốn inference và cần calibration |

---

## 4. Bảng So sánh 6 Model

| Model | Front-end | Back-end | Params (approx.) | EER báo cáo (ASV19 LA) | Điểm mạnh | Điểm yếu |
|-------|-----------|----------|------------------|----------------------|-----------|-----------|
| LFCC + LCNN | LFCC (hand-crafted) | LCNN | ~60K | ~4–5% (LCNN); baseline | Nhẹ, huấn luyện nhanh | Yếu với codec, không generalize |
| AASIST | Raw waveform | Het. Graph Attention | ~297K | **0.83%** | Compact, mạnh, không cần ensemble | Yếu hơn SSL trên DF/codec |
| AASIST-L | Raw waveform | GAT (lightweight) | ~85K | **0.99%** | Cực nhẹ (~85K) | EER cao hơn AASIST, codec sensitivity |
| AASIST3 | Wav2Vec2 | KAN + AASIST | ~300M | — (minDCF 0.1414 ASV24 open) | SSL prior + KAN flexibility | Rất nặng, phụ thuộc pretrain |
| XLS-R + AASIST | XLS-R 300M | AASIST | ~300M | ~0.82% (ASV21 LA) | Generalization tốt trong các benchmark đã báo cáo | Nặng, cần fine-tune |
| XLS-R + Nes2Net-X | XLS-R 300M | Nes2Net-X | ~300M (+511K back-end) | ~1.66% (ASV21 LA) | Back-end nhẹ, kết quả mạnh trên In-the-Wild trong paper | SSL front-end vẫn tốn kém |

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

11. **Guo, Y., Huang, H., Chen, X., Zhao, H., & Wang, Y.** (2024). Audio Deepfake Detection with Self-Supervised WavLM and Multi-Fusion Attentive Classifier. *ICASSP 2024*, arXiv:2312.08089.

12. **Stourbe, T., Miara, V., Lepage, T., & Dehak, R.** (2024). Exploring WavLM Back-ends for Speech Spoofing and Deepfake Detection. *ASVspoof 2024 Workshop*, arXiv:2409.05032.

13. **Combei, D., Stan, A., Oneata, D., & Cucu, H.** (2024). WavLM Model Ensemble for Audio Deepfake Detection. *ASVspoof 2024 Workshop*, doi:10.21437/ASVspoof.2024-25.

14. **Luo, Q., & Kalyani, V. S.** (2024). Whisper+AASIST for DeepFake Audio Detection. *Lecture Notes in Computer Science*, 14729, 121-133.

15. **Viakhirev, I., Sirota, D., Smirnov, A., & Borodin, K.** (2025). Towards Scalable AASIST: Refining Graph Attention for Speech Deepfake Detection. arXiv:2507.11777.

16. **Wang, X., Delgado, H., Evans, N., et al.** (2026). ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech. *IEEE TASLP*, arXiv:2601.03944.
