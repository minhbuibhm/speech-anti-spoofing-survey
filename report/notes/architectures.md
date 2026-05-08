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

**Vai trò:** Nhóm này nên được trình bày sau ba nhóm model chính. Nó giải thích vì sao các hệ thống challenge thường là tổ hợp của SSL front-end, augmentation, calibration và fusion, thay vì một single backbone thuần. Trong report Thực tập 1, nhóm này chưa reproduce đầy đủ; nó là hướng trực tiếp cho Chương 4/Future work.

---

### 1.5 Emerging foundation-model approaches

Whisper-based detector, multi-task learning với ASR/speaker tasks và distillation từ speech foundation model là các hướng mới hơn. Whisper+AASIST là ví dụ 2024 cho việc dùng Whisper encoder như front-end. Tuy nhiên, nhóm này chưa có vai trò canonical bằng Wav2Vec2/XLS-R/WavLM trong anti-spoofing benchmark, nên nên viết ở mức emerging thay vì đặt ngang hàng với các nhóm chính trong phần đánh giá.

---

## 2. Mô tả Chi tiết 6 Model Được Reproduce

### 2.1 LFCC + LCNN

**Kiến trúc:** LFCC (Linear Frequency Cepstral Coefficients) trích xuất đặc trưng phổ tuyến tính theo frame, thường kèm delta và delta-delta, rồi đưa vào LCNN (Light Convolutional Neural Network) với Max Feature Map (MFM) activation. Đây là đại diện cho hướng hand-crafted acoustic feature + supervised classifier. **Paper/context:** LFCC-GMM và LFCC-LCNN là các baseline phổ biến trong ASVspoof 2019/2021; một số biến thể LCNN/LCNN-LSTM tốt hơn baseline GMM, nhưng không nên gọi LFCC+LCNN nói chung là SOTA nếu không nêu đúng variant và training recipe. **Params:** phụ thuộc implementation; nhìn chung nhỏ hơn rất nhiều so với SSL models, nhưng cần tránh khẳng định một con số cố định nếu chưa verify từ code project. **Đặc điểm nổi bật:** Rẻ nhất về compute, dễ train, phù hợp làm baseline. **Failure mode:** Nhạy với codec/compression, channel mismatch và unseen generator artifacts vì mô hình dễ học cue phổ cục bộ gắn với training distribution. Trong report, LFCC+LCNN nên được dùng như baseline truyền thống, không phải model đại diện cho best published performance.

---

### 2.2 AASIST

**Kiến trúc:** AASIST là end-to-end detector nhận raw waveform. Mô hình dùng RawNet2-style encoder để học representation ban đầu, sau đó xây dựng heterogeneous graph gồm spectral graph và temporal graph. Heterogeneous stacking graph attention layer (HS-GAL) kết hợp thông tin giữa hai miền bằng attention dị thể và stack node, nhằm phát hiện artifacts không chỉ nằm ở một frame hoặc một dải tần riêng lẻ mà có thể trải trên quan hệ spectro-temporal dài hơn. **Paper gốc:** Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022. **Metric paper:** ASVspoof 2019 LA, 0.83% EER và 0.0275 min t-DCF cho best single system. **Params:** khoảng 297K. **Vai trò SOTA/former-SOTA:** Đây là former-SOTA compact single model trên ASVspoof 2019 LA, mạnh vì đạt performance cao mà không cần ensemble/score fusion. **Failure mode:** Không có large-scale speech pretraining; vẫn học chủ yếu từ labeled anti-spoofing data, nên có thể suy giảm khi gặp codec, in-the-wild audio, ASVspoof 5 modern attacks hoặc distribution khác ASVspoof 2019.

---

### 2.3 AASIST-L

**Kiến trúc:** AASIST-L là phiên bản lightweight của AASIST, thu nhỏ số kênh và độ phức tạp của graph attention layers nhưng giữ ý tưởng heterogeneous spectro-temporal graph. **Paper/context:** Cùng paper ICASSP 2022 với AASIST. **Metric paper:** ASVspoof 2019 LA, 0.99% EER và 0.0309 min t-DCF. **Params:** 85,306, khoảng 85K. **Vai trò:** Đại diện cho hướng edge/constrained deployment: nhỏ hơn AASIST khoảng 3.5 lần nhưng chỉ kém 0.16 điểm EER tuyệt đối trên ASV19 LA paper. **Failure mode:** Capacity thấp hơn AASIST và có cùng rủi ro thiếu SSL prior; khi domain shift mạnh, lợi thế nhỏ gọn không đồng nghĩa robustness tốt.

---

### 2.4 AASIST3

**Kiến trúc:** AASIST3 mở rộng AASIST bằng SSL front-end (Wav2Vec2), KAN bridge (Kolmogorov-Arnold Networks), residual encoder, pre-emphasis và regularization. Pipeline tổng quát gồm: SSL encoder trích xuất features từ waveform, KAN bridge biến đổi/nén high-dimensional SSL features, sau đó AASIST-style back-end phân loại. **Paper gốc:** Borodin et al., "AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization for the ASVspoof 2024 Challenge," ASVspoof 2024 Workshop. **Metric paper:** ASVspoof 2024 minDCF 0.5357 trong closed condition và 0.1414 trong open condition. Đây là **minDCF**, không nên so trực tiếp với EER hoặc min t-DCF của ASVspoof 2019. **Params:** hàng trăm triệu nếu tính SSL front-end; compute dominated by Wav2Vec2. **Vai trò:** Đại diện cho ASVspoof 2024-context system dùng SSL + KAN, nhưng không nên gọi là SOTA rộng cho mọi benchmark. **Failure mode:** Phụ thuộc checkpoint/challenge condition; kết quả reproduce trong project cho thấy SSL front-end không tự động đảm bảo generalization nếu front-end/back-end/training không khớp domain.

---

### 2.5 XLS-R + AASIST (SSL-AASIST)

**Kiến trúc:** Front-end là XLS-R 300M, phiên bản cross-lingual của wav2vec 2.0 được pre-train trên dữ liệu audio đa ngôn ngữ quy mô lớn. Back-end là AASIST, tận dụng graph attention đã được thiết kế cho anti-spoofing. Hệ thống có thể fine-tune SSL front-end cùng back-end hoặc dùng checkpoint đã công bố. **Paper gốc:** Tak et al., "Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation," Odyssey 2022. **Metric paper:** ASVspoof 2021 LA EER khoảng 0.82%; ASVspoof 2021 DF EER khoảng 2.85% trong setup có data augmentation. Đây là EER cho CM/deepfake detection, không phải t-DCF. **Params:** khoảng 300M+; XLS-R chiếm gần như toàn bộ tham số và FLOPs. **Vai trò SOTA/former-SOTA:** Một trong các evidence sớm và mạnh cho thấy SSL/foundation speech representation cải thiện anti-spoofing trên LA/DF, đặc biệt khi có codec/channel shift. **Failure mode:** Tốn VRAM, latency cao; vẫn có thể suy giảm với codec/attack ngoài training hoặc khi calibration/threshold không phù hợp.

---

### 2.6 XLS-R + Nes2Net (Nes2Net-X)

**Kiến trúc:** Front-end là XLS-R 300M. Back-end là Nes2Net-X, một nested lightweight architecture được thiết kế để xử lý trực tiếp high-dimensional SSL features. Thay vì chỉ pooling một tầng SSL output, Nes2Net-X dùng cấu trúc nested/multi-scale, concatenation và learnable weighted summation để khai thác nhiều mức thông tin từ SSL representation. **Paper gốc:** Liu et al., "Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing," IEEE TIFS 2025 / arXiv:2504.05657. **Metric paper:** báo cáo kết quả mạnh trên ASVspoof 2021 LA/DF và In-the-Wild; các số thường được trích gồm ASVspoof 2021 LA khoảng 1.66% EER best run, ASVspoof 2021 DF khoảng 1.49% EER, In-the-Wild 5.52% best / 6.60% average. Cần lock lại từ PDF nếu đưa từng số vào report chính thức. **Params:** total vẫn khoảng 300M vì XLS-R dominates; Nes2Net-X back-end khoảng 0.5M. **Vai trò SOTA/former-SOTA:** Đại diện tốt cho hướng foundation front-end + lightweight anti-spoofing back-end. **Failure mode:** Backend nhẹ không xoá được bottleneck compute của XLS-R; nếu codec/attack quá khác domain, SSL front-end vẫn có thể tạo EER cao như project quan sát trên ASVspoof 5.

---

### 2.7 Hai SOTA/former-SOTA nên viết sâu trong report

**AASIST / AASIST-L** nên được chọn làm former-SOTA compact family. Lý do: paper ICASSP 2022 báo cáo AASIST đạt 0.83% EER và 0.0275 min t-DCF trên ASVspoof 2019 LA với khoảng 297K tham số, còn AASIST-L đạt 0.99% EER và 0.0309 min t-DCF với khoảng 85K tham số. Đây là evidence rõ cho một single compact countermeasure không cần ensemble. Khi viết report, cần nhấn mạnh context: former-SOTA trên ASVspoof 2019 LA, không phải đảm bảo robust trên ASVspoof 5 hoặc In-the-Wild.

**XLS-R + Nes2Net-X** nên được chọn làm đại diện SOTA/former-SOTA cho nhóm SSL/foundation front-end + lightweight back-end. Lý do: nó trực tiếp nằm trong 6 model reproduce, thể hiện trade-off hiện đại giữa representation mạnh và back-end gọn, đồng thời kết quả project cho thấy generalization tốt trên ASVspoof 2021 DF và In-the-Wild hơn nhóm không dùng SSL. Khi viết report, cần ghi rõ tổng hệ thống vẫn nặng vì XLS-R 300M; "lightweight" chủ yếu nói về back-end, không phải toàn pipeline. Có thể nhắc **XLS-R + AASIST** như mốc SSL-AASIST sớm của Tak et al. chứng minh giá trị của wav2vec2/XLS-R front-end cho ASVspoof 2021 LA/DF.

**AASIST3** nên được mô tả là ASVspoof 2024 challenge-context system dùng Wav2Vec2 + KAN + AASIST, không chọn làm SOTA chính cho report nếu mục tiêu là general SDD benchmark. Metric paper của AASIST3 là minDCF trong ASVspoof 2024 closed/open condition, nên phải tách khỏi EER của project.

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
| Hand-crafted + LCNN | Nhỏ, tuỳ implementation | Thấp | Kém đến trung bình; codec-sensitive |
| End-to-end (AASIST) | ~297K | Rất thấp | Trung bình |
| SSL + AASIST | ~300M | Cao (GPU) | Tốt |
| SSL + Nes2Net-X | ~300M total; ~0.5M back-end | Cao do XLS-R, back-end nhẹ | Tốt–Rất tốt trên ASV21/ITW; ASV5 vẫn khó |
| Ensemble/fusion | N×system | Rất cao | Thường tốt trong challenge, nhưng tốn inference và cần calibration |

---

## 4. Bảng So sánh 6 Model

| Model | Front-end | Back-end | Params (approx.) | Paper metric đúng context | Vai trò | Failure mode chính |
|-------|-----------|----------|------------------|---------------------------|---------|--------------------|
| LFCC + LCNN | LFCC hand-crafted | LCNN/MFM | Nhỏ, tuỳ implementation | Baseline family trong ASVspoof; không cite như SOTA nếu không đúng variant | Baseline signal-processing truyền thống | Nhạy codec/compression, unseen artifacts, domain shift |
| AASIST | Raw waveform | Integrated spectro-temporal heterogeneous graph attention | ~297K | ASVspoof 2019 LA: **0.83% EER**, **0.0275 min t-DCF** | Former-SOTA compact single system | Thiếu SSL prior; giảm khi domain/codec/attack thay đổi |
| AASIST-L | Raw waveform | Lightweight AASIST | ~85K | ASVspoof 2019 LA: **0.99% EER**, **0.0309 min t-DCF** | Edge/compact representative | Capacity thấp hơn AASIST; cùng họ lỗi với AASIST |
| AASIST3 | Wav2Vec2/SSL | KAN bridge + modified AASIST | Hàng trăm M nếu tính SSL | ASVspoof 2024: minDCF 0.5357 closed, 0.1414 open | Challenge-context SSL+KAN system | Không nên gọi SOTA rộng; phụ thuộc checkpoint/condition |
| XLS-R + AASIST | XLS-R 300M | AASIST | ~300M+ | ASVspoof 2021 LA: ~0.82% EER; DF: ~2.85% EER | Early strong SSL-AASIST evidence | Nặng, tốn VRAM; vẫn cần threshold/calibration tốt |
| XLS-R + Nes2Net-X | XLS-R 300M | Nes2Net-X nested backend | ~300M total; ~0.5M back-end | Strong reported results on ASVspoof 2021 and In-the-Wild; exact table nên verify từ PDF khi viết final | Modern SSL + lightweight back-end representative | Front-end vẫn là bottleneck; ASV5/codec khó vẫn gây EER cao |

*Lưu ý metric: EER đo CM độc lập và là metric chính trong reproduce. Min t-DCF/t-DCF phụ thuộc ASV score và cost model, không so trực tiếp với EER. ASVspoof 5 Track 1 dùng minDCF làm primary challenge metric, còn project hiện dùng EER để so sánh nhất quán giữa bốn dataset.*

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
