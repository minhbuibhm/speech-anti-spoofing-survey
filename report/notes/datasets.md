# Notes: Datasets cho Speech Deepfake Detection

Mục đích: nơi lưu kết quả research chi tiết về các dataset SDD. Sẽ chắt lọc vào §2.2 của `report.md`.

---

## 1. Phân loại theo dạng cây

Các dataset SDD hiện có thể được tổ chức theo nhiều trục phân loại khác nhau. Dưới đây là cấu trúc cây từ high-level đến low-level, dựa trên Li et al. (2024) và các survey liên quan.

### 1.1 Theo kịch bản tấn công

```
Tấn công giọng nói giả mạo
├── Logical Access (LA) — giả mạo số, tiêm thẳng vào pipeline
│   ├── Text-to-Speech (TTS)
│   │   ├── Statistical parametric (HMM-based, GMM-based)
│   │   └── Neural TTS (Tacotron, FastSpeech, VITS, codec-based)
│   └── Voice Conversion (VC)
│       ├── GMM-based VC
│       └── Neural VC (kNN-VC, FreeVC, RVC)
└── Physical Access (PA / Replay) — phát lại qua loa/mic
    ├── Replay đơn giản (playback qua thiết bị)
    └── Replay qua các channel đặc trưng (phòng dội, loa chất lượng thấp)
```

**Ưu điểm khi dùng LA cho evaluation:** Loại trừ biến số phòng thu, dễ kiểm soát điều kiện thực nghiệm, benchmark rõ ràng về attack type. ASVspoof 2019/2021 và ASVspoof 5 đều thuộc nhóm này (với component DF/LA riêng).

**Hạn chế của LA-only:** Không phản ánh đầy đủ kịch bản thực tế nơi attacker phát lại qua thiết bị hoặc qua kênh thoại có thêm nhiễu môi trường.

**Ưu điểm của PA/Replay:** Phản ánh tấn công "thô sơ" nhưng thực tế cao — không cần kỹ năng lập trình. ASVspoof 2019 PA là benchmark chuẩn cho replay detection.

**Hạn chế PA:** Mô hình PA không tổng quát sang LA và ngược lại; evaluation chéo hai kịch bản thường không có ý nghĩa thực tiễn.

---

### 1.2 Theo điều kiện thu âm / nguồn gốc dữ liệu

```
Theo điều kiện thu âm
├── Clean lab-recorded
│   ├── Được kiểm soát về phòng thu, mic, điều kiện âm thanh
│   ├── Ví dụ: ASVspoof 2019 LA/PA, WaveFake, FoR
│   └── Đặc điểm: EER thấp, nhưng domain gap lớn so với thực tế
└── In-the-wild (scraped từ internet / thực tế)
    ├── Scraping từ mạng xã hội, YouTube, podcast
    ├── Ví dụ: In-the-Wild (Müller et al., 2022), EchoFake (replay practical), MLAAD (nhiều phần)
    └── Đặc điểm: Điều kiện đa dạng, codec đa dạng, khó kiểm soát metadata
```

**Ưu điểm clean lab:** Metadata đầy đủ, ground truth chính xác, cho phép phân tích chi tiết từng attack type. Phù hợp cho nghiên cứu cơ bản và so sánh kiến trúc.

**Hạn chế clean lab:** Mô hình train trên lab data thường suy giảm EER mạnh khi deploy thực tế do domain shift. Ví dụ trong dự án này: AASIST đạt 4.6% EER trên ASVspoof 2019 LA nhưng lên 41.8% trên In-the-Wild.

**Ưu điểm in-the-wild:** Phản ánh thực tế deployment, bao gồm tất cả biến số codec/noise/channel. Là "stress test" cho khả năng tổng quát hóa.

**Hạn chế in-the-wild:** Ground truth đôi khi không hoàn toàn chắc chắn, metadata về attack type thường thiếu hoặc không đồng nhất.

---

### 1.3 Theo số lượng và tính chất attack

```
Theo tính chất attack
├── Single-attack (1 loại TTS hoặc VC)
│   └── Ví dụ: Các thí nghiệm nghiên cứu đơn lẻ (ít dataset benchmark)
├── Multi-attack (nhiều TTS + VC systems)
│   ├── ASVspoof 2019 LA: 19 attacks (A01–A19)
│   ├── ASVspoof 2021 DF: kế thừa ASV2019 + lossy codec processing
│   ├── ASVspoof 5 (2024/2025): database 32 attack algorithms; Track 1 eval có 16 attacks
│   ├── WaveFake: 6 GAN-based TTS architectures
│   ├── MLAAD v9: 140 TTS models, 51 ngôn ngữ
│   ├── CodecFake/CodecFake+: neural codec / CoSG attacks
│   └── ADD 2022/2023: low-quality, partially fake, fake game, algorithm recognition
├── Adversarial attacks
│   ├── ASVspoof 5 (2024): bao gồm adversarial perturbation track
│   └── Đặc điểm: Attacker tối ưu hóa để qua mặt mô hình phòng thủ cụ thể
└── Partial / localized manipulation
    ├── PartialSpoof: một utterance trộn bonafide + spoofed segments
    └── ADD 2022/2023 Track RL/PF: phát hiện hoặc định vị vùng bị sửa
```

**Ưu điểm multi-attack:** Đánh giá được khả năng tổng quát hóa sang các TTS/VC system mới chưa thấy trong train. Là tiêu chuẩn của ASVspoof challenges từ 2019 trở đi.

**Hạn chế multi-attack không có adversarial:** Vẫn có thể bỏ qua các tấn công được thiết kế đặc biệt để bypass detector. Adversarial robustness là thách thức riêng.

**Ưu điểm adversarial track:** Gần với mô hình đe dọa thực tế hơn, buộc mô hình phải robust với perturbation cố ý.

**Hạn chế adversarial benchmark:** Việc chuẩn hóa "threat model" cho adversarial track vẫn là vấn đề mở; kết quả phụ thuộc nhiều vào detector target được chọn khi tạo adversarial examples.

---

### 1.4 Theo ngôn ngữ

```
Theo ngôn ngữ
├── English-centric / controlled English
│   ├── ASVspoof 2019/2021 (tiếng Anh — VCTK/VCC-style sources)
│   ├── ASVspoof 5 (MLS English partition, crowdsourced recording conditions)
│   ├── In-the-Wild (đa phần tiếng Anh)
│   └── FoR (tiếng Anh)
├── Bilingual / limited multilingual
│   └── WaveFake (tiếng Anh + tiếng Nhật)
└── Multi-lingual
    ├── MLAAD (Multi-Language Audio Anti-Spoofing Dataset)
    │   ├── v9: 140 TTS models, 78 architectures, 51 languages
    │   └── Phù hợp cho nghiên cứu cross-lingual generalization
```

**Ưu điểm multi-lingual:** Cho phép kiểm tra xem mô hình học đặc trưng ngôn ngữ-độc-lập hay phụ thuộc ngôn ngữ. Quan trọng cho deployment toàn cầu.

**Hạn chế mono-lingual benchmark:** Hiệu năng cao trên tiếng Anh có thể không đại diện cho các ngôn ngữ khác, đặc biệt các ngôn ngữ thanh điệu hoặc ít tài nguyên.

---

## 2. Bảng tổng hợp các dataset chính

| Dataset | Năm | Ngôn ngữ | Kích thước (ước tính) | #Bonafide | #Spoof | #Attack types | Đặc điểm chính | Hạn chế |
|---------|-----|----------|----------------------|-----------|--------|---------------|----------------|---------|
| ASVspoof 2019 LA | 2019 | Tiếng Anh | project dùng eval split 71,237 utterances | 7,355 (eval) | 63,882 (eval) | 19 attacks (A01–A19, TTS+VC) | Clean benchmark trên nền VCTK; train/dev/eval attack-disjoint; metadata attack rõ | Không có codec/post-processing phức tạp; attack generation cũ hơn neural codec/diffusion hiện nay |
| ASVspoof 2021 DF | 2021 | Tiếng Anh | project dùng eval subset 458,868 utterances | 16,977 | 441,891 | TTS/VC + codec/compression; official DF release có metadata riêng | Speech deepfake task không gắn ASV; audio được xử lý qua nhiều codec/bitrate để kiểm tra robustness | Cần ghi rõ project subset; không đồng nhất mặc định với mọi official/full protocol |
| ASVspoof 5 Track 1 | 2024/2025 | Tiếng Anh (MLS English partition) | project dùng Track 1 eval 680,774 utterances | 138,688 (eval) | 542,086 (eval) | Database có 32 attack algorithms; Track 1 eval có 16 attacks | Crowdsourced speech từ nhiều speaker/recording condition; legacy + contemporary TTS/VC; surrogate/adversarial setting; metadata codec/encoding | Dataset lớn; cần phân biệt Track 1 CM và Track 2 SASV; không nên gọi là multilingual trong setup này |
| In-the-Wild | 2022 | Chủ yếu tiếng Anh | 31,779 utterances | 19,963 | 11,816 | Không rõ chi tiết từng generator (scraped/public-source deepfakes) | 58 public figures; 20.8h bonafide và 17.2h spoofed audio; điều kiện nguồn/codec/channel thực tế | Có thể có label/source noise; thiếu metadata attack type để phân tích theo mechanism |
| MLAAD v9 | 2024/2026 | Đa ngôn ngữ (51) | 678.3h synthetic voice theo arXiv v9 | — | — | 140 TTS models / 78 architectures | Dataset multi-lingual mạnh nhất để kiểm tra cross-lingual generalization | Chủ yếu TTS; cần ghép bonafide tương ứng để tạo balanced setup |
| FoR (Fake-or-Real) | 2019 | Tiếng Anh | ~198K utterances | ~87K | ~111K | 7 TTS systems | Cân bằng bonafide/spoof, đơn giản | Attack type cũ, không còn đại diện cho các tấn công hiện nay |
| WaveFake | 2021 | Tiếng Anh + Nhật | ~104K utterances | ~27K | ~77K | 6 GAN-based vocoders | Tập trung vocoder artifacts | Chỉ TTS (không có VC), attack type giới hạn |
| CodecFake | 2024 | Anh/Trung (tuỳ split) | 707K+ rows trên HF mirror | — | — | Codec-based TTS/CoSG | Dataset đầu tiên nhắm vào codec-based deepfake audio; cho thấy detector train trên dataset truyền thống kém với CoSG | Rất chuyên biệt cho neural codec; chưa phải challenge canonical |
| CodecFake+ | 2025 | Nhiều nguồn | training qua 31 codec models, eval từ 17 CoSG models | — | — | 31 codec / 17 CoSG | Mở rộng lớn cho neural codec taxonomy và CodecFake detection | Work-in-progress, chi phí lớn, chưa cần cho Thực tập 1 |
| EchoFake | 2025 | Tiếng Anh | >120h audio, >13K speakers | — | — | Zero-shot TTS + physical replay | Replay-aware practical SDD, kết hợp synthetic + replay dưới device/environment đa dạng | Preprint mới; lệch khỏi LA-only evaluation hiện tại |
| ADD 2022/2023 | 2022/2023 | Mandarin-heavy | nhiều track challenge | — | — | Low-quality, partial fake, fake game, algorithm recognition | Mở rộng task ngoài binary utterance-level detection | Lệch scope report hiện tại; protocol theo challenge riêng |
| PartialSpoof | 2021/2022 | Anh | dựa trên ASVspoof 2019 LA | — | — | Partially spoofed segments | Benchmark điển hình cho partial/segment-level spoof detection | Cần segment-level labels/metrics, không khớp pipeline EER utterance-level hiện tại |

*Ghi chú: Các số liệu kích thước ghi "ước tính" là từ paper gốc hoặc repo; số liệu chính xác cần verify từ tài liệu tương ứng.*

---

## 3. Mô tả chi tiết từng dataset

### 3.1 ASVspoof 2019 LA

ASVspoof 2019 LA (Logical Access) được giới thiệu trong khuôn khổ ASVspoof Challenge 2019 (Wang et al., 2020) và nhanh chóng trở thành benchmark chuẩn cho nghiên cứu SDD. Dataset được xây dựng trên VCTK corpus với giọng nói tiếng Anh; các spoofed utterances được tạo bằng nhiều hệ thống TTS và VC. Cần phân biệt rõ split: 71,237 utterance là **eval split** được dùng trong project, gồm 7,355 bonafide và 63,882 spoof, không phải toàn bộ LA dataset. Official LA protocol có train/dev/eval tương ứng 25,380 / 24,844 / 71,237 utterance; trong đó train gồm 2,580 bonafide / 22,800 spoof và dev gồm 2,548 bonafide / 22,296 spoof. Eval split chứa 19 attack IDs (A01–A19), với metadata attack rõ để phân tích theo TTS/VC system.

Điểm mạnh lớn nhất của ASVspoof 2019 LA là điều kiện thu âm sạch, protocol rõ, attack metadata đầy đủ và cách tính metric chuẩn hoá. Vì vậy nó phù hợp làm benchmark baseline và điểm xuất phát để đo generalization gap. Hạn chế cũng đến từ chính thiết kế này: dữ liệu không có codec/post-processing phức tạp, không phản ánh nhiều điều kiện channel thực tế, và các attack generation system chủ yếu thuộc thế hệ trước neural codec/diffusion hiện đại. Do đó EER thấp trên ASVspoof 2019 LA chỉ chứng minh mô hình hoạt động tốt trên clean benchmark, chưa đủ để kết luận robustness ngoài domain.

### 3.2 ASVspoof 2021 DF

ASVspoof 2021 DF (Deepfake) được thiết kế cho speech deepfake detection không gắn trực tiếp với ASV enrollment/trial như LA/PA. Mục tiêu chính là kiểm tra khả năng phát hiện deepfake speech trong điều kiện audio đã qua codec/compression, tương tự audio được lưu trữ hoặc lan truyền qua media/news/social platforms. Official ASVspoof 2021 phát hành riêng DF speech data, evaluation keys và metadata; project hiện tại dùng một eval subset/protocol gồm 458,868 utterance, trong đó 16,977 bonafide và 441,891 spoof. Vì vậy trong report cần ghi rõ đây là **subset được project đánh giá**, không dùng con số này như mô tả tuyệt đối cho mọi official/full release.

Về threat model, ASVspoof 2021 DF giữ trọng tâm vào TTS/VC deepfake nhưng thêm biến thiên codec/bitrate, làm mờ hoặc phá vỡ các spectral artifacts mà detector học được từ clean ASVspoof 2019 LA. Dataset này vì vậy là stress test trực tiếp cho codec robustness. Trong project, EER của LFCC+LCNN và AASIST tăng mạnh khi chuyển từ ASVspoof 2019 LA sang ASVspoof 2021 DF, trong khi XLS-R+Nes2Net tăng nhẹ hơn. Hạn chế chính là cần cẩn thận khi align metadata/utt_id giữa các nguồn và khi so sánh số lượng mẫu, vì official release, keys và project subset có thể không cùng phạm vi.

### 3.3 ASVspoof 5 (2024)

ASVspoof 5 (Wang et al., 2024/2025) là benchmark hiện đại hơn các phiên bản trước nhờ sử dụng dữ liệu crowdsourced từ **MLS English partition**, với số speaker lớn hơn và điều kiện thu đa dạng hơn studio-style corpora. Vì nguồn dùng trong ASVspoof 5 là English partition, không nên mô tả dataset reproduce này là multi-lingual, dù MLS bản đầy đủ là Multilingual LibriSpeech. Database có 32 attack algorithms trên toàn bộ thiết kế, bao gồm legacy và contemporary TTS/VC, các attack được tối ưu ở mức độ khác nhau bằng surrogate detection models, và adversarial attacks được đưa vào lần đầu trong chuỗi ASVspoof.

Trong project, phần được đánh giá là **Track 1 eval** cho stand-alone countermeasure/deepfake detection, gồm 680,774 utterance với 138,688 bonafide và 542,086 spoof. Track 1 eval chứa 16 attacks; do đó không nên viết rằng eval split có 32 attacks. Theo thống kê challenge, train/dev dùng 8 attacks mỗi split, còn eval Track 1/Track 2 dùng 16 attacks. Track 2 SASV có cấu trúc trial và metric khác, không nằm trong pipeline hiện tại nếu chỉ dùng CM scores và labels. ASVspoof 5 là trọng tâm error analysis vì metadata codec/encoding và attack cho phép phân tích failure mode hiện đại; tuy nhiên kết luận trong Thực tập 1 chỉ nên hiểu là preliminary diagnosis trên sáu checkpoint hiện có, chưa đại diện đầy đủ cho Track 2 hoặc mọi operating metric của challenge.

### 3.4 In-the-Wild (Müller et al., 2022)

In-the-Wild dataset (Müller et al., 2022) được thu thập từ các nguồn công khai như social networks và video streaming platforms, với mục tiêu phản ánh phân bố audio thực tế mà detector có thể gặp khi triển khai. Official dataset gồm 58 public figures, 31,779 utterance, trong đó 19,963 bonafide và 11,816 spoof. Tính theo thời lượng, dataset có 20.8 giờ bonafide và 17.2 giờ spoofed audio, trung bình khoảng 23 phút bonafide và 18 phút spoof cho mỗi speaker.

In-the-Wild khác ASVspoof ở chỗ nó không kiểm soát chặt attack generator và không cung cấp attack metadata chi tiết cho từng mẫu. Điểm mạnh là phản ánh real-world condition: codec/channel/noise không đồng nhất, nguồn audio đa dạng, speaker distribution khác training data và có khả năng chứa post-processing từ nền tảng công khai. Điểm yếu cũng nằm ở đây: ground truth có thể chịu ảnh hưởng của quá trình xác minh thủ công hoặc nguồn công khai, và thiếu metadata khiến khó phân tích lỗi theo attack mechanism. Trong project, dataset này đóng vai trò probe cho cross-domain generalization, không phải benchmark để kết luận chi tiết về từng loại TTS/VC.

### 3.5 MLAAD (Multi-Language Audio Anti-Spoofing Dataset)

MLAAD (Müller et al., 2024; arXiv v9 cập nhật 2026) được thiết kế để giải quyết vấn đề mono-lingual của nhiều benchmark SDD. Bản v9 công bố 140 TTS models, 78 architectures và 678.3 giờ synthetic voice trên 51 ngôn ngữ. MLAAD phù hợp để kiểm tra xem detector train trên tiếng Anh hoặc tiếng Trung có thể tổng quát hóa sang ngôn ngữ khác không — câu hỏi quan trọng cho deployment ở các ngôn ngữ như tiếng Việt. Hạn chế chính là phạm vi thiên về TTS; nếu dùng cho evaluation cần ghép với bonafide data tương ứng và thiết kế split cẩn thận để tránh bias theo nguồn dữ liệu.

### 3.6 FoR (Fake-or-Real)

FoR dataset (Reimao và Tzerpos, 2019) là một trong những dataset SDD đầu tiên được xây dựng có chủ đích cho bài toán phân biệt real/fake speech. Dataset bao gồm khoảng 198K utterance từ 7 TTS system, với tỉ lệ bonafide/spoof khá cân bằng (87K/111K). Bonafide audio lấy từ nhiều nguồn công cộng đa dạng (LibriSpeech, VCTK, v.v.), giúp cải thiện diversity của giọng nói thật. Điểm yếu của FoR là các TTS system được sử dụng (Google WaveNet-based, Festival, v.v.) đã cũ so với các hệ TTS/VC/neural codec gần đây, và không có codec processing hay điều kiện thực tế. FoR hiện chủ yếu được dùng như quick sanity check hoặc để thử nghiệm trong paper, ít được dùng làm main benchmark trong nghiên cứu mới.

### 3.7 WaveFake

WaveFake (Frank và Schönherr, 2021) tập trung vào một phân khúc cụ thể: phát hiện giọng nói được tạo bởi các GAN-based neural vocoder (MelGAN, HiFi-GAN, Multi-Band MelGAN, v.v.). Dataset gồm khoảng 104K utterance cho tiếng Anh (dựa trên LJ Speech) và tiếng Nhật (dựa trên JSUT), với 6 vocoder architecture khác nhau. Đây là dataset nhỏ gọn, thường dùng trong các nghiên cứu về neural vocoder artifact detection. Tuy nhiên phạm vi của WaveFake khá hẹp — chỉ có TTS/vocoder, không có VC, và không bao gồm các kiến trúc mới hơn như codec-based TTS hay diffusion-based vocoder — nên vai trò của nó trong benchmark tổng quát ngày càng hạn chế.

### 3.8 EchoFake (2025)

EchoFake (Zhang et al., 2025) là dataset replay-aware nhắm đến tình huống thực tế: audio synthetic từ zero-shot TTS được phát lại qua thiết bị vật lý và thu trong nhiều cấu hình device/environment. Paper công bố hơn 120 giờ audio từ hơn 13,000 speakers và chỉ ra detector train trên dataset synthetic sạch có thể suy giảm mạnh khi gặp replayed audio. EchoFake phù hợp cho Thực tập 2 nếu câu hỏi nghiên cứu mở rộng từ LA/DF sang practical deployment; trong Thực tập 1 chưa đưa vào reproduce vì pipeline hiện tại tập trung utterance-level LA/DF và chưa xử lý PA/replay metadata.

### 3.9 CodecFake và CodecFake+

CodecFake (Wu et al., 2024) và CodecFake+ (Chen et al., 2025) nhắm đến một khoảng trống mới: deepfake speech từ codec-based speech generation (CoSG), nơi audio được sinh từ discrete neural codec tokens thay vì pipeline vocoder truyền thống. CodecFake chứng minh các detector train trên dataset phổ biến như ASVspoof/WaveFake có thể kém hiệu quả với CoSG. CodecFake+ mở rộng đáng kể bằng cách dùng 31 open-source neural codec models để tạo training data và dùng web-sourced samples từ 17 CoSG models cho evaluation, đồng thời đề xuất taxonomy theo vector quantizer, auxiliary objectives và decoder types. Đây là nhóm dataset nên nhắc trong literature review vì liên quan trực tiếp đến "modern neural codec attacks"; chưa reproduce trong Thực tập 1 vì chi phí dữ liệu lớn và vì ASVspoof 5 đã đóng vai trò benchmark hiện đại chính trong pipeline hiện tại.

### 3.10 ADD 2022/2023

ADD 2022 và ADD 2023 là chuỗi Audio Deepfake Detection Challenge tập trung vào các tình huống ngoài binary clean LA: low-quality fake audio, partially fake audio, fake game, manipulation region localization và deepfake algorithm recognition. Chúng quan trọng vì mở rộng bài toán từ "utterance này real/fake?" sang "đoạn nào bị chỉnh sửa?" hoặc "nguồn sinh nào tạo ra audio?". Tuy nhiên, nhiều track có protocol/metric riêng và dữ liệu Mandarin-heavy, nên không phù hợp để đưa vào reproduce 6 model x 4 dataset của báo cáo hiện tại.

### 3.11 PartialSpoof

PartialSpoof (Zhang et al., 2021/2022) được xây dựng từ ASVspoof 2019 LA để nghiên cứu utterance chỉ bị giả mạo một phần. Dataset này có nhãn segment-level, phù hợp cho bài toán localization hoặc multi-task utterance/segment detection. Nó nên được nhắc trong taxonomy vì phản ánh threat model thực tế hơn full-utterance spoof, nhưng không đưa vào benchmark chính vì code hiện tại chỉ tính EER utterance-level và không có module phát hiện vị trí đoạn giả.

---

## 4. Nhóm thách thức nổi bật từ góc độ dữ liệu

### 4.1 Codec/Compression Robustness

**Dataset đại diện:** ASVspoof 2021 DF, ASVspoof 5 (2024)

Nhiều mô hình SDD, đặc biệt là các mô hình sử dụng đặc trưng phổ hand-crafted (LFCC, CQCC) hoặc học artifact trực tiếp từ waveform gốc (AASIST), phụ thuộc vào các dấu hiệu phổ vi mô đặc trưng cho quá trình synthesis. Các codec lossy phổ biến như MP3, AAC, hay OPUS khi nén audio ở bitrate thấp đến trung bình sẽ loại bỏ các thành phần tần số cao và làm mờ các artifact này. ASVspoof 2021 DF được thiết kế đặc biệt để kiểm tra điều này bằng cách re-encode toàn bộ audio qua hơn 100 cấu hình codec. Kết quả trong dự án này xác nhận: mô hình AASIST bị tăng EER từ 4.6% lên 17.7% (+13 pp) khi chuyển từ ASVspoof 2019 LA sang ASVspoof 2021 DF. Trong khi đó, XLS-R+Nes2Net — với SSL front-end được pretrained trên hàng nghìn giờ audio đa dạng — chỉ tăng từ 0.45% lên 2.93% (+2.5 pp), cho thấy SSL representation có khả năng trừu tượng hóa vượt qua codec artifact tốt hơn đáng kể. Đây là thách thức dữ liệu thực sự vì trong môi trường deployment thực tế, hầu như tất cả audio đều đã qua ít nhất một tầng codec.

### 4.2 In-the-Wild Domain Shift

**Dataset đại diện:** In-the-Wild (Müller et al., 2022)

Ngay cả khi đã xử lý codec robustness, các mô hình vẫn gặp vấn đề nghiêm trọng với dữ liệu "thực sự trong tự nhiên" — tức là audio được scraping từ internet với đầy đủ các yếu tố: codec đa dạng, chất lượng mic khác nhau, nhiễu nền, kênh truyền, và quan trọng nhất là **phân bố speaker và attack type hoàn toàn khác** với các tập train được dùng. In-the-Wild là dataset đại diện rõ cho thách thức này. Kết quả dự án cho thấy tất cả các mô hình đều bị degradation mạnh: AASIST từ 4.6% lên 41.8%, LFCC+LCNN từ 19.6% lên 70.2%. Mô hình có EER thấp nhất trong thí nghiệm hiện tại, XLS-R+Nes2Net, cũng tăng từ 0.45% lên 5.6%. Thách thức này khó giải quyết đơn thuần bằng augmentation vì không biết trước phân bố của "wild audio"; cần các phương pháp như domain adaptation, test-time adaptation, hoặc contrastive learning để học representation domain-agnostic.

### 4.3 Adversarial Attacks và Neural Codec Attacks

**Dataset đại diện:** ASVspoof 5 Track 2 (2024), các benchmark adversarial robustness

Thế hệ tấn công mới nhất trong ASVspoof 5 bao gồm các hệ thống TTS/VC dựa trên neural codec (EnCodec, Vocos, SNAC) và adversarial perturbation được tối ưu để bypass các detector cụ thể. Đây là thách thức về "arms race" giữa detector và attacker: attacker có thể sử dụng chính detector làm oracle để tối ưu adversarial example, buộc detector phải phát hiện các tín hiệu bị che giấu chủ ý. Neural codec TTS hiện đại (như VALL-E, VoiceBox, hay các mô hình dựa trên EnCodec) tạo ra audio chất lượng cao hơn đáng kể so với các TTS trong ASVspoof 2019, đồng thời artifact ở domain codec token khác biệt hoàn toàn so với artifact phổ truyền thống. Các mô hình được huấn luyện thuần trên ASVspoof 2019 thường không nhận dạng được tấn công từ ASVspoof 5 (ví dụ AASIST: 35.75% EER trên ASV5 eval trong project này). Dataset ASVspoof 5 là bước đầu chuẩn hóa benchmark cho cả hai loại thách thức này, nhưng phương pháp đánh giá adversarial robustness một cách nhất quán vẫn là vấn đề mở trong cộng đồng nghiên cứu.

---

## 5. Tham chiếu

1. **Li, Z., Yi, J., Wang, X., & Zhao, H.** (2024). *A Survey on Speech Deepfake Detection*. arXiv:2408.10284. — Survey tổng quan nhất tính đến 2024, phân loại dataset, kiến trúc, và challenge theo hệ thống.

2. **Wang, X., Yamagishi, J., et al.** (2020). *ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech*. Computer Speech & Language, 64, 101114. — Paper mô tả dataset ASVspoof 2019 LA/PA.

3. **Yamagishi, J., Wang, X., et al.** (2021). *ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild*. Proceedings of ASVspoof 2021 Workshop. — Paper mô tả ASVspoof 2021 LA/DF/PA và rationale cho codec processing.

4. **Wang, X., et al.** (2024). *ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale*. arXiv:2408.08739. — Paper chính thức của ASVspoof 5 (2024), mô tả 32 attacks, adversarial track, và multi-codec setup.

5. **Müller, N. M., Czempin, P., Dieckmann, F., Hummel, A., & Böttinger, K.** (2022). *Does Audio Deepfake Detection Generalize?* Interspeech 2022, 2783–2787. — Paper giới thiệu In-the-Wild dataset, đánh giá cross-dataset generalization của nhiều mô hình.

6. **Yi, J., Tao, J., Fu, R., Yan, X., Wang, C., Wang, T., et al.** (2023). *ADD 2023: The Second Audio Deepfake Detection Challenge*. arXiv:2305.13774. — Bối cảnh thêm cho audio deepfake challenge, localization và algorithm recognition.

7. **Frank, J., & Schönherr, L.** (2021). *WaveFake: A Data Set to Facilitate Audio Deepfake Detection*. Proceedings of NeurIPS 2021 Track on Datasets and Benchmarks. — Paper giới thiệu WaveFake, dataset tập trung GAN-based vocoder.

8. **Reimao, R., & Tzerpos, V.** (2019). *For: A Dataset for Synthetic Speech Detection*. Proceedings of ICDSP 2019. — Paper giới thiệu Fake-or-Real (FoR) dataset.

9. **Müller, N. M., et al.** (2024/2026). *MLAAD: The Multi-Language Audio Anti-Spoofing Dataset*. arXiv:2401.09512. — Dataset đa ngôn ngữ; bản v9 công bố 140 TTS models, 78 architectures, 51 languages.

10. **Yi, J., et al.** (2022). *ADD 2022: The First Audio Deep Synthesis Detection Challenge*. arXiv:2202.08433. — Challenge low-quality fake, partially fake và fake game.

11. **Yi, J., et al.** (2023). *ADD 2023: The Second Audio Deepfake Detection Challenge*. arXiv:2305.13774. — Challenge mở rộng sang localization và deepfake algorithm recognition.

12. **Zhang, L., Wang, X., Cooper, E., Yamagishi, J., Patino, J., & Evans, N.** (2021/2022). *PartialSpoof Database / The PartialSpoof Database and Countermeasures*. — Dataset partially-spoofed audio với utterance- và segment-level labels.

13. **Wu, H., Tseng, Y., & Lee, H.-y.** (2024). *CodecFake: Enhancing Anti-Spoofing Models Against Deepfake Audios from Codec-Based Speech Synthesis Systems*. arXiv:2406.07237. — Dataset đầu tiên cho codec-based deepfake audio.

14. **Chen, X., Du, J., Wu, H., et al.** (2025). *CodecFake+: A Large-Scale Neural Audio Codec-Based Deepfake Speech Dataset*. arXiv:2501.08238. — Mở rộng CodecFake với 31 codec models và 17 CoSG models.

15. **Zhang, T., Huang, Y., & Ren, Y.** (2025). *EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection*. arXiv:2510.19414. — Dataset replay-aware cho practical SDD.

16. **Wang, X., et al.** (2025). *ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech*. arXiv:2502.08857. — Paper thiết kế database ASVspoof 5.

17. **Wang, X., Delgado, H., Evans, N., et al.** (2026). *ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech*. arXiv:2601.03944. — Paper phân tích kết quả challenge, calibration và degradation dưới adversarial/neural encoding.

---

*Ghi chú: Một số số liệu kích thước dataset được lấy từ paper gốc hoặc repo công bố; những giá trị được ghi "ước tính" cần được verify từ tài liệu chính thức trước khi đưa vào báo cáo cuối. Số liệu cụ thể từ dự án này (EER, số utterance trong eval) phản ánh kết quả reproduce thực tế trên Kaggle.*
