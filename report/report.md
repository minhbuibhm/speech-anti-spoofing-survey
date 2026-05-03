# Báo cáo Đồ án Internship 1
## Khảo sát và đánh giá các phương pháp phát hiện giọng nói giả mạo (Speech Deepfake Detection)

---

## Lời cam đoan

*[Phần này sẽ được tác giả tự điền sau.]*

---

## Lời ngỏ

*[Phần này sẽ được tác giả tự điền sau.]*

---

## Tóm tắt nội dung

*[Phần abstract — sẽ viết sau cùng khi 4 chương đã hoàn thiện. Dự kiến ~250 từ tóm tắt: bối cảnh, mục tiêu, phương pháp (survey + reproduce 6 models × 4 datasets), kết quả chính (bảng EER, phát hiện ban đầu từ error analysis ASV5), đóng góp và hướng tiếp theo.]*

---

## Chương 1: Giới thiệu

### 1.1 Bối cảnh và động lực

Trong khoảng một thập kỷ trở lại đây, các kỹ thuật tổng hợp giọng nói (Text-to-Speech, TTS) và chuyển đổi giọng nói (Voice Conversion, VC) đã có những bước tiến vượt bậc. Từ các mô hình tự hồi tiếp như WaveNet và Tacotron đến các kiến trúc end-to-end hiện đại như VITS, neural codec TTS, hay các mô hình dựa trên diffusion, chất lượng giọng nói tổng hợp ngày càng tiệm cận với giọng nói thật về cả độ tự nhiên lẫn độ giống người nói gốc. Nhiều hệ thống thương mại hiện nay có thể nhân bản (voice cloning) một giọng nói từ chỉ vài giây mẫu âm thanh, mở ra nhiều ứng dụng tích cực như trợ lý ảo, đọc sách nói, hay khôi phục giọng cho người mất khả năng nói.

Tuy nhiên, sự phát triển nhanh của các kỹ thuật này cũng kéo theo những rủi ro nghiêm trọng. Các vụ lừa đảo qua điện thoại bằng giọng nói giả mạo người thân, mạo danh lãnh đạo doanh nghiệp để lừa chuyển khoản (CEO fraud), hay phát tán thông tin sai lệch bằng cách giả giọng nhân vật công chúng đã được ghi nhận tại nhiều quốc gia trong những năm gần đây. Rủi ro này càng đáng quan ngại khi các công cụ voice cloning ngày càng dễ tiếp cận và chi phí thấp.

Trước thực trạng đó, bài toán phát hiện giọng nói giả mạo (Speech Deepfake Detection — SDD) trở thành một hướng nghiên cứu quan trọng. Ở dạng cơ bản nhất, bài toán được phát biểu như một tác vụ phân loại nhị phân: cho một đoạn audio đầu vào, hệ thống cần xác định đoạn đó là giọng thật (*bonafide*) hay đã được tổng hợp/chỉnh sửa (*spoof*). Đầu ra của hệ thống thường là một điểm số liên tục (score) phản ánh xác suất *bonafide*, từ đó có thể đưa ra quyết định phân loại bằng cách so sánh với một ngưỡng. Để đánh giá hệ thống, cộng đồng nghiên cứu thường sử dụng chỉ số Equal Error Rate (EER) — tỉ lệ lỗi tại điểm cân bằng giữa False Acceptance Rate và False Rejection Rate, càng thấp càng tốt.

### 1.2 Các thách thức chính

Mặc dù đã có nhiều tiến bộ, các hệ thống SDD hiện nay vẫn đối mặt với một số thách thức cốt lõi:

**Khả năng tổng quát hoá liên miền (cross-domain generalization).** Phần lớn các mô hình SOTA đạt EER rất thấp trên benchmark mà chúng được huấn luyện (ví dụ ASVspoof 2019 LA), nhưng EER tăng đáng kể khi được đánh giá trên dữ liệu có phân bố khác — chẳng hạn audio đã qua nén lossy (ASVspoof 2021 DF) hoặc được scraping từ mạng xã hội (In-the-Wild). Hiện tượng này cho thấy nhiều mô hình đang học các đặc trưng phụ thuộc vào điều kiện thu âm hoặc artifact cụ thể của tập huấn luyện, thay vì các dấu hiệu tổng quát của giọng nói giả mạo.

**Tốc độ tiến hoá của các kỹ thuật tấn công.** Các phương pháp TTS/VC mới liên tục xuất hiện, đặc biệt là các kiến trúc dựa trên neural codec, diffusion, và adversarial training. Nhiều dataset benchmark được xây dựng cách đây vài năm có thể không bao gồm các loại tấn công mới nhất, dẫn đến rủi ro mô hình "vô hình" trước các dạng tấn công mà nó chưa từng thấy. Việc đảm bảo các benchmark cập nhật kịp thời là một thách thức về mặt dữ liệu và quy trình đánh giá.

**Điều kiện thực tế (real-world conditions).** Audio trong thực tế thường đi kèm các yếu tố như nén lossy (MP3, AAC, OPUS), nhiễu nền, mic chất lượng kém, hoặc qua nhiều khâu xử lý hậu kỳ. Các yếu tố này có thể che lấp hoặc làm biến dạng các dấu vết phổ (spectral artifact) mà nhiều mô hình dựa vào để phát hiện giọng nói giả mạo. Các mô hình huấn luyện trên audio sạch trong phòng lab thường suy giảm hiệu năng đáng kể khi áp dụng cho audio thực tế.

**Mất cân bằng và đa dạng của dữ liệu.** Trong nhiều dataset, số lượng mẫu *spoof* lớn hơn nhiều so với *bonafide* (ví dụ ASVspoof 2019 LA có khoảng 7,3 nghìn *bonafide* trên 64 nghìn *spoof*), đồng thời độ đa dạng của *bonafide* về người nói, ngôn ngữ, và điều kiện thu âm thường hạn chế hơn. Điều này có thể ảnh hưởng đến khả năng học các đặc trưng tổng quát của giọng nói thật, đặc biệt khi áp dụng sang các tập dữ liệu cross-domain.

### 1.3 Hướng tiếp cận và cấu trúc báo cáo

Báo cáo này được thực hiện như bước khởi đầu trong một quy trình nghiên cứu dài hơi (Internship 1 → Internship 2 → Luận văn tốt nghiệp), tập trung vào ba hoạt động chính: (i) khảo sát cơ sở lý thuyết về dữ liệu, kiến trúc và phương pháp đánh giá; (ii) tái lập (reproduce) một số mô hình tiêu biểu trên các tập dữ liệu phổ biến để có số liệu tham chiếu thực tế; và (iii) phân tích lỗi sơ bộ (preliminary error analysis) trên một dataset trọng tâm, từ đó đưa ra đề xuất hướng tiếp cận ở mức ý niệm. Các đề xuất chi tiết về thiết kế thí nghiệm và triển khai thực tế sẽ được phát triển ở Internship 2 và Luận văn.

Cụ thể, báo cáo tái lập sáu mô hình đại diện cho các nhóm tiếp cận khác nhau (LFCC+LCNN, AASIST, AASIST-L, AASIST3, XLS-R+AASIST, XLS-R+Nes2Net) trên bốn dataset phổ biến (ASVspoof 2019 LA, ASVspoof 2021 DF, ASVspoof 5, In-the-Wild). Toàn bộ thí nghiệm được thực hiện trên nền tảng Kaggle với GPU P100/T4, sử dụng các checkpoint công bố công khai để đảm bảo tính khả lập (reproducibility). Phân tích lỗi sơ bộ được tập trung vào ASVspoof 5 — tập dữ liệu mới nhất trong chuỗi ASVspoof Challenge với 32 loại tấn công và metadata về codec phong phú — nhằm đưa ra cái nhìn cụ thể về điểm mạnh và điểm yếu của từng nhóm mô hình trong các kịch bản tấn công gần đây.

Phần còn lại của báo cáo được tổ chức như sau. **Chương 2** trình bày cơ sở lý thuyết, bao gồm phát biểu bài toán, các loại tấn công, các tập dữ liệu benchmark, các nhóm kiến trúc tiếp cận, và phương pháp đánh giá. **Chương 3** mô tả thiết lập thí nghiệm, trình bày kết quả EER tổng hợp trên sáu mô hình và bốn dataset, và phân tích lỗi sơ bộ trên ASVspoof 5. **Chương 4** tổng kết các phát hiện chính, đưa ra đề xuất hướng tiếp cận ở mức ý niệm và các bước phát triển tiếp theo trong khuôn khổ Internship 2 và Luận văn.

---

## Chương 2: Cơ sở lý thuyết

### 2.1 Bài toán và các loại tấn công

Ở dạng cơ bản, bài toán phát hiện giọng nói giả mạo được phát biểu như một tác vụ phân loại nhị phân ở mức từng đoạn audio (utterance-level): cho đầu vào là một đoạn tín hiệu âm thanh $x$, hệ thống countermeasure (CM) cần đưa ra điểm số $s = f(x) \in \mathbb{R}$ phản ánh mức độ tin cậy rằng $x$ là giọng nói thật (*bonafide*). Quyết định phân loại sau cùng được đưa ra bằng cách so sánh $s$ với một ngưỡng $\tau$: nếu $s \geq \tau$ thì kết luận *bonafide* (nhãn 1), ngược lại là *spoof* (nhãn 0). Nhãn được gán cố định theo quy ước: *bonafide* → 1, *spoof* → 0.

Một hệ thống CM tiêu biểu được tổ chức theo kiến trúc *front-end* + *back-end*. Front-end chịu trách nhiệm trích xuất biểu diễn đặc trưng từ tín hiệu thô — có thể là đặc trưng phổ thiết kế bằng tay (LFCC, CQCC), biểu diễn học từ raw waveform (RawNet2, SincConv), hoặc biểu diễn từ một mô hình self-supervised learning (SSL) đã pre-train trên lượng lớn dữ liệu (Wav2Vec2, XLS-R, WavLM). Back-end là một mạng phân loại nhận biểu diễn từ front-end để sinh điểm số *bonafide*. Đây là pipeline mặc định trong các challenge ASVspoof và là khung tham chiếu để mô tả các nhóm kiến trúc ở §2.3.

Các loại tấn công có thể được phân loại theo kịch bản truy cập vào hệ thống:

- **Logical Access (LA)** — tiêm tín hiệu giả vào pipeline ở mức số (digital injection). Hai dạng chính là **Text-to-Speech (TTS)** sinh giọng từ văn bản, từ các mô hình tham số thống kê (HMM, GMM) đến neural TTS hiện đại (Tacotron, FastSpeech, VITS, neural codec TTS như VALL-E, VoiceBox), và **Voice Conversion (VC)** chuyển giọng nguồn thành giọng đích, từ GMM-based đến neural VC (kNN-VC, FreeVC, RVC).
- **Physical Access (PA) / Replay** — phát lại bản ghi qua loa và thu lại bằng micro, không cần mô hình sinh nhưng đem theo dấu vết phòng thu và đặc tính thiết bị.
- **Adversarial attacks** — tạo perturbation nhỏ có chủ đích nhằm bypass một detector cụ thể, thường khó nhận biết bằng tai nhưng làm sai lệch đáng kể quyết định của mô hình.
- **Neural codec attacks** — xu hướng mới (2024–2026) trong đó neural codec đóng vai trò bottleneck vừa nén vừa sinh, làm dấu vết artifact dịch chuyển sang miền codec token thay vì miền phổ truyền thống.

Báo cáo này tập trung vào các tấn công thuộc nhóm Logical Access (TTS/VC) — vốn là kịch bản chính trong các benchmark được sử dụng — và đề cập tới adversarial cùng neural codec attacks ở mức nhận diện thách thức.

### 2.2 Các tập dữ liệu benchmark

Các tập dữ liệu SDD có thể được tổ chức theo nhiều trục: theo kịch bản tấn công (LA vs PA), theo điều kiện thu âm (clean lab vs in-the-wild), theo số lượng và tính chất tấn công (single-attack vs multi-attack vs adversarial), và theo ngôn ngữ (mono-lingual vs multi-lingual). Mỗi trục mang một loại thông tin đánh giá riêng: trục thu âm cho biết mức độ domain shift khi triển khai, trục tấn công cho biết khả năng tổng quát hoá sang các hệ TTS/VC mới, và trục ngôn ngữ liên quan đến triển khai ở các ngôn ngữ khác tiếng Anh.

Bảng 2.1 tổng hợp các dataset chính được sử dụng hoặc tham chiếu trong báo cáo này.

**Bảng 2.1.** Tổng hợp một số tập dữ liệu SDD tiêu biểu.

| Dataset | Năm | Ngôn ngữ | Kích thước | #Attack | Đặc điểm chính |
|---------|-----|----------|------------|---------|----------------|
| ASVspoof 2019 LA | 2019 | Anh | ~71K (eval) | 19 (TTS+VC) | Clean lab, benchmark chuẩn |
| ASVspoof 2021 DF | 2021 | Anh | ~458K (eval) | Kế thừa ASV2019 + codec | Re-encode qua >100 cấu hình codec lossy |
| ASVspoof 5 (Track 1) | 2024 | Đa ngôn ngữ | ~140K (dev), ~680K (eval) | 32 | Neural codec, diffusion, adversarial track |
| In-the-Wild | 2022 | Chủ yếu Anh | ~31.8K | Không xác định | Scraped mạng xã hội, 58 speaker |
| MLAAD | 2023 | ~23 ngôn ngữ | ~76K (ước tính) | ~23 TTS | Multi-lingual; chỉ TTS |
| WaveFake | 2021 | Anh + Nhật | ~104K | 6 GAN-based vocoder | Tập trung vocoder artifact |
| FoR (Fake-or-Real) | 2019 | Anh | ~198K | 7 TTS | TTS thế hệ cũ |
| EchoFake | 2025 | Anh | ~40K (ước tính) | Neural codec + replay | Kết hợp LA và PA |

Trong số này, bốn tập **ASVspoof 2019 LA**, **ASVspoof 2021 DF**, **ASVspoof 5** và **In-the-Wild** được lựa chọn cho phần tái lập (Chương 3), với lý do sau. *ASVspoof 2019 LA* là benchmark chuẩn, đại diện cho điều kiện sạch và dùng làm điểm khởi đầu để tham chiếu. *ASVspoof 2021 DF* tái sử dụng nguồn audio của ASVspoof 2019 nhưng được re-encode qua hơn 100 cấu hình codec lossy, cho phép cô lập ảnh hưởng của codec lên hiệu năng mô hình. *ASVspoof 5* là phiên bản mới nhất trong chuỗi ASVspoof Challenge (công bố 2024) với 32 loại tấn công bao gồm neural codec TTS, diffusion-based VC và adversarial track — phản ánh các dạng tấn công xuất hiện gần đây. *In-the-Wild* được scraping từ mạng xã hội với 58 người nói thực tế, đóng vai trò probe cho khả năng tổng quát hoá liên miền — không có mô hình nào trong báo cáo được huấn luyện trên tập này.

Các tập MLAAD, WaveFake, FoR và EchoFake không được tái lập trong phạm vi Internship 1, nhưng được liệt kê ở đây để làm rõ bức tranh dữ liệu tổng thể và để định hướng cho các bước mở rộng ở Internship 2 (đặc biệt là EchoFake — vốn kết hợp tấn công deepfake với replay vật lý — và MLAAD cho khả năng đa ngôn ngữ).

### 2.3 Các nhóm kiến trúc

Các phương pháp tiếp cận SDD có thể được tổ chức theo bốn nhóm chính, sắp xếp theo mức độ phụ thuộc vào pre-training quy mô lớn:

**(a) Đặc trưng thiết kế bằng tay + bộ phân loại nhỏ.** Đại diện điển hình là **LFCC + LCNN** (baseline ASVspoof 2019). Đặc trưng phổ tuyến tính LFCC được trích xuất bằng các bước xử lý tín hiệu cổ điển (FFT, filter bank), sau đó đưa vào Light CNN với Max Feature Map activation. Ưu điểm là số tham số rất nhỏ (~60K), huấn luyện nhanh, không yêu cầu GPU lớn — phù hợp triển khai biên (edge). Hạn chế: thường phụ thuộc vào dấu vết phổ vi mô của các hệ tấn công có trong tập huấn luyện; khi audio đi qua codec lossy, các dấu vết này có thể bị mờ, dẫn đến hiệu năng có xu hướng suy giảm.

**(b) End-to-end DNN trên raw waveform.** Đại diện là **RawNet2** và **họ AASIST** (Jung et al., ICASSP 2022). RawNet2 dùng SincConv + residual blocks + GRU. AASIST mở rộng bằng cách mô hình hoá tương quan trên cả hai miền phổ và thời gian thông qua mạng graph attention dị thể (heterogeneous graph attention). AASIST đạt EER 0.83% trên ASVspoof 2019 LA với chỉ ~297K tham số, AASIST-L là phiên bản nhỏ hơn (~85K) với EER 0.99% — một cân bằng đáng chú ý giữa kích thước và hiệu năng. Hạn chế: nhóm này không mang prior từ pre-training quy mô lớn; theo các kết quả công bố trong các challenge gần đây, hiệu năng có xu hướng suy giảm khi đánh giá trên audio đã qua codec hoặc trên dữ liệu có phân bố khác tập huấn luyện.

**(c) SSL front-end + back-end nhẹ.** Các kết quả công bố tại ASVspoof 2021 và ASVspoof 5 (2024) cho thấy nhóm này có nhiều hệ thống đạt thứ hạng cao trên các bảng leaderboard. **Front-end** là một mô hình self-supervised đã pre-train trên hàng chục đến hàng trăm nghìn giờ audio — Wav2Vec2 (Baevski et al., 2020), XLS-R 300M cross-lingual trên ~436K giờ (Babu et al., 2022), WavLM Large với mục tiêu denoising (Chen et al., 2022), HuBERT (Hsu et al., 2021). **Back-end** thường nhẹ hơn nhiều: AASIST back-end, Nes2Net-X (Liu et al., TIFS 2025), MFA, hoặc MLP đơn giản. Trong báo cáo này, ba mô hình SSL được tái lập là **AASIST3** (Wav2Vec2 + KAN bridge + AASIST, Borodin et al., 2024), **XLS-R + AASIST** (SSL-AASIST, Tak et al., Odyssey 2022), và **XLS-R + Nes2Net-X**. Ưu điểm chính của nhóm này là biểu diễn từ pre-training mang theo prior về cấu trúc âm thanh học được trên dữ liệu quy mô lớn; nhiều kết quả công bố gợi ý rằng các biểu diễn này có xu hướng tổng quát hoá tốt hơn trên các tấn công không gặp trong huấn luyện và trên audio đã qua codec, dù mức độ cải thiện cụ thể phụ thuộc vào back-end và cấu hình huấn luyện. Hạn chế: chi phí tính toán và bộ nhớ lớn — XLS-R 300M có khoảng 300 triệu tham số, đòi hỏi GPU cao cấp cho cả huấn luyện lẫn suy luận.

**(d) Foundation model và ensemble.** Hướng nghiên cứu mới (2024–2026) khám phá các mô hình nền tảng quy mô cực lớn như Whisper encoder làm front-end, multi-task learning kết hợp ASR với SDD, và codec-aware training (augment dữ liệu với nhiều codec/bitrate trong huấn luyện). Ensemble cấp điểm hoặc cấp đặc trưng cũng phổ biến trong các hệ thống top-tier ở các challenge, nhưng chi phí suy luận tăng tuyến tính theo số hệ thống con. Báo cáo này không tái lập các phương pháp thuộc nhóm (d), nhưng ghi nhận đây là hướng có tiềm năng và có thể được khảo sát ở Internship 2.

Bảng 2.2 tổng hợp đặc trưng của sáu mô hình được tái lập trong Chương 3.

**Bảng 2.2.** So sánh đặc trưng của sáu mô hình được tái lập.

| Mô hình | Front-end | Back-end | Số tham số | EER báo cáo (ASV19 LA) |
|---------|-----------|----------|-----------|------------------------|
| LFCC + LCNN | LFCC | LCNN | ~60K | ~4–5% (LCNN), tuỳ cấu hình |
| AASIST | Raw waveform | Het. Graph Attention | ~297K | 0.83% |
| AASIST-L | Raw waveform | GAT (lightweight) | ~85K | 0.99% |
| AASIST3 | Wav2Vec2 | KAN + AASIST | ~300M | minDCF 0.1414 (ASV24 open) |
| XLS-R + AASIST | XLS-R 300M | AASIST | ~300M | ~0.82% (ASV21 LA) |
| XLS-R + Nes2Net-X | XLS-R 300M | Nes2Net-X | ~300M | ~1.66% (ASV21 LA) |

*Các giá trị EER trong bảng được trích từ paper gốc tương ứng và phản ánh điều kiện huấn luyện/đánh giá đặc thù của từng paper; số liệu tái lập của báo cáo này được trình bày trong Chương 3.*

### 2.4 Phương pháp đánh giá

Chỉ số chính được sử dụng trong báo cáo là **Equal Error Rate (EER)**. Với hàm False Acceptance Rate $\mathrm{FAR}(\tau)$ và False Rejection Rate $\mathrm{FRR}(\tau)$ phụ thuộc vào ngưỡng $\tau$, EER là giá trị $\mathrm{FAR}(\tau^*) = \mathrm{FRR}(\tau^*)$ tại ngưỡng $\tau^*$ làm cho hai tỉ lệ này bằng nhau. Giá trị EER thường được biểu diễn dưới dạng phần trăm và càng thấp càng tốt. Ưu điểm của EER là không phụ thuộc vào việc chọn ngưỡng cố định, giúp so sánh công bằng giữa các hệ thống có thang điểm số khác nhau.

Bên cạnh EER, các challenge ASVspoof còn sử dụng **Minimum tandem Detection Cost Function (min t-DCF)** — chỉ số đánh giá CM khi tích hợp vào hệ thống Automatic Speaker Verification (ASV). Min t-DCF có ý nghĩa khi cần đánh giá tổng thể CM+ASV, nhưng đòi hỏi thông tin về ASV system kèm theo, điều này không có trong các kết quả tái lập của báo cáo. Vì vậy báo cáo chỉ sử dụng EER là chỉ số định lượng chính, và đề cập min t-DCF khi trích dẫn kết quả paper gốc.

**Đường cong DET (Detection Error Tradeoff)** — biểu diễn FAR theo FRR trên thang xác suất Gaussian — là công cụ trực quan chuẩn của cộng đồng SDD và speaker verification, được sử dụng để so sánh hành vi mô hình ở các ngưỡng vận hành khác nhau. **Calibration** (mức độ score phản ánh xác suất thực) là khía cạnh đáng quan tâm khi triển khai với ngưỡng cố định, nhưng nằm ngoài phạm vi của Internship 1 và chỉ được đề cập sơ bộ.

### 2.5 Các thách thức hiện tại

Tổng hợp từ các survey gần đây (Li et al., 2024; PMC Review, 2025), có thể nhận diện ba nhóm thách thức nổi bật trong nghiên cứu SDD hiện nay. Các thách thức này được trình bày ở đây ở mức cơ sở lý thuyết; số liệu thực nghiệm tương ứng từ phần tái lập của báo cáo được trình bày ở Chương 3.

**Tổng quát hoá liên miền (cross-domain generalization).** Theo Müller et al. (2022) và các báo cáo của ASVspoof 2021/2024, các mô hình đạt EER rất thấp trên tập huấn luyện thường có hiệu năng suy giảm khi đánh giá trên dữ liệu có phân bố khác (domain shift). Đây là động lực chính cho nhiều hướng nghiên cứu về học biểu diễn không phụ thuộc miền (domain-agnostic representation learning) và domain adaptation.

**Bền vững với codec và nén lossy (codec robustness).** Audio trong thực tế hầu như luôn đi qua ít nhất một tầng codec lossy (MP3, AAC, OPUS) ở các bitrate khác nhau. Các codec này có thể loại bỏ các thành phần tần số cao và làm mờ dấu vết phổ vi mô — vốn là tín hiệu mà nhiều mô hình hand-crafted hoặc end-to-end nhỏ dựa vào để phát hiện giọng nói tổng hợp. ASVspoof 2021 DF và ASVspoof 5 đều được thiết kế để kiểm tra khía cạnh này.

**Tấn công hiện đại và adversarial.** Sự xuất hiện của neural codec TTS (VALL-E, VoiceBox), diffusion-based VC, và các kỹ thuật adversarial perturbation đặt ra thách thức về một cuộc "chạy đua vũ trang" giữa attacker và detector. ASVspoof 5 là challenge đầu tiên tích hợp adversarial track ở quy mô lớn. Phương pháp đánh giá adversarial robustness một cách nhất quán (ví dụ: chuẩn hoá threat model, lựa chọn target detector) hiện vẫn là vấn đề mở trong cộng đồng nghiên cứu.

Ba nhóm thách thức trên là khung tham chiếu để diễn giải kết quả tái lập ở Chương 3 và để định hướng cho phần đề xuất ở Chương 4.

---

## Chương 3: Reproduce và phân tích kết quả

### 3.1 Thiết lập thí nghiệm

Toàn bộ thí nghiệm trong báo cáo được thực hiện trên nền tảng Kaggle với GPU P100 hoặc T4. Việc lựa chọn Kaggle làm môi trường tính toán là do nền tảng cung cấp GPU miễn phí phù hợp với quy mô tái lập (suy luận mô hình SSL 300M trên các tập eval cỡ vài chục đến vài trăm nghìn utterance), đồng thời cho phép quản lý dataset và checkpoint dưới dạng *Kaggle Datasets* công khai, hỗ trợ tính khả lập (reproducibility).

**Mã nguồn và cấu trúc.** Mỗi dataset được đánh giá bằng một notebook riêng (`notebooks/eval_asvspoof_2019.ipynb`, `notebooks/eval_asvspoof_2021.ipynb`, `notebooks/eval_asvspoof_5.ipynb`, `notebooks/eval_in_the_wild.ipynb`), được sinh tự động từ một template chung trong `scripts/create_eval_notebooks.py`. Cách tổ chức này giúp giữ logic suy luận nhất quán giữa các dataset (cùng cách load checkpoint, cùng cách tính EER) trong khi vẫn cho phép tuỳ biến phần parser metadata cho từng dataset.

**Nguồn checkpoint.** Tất cả các mô hình được sử dụng ở chế độ pretrained, không thực hiện fine-tuning thêm trong phạm vi báo cáo này. Các checkpoint được lấy từ các nguồn công khai như sau: AASIST và AASIST-L từ repository chính thức của tác giả; AASIST3 từ HuggingFace (`MTUCI/AASIST3`); XLS-R 300M từ release Fairseq; XLS-R + AASIST từ checkpoint công bố bởi nhóm tác giả SSL-AASIST (Tak et al., 2022); XLS-R + Nes2Net từ release của nhóm tác giả Nes2Net (Liu et al., 2025); LFCC+LCNN được huấn luyện lại từ đầu trên tập huấn luyện ASVspoof 2019 LA (~10 epoch) do không có checkpoint công khai phù hợp.

**Cấu trúc kết quả.** Toàn bộ điểm số đầu ra được lưu vào các file `results/<dataset>/results.pkl`, mỗi file là một `dict` Python với cấu trúc:

```python
{
    "<model_name>": {
        "eer":    float,           # EER (%)
        "scores": np.ndarray,      # (N,) — điểm bonafide
        "labels": np.ndarray,      # (N,) — 1=bonafide, 0=spoof
    },
    ...
}
```

Cấu trúc đồng nhất này cho phép các bước phân tích hậu kỳ (error analysis, score correlation, cross-dataset comparison) hoạt động trên mọi dataset mà không cần xử lý đặc thù.

**Hướng mở rộng.** Để bổ sung một mô hình mới, cần thêm hàm load checkpoint và một cell suy luận tương ứng vào `scripts/create_eval_notebooks.py` rồi sinh lại các notebook. Để bổ sung một dataset mới, cần thêm parser protocol và đường dẫn audio vào cùng file generator. Cách tổ chức này được thiết kế để giảm chi phí mở rộng ở Internship 2.

### 3.2 Kết quả EER tổng hợp

Bảng 3.1 tổng hợp EER (%) của sáu mô hình trên bốn dataset. Các giá trị được tính trên tập eval của từng dataset bằng `sklearn` từ điểm số *bonafide* và nhãn ground truth (1 = *bonafide*, 0 = *spoof*).

**Bảng 3.1.** EER (%) của sáu mô hình trên bốn dataset. Số nhỏ hơn là tốt hơn. Các ô "—" tương ứng với các đánh giá chưa hoàn thành tại thời điểm viết báo cáo (XLS-R + AASIST trên ASVspoof 2021 DF và In-the-Wild).

| Mô hình | ASV19 LA | ASV21 DF | ASV5 (Track 1) | In-the-Wild |
|---------|----------|----------|----------------|-------------|
| LFCC + LCNN | 19.64 | 33.81 | 22.60 | 70.23 |
| AASIST | 4.59 | 17.70 | 37.81 | 41.79 |
| AASIST-L | 6.74 | 19.13 | 39.47 | 45.28 |
| AASIST3 | 20.83 | 29.18 | 19.03 | 40.12 |
| XLS-R + AASIST | 1.17 | — | 2.55 | — |
| XLS-R + Nes2Net | 0.45 | 2.93 | 1.80 | 5.57 |

Một số xu hướng có thể quan sát từ Bảng 3.1:

- **Dải EER trải rộng đáng kể** giữa các mô hình trên cùng một dataset. Trên ASVspoof 2019 LA, khoảng cách giữa mô hình tốt nhất quan sát được (XLS-R + Nes2Net, 0.45%) và kém nhất (AASIST3, 20.83%) là khoảng 20 điểm phần trăm. Trên In-the-Wild, khoảng cách này rộng hơn nữa (5.57% so với 70.23%).
- **EER có xu hướng tăng khi chuyển từ ASV19 LA sang các dataset khác** đối với phần lớn các mô hình, dù mức độ tăng khác nhau. Riêng AASIST3 và LFCC+LCNN không tuân theo xu hướng đơn điệu này (xem §3.3).
- **Hai mô hình dùng XLS-R 300M làm front-end (XLS-R + AASIST, XLS-R + Nes2Net)** duy trì EER ở dải thấp hơn đáng kể trên các dataset khó (ASV21 DF, ASV5, In-the-Wild) so với các mô hình không dùng SSL front-end. Đây là quan sát chính của báo cáo và là dẫn dắt cho phần đề xuất ở Chương 4.

Cần lưu ý rằng các nhận xét trên chỉ giới hạn trong phạm vi sáu mô hình và bốn dataset được khảo sát; việc kết luận tổng quát hơn (ví dụ về tính tổng quát hoá của SSL front-end nói chung) đòi hỏi phân tích lỗi chi tiết và mở rộng tập mô hình/dataset, sẽ được thực hiện ở Internship 2.

### 3.3 Phân tích EER theo từng dataset

**ASVspoof 2019 LA (71,237 utterance — 7,355 bonafide / 63,882 spoof).** Đây là benchmark sạch nhất trong bốn dataset. Bốn mô hình đạt EER dưới 7% (XLS-R + Nes2Net 0.45, XLS-R + AASIST 1.17, AASIST 4.59, AASIST-L 6.74), trong khi LFCC+LCNN (19.64) và AASIST3 (20.83) cao hơn rõ rệt. Riêng AASIST3 cao bất ngờ trên dataset này — mô hình được công bố cho ASVspoof 2024 và có thể đã được huấn luyện theo phân bố tấn công khác với ASVspoof 2019 LA; điều này cần được xác nhận thêm trong phần error analysis. LFCC+LCNN ở mức 19.64 phù hợp với baseline thấp đã biết của nhóm hand-crafted features khi không được fine-tune kỹ.

**ASVspoof 2021 DF (458,868 utterance — 16,977 bonafide / 441,891 spoof).** Dataset này tái sử dụng nguồn audio của ASVspoof 2019 nhưng được re-encode qua nhiều cấu hình codec lossy. Các mô hình không dùng SSL có xu hướng tăng EER khi chuyển từ ASV19 LA sang ASV21 DF: AASIST từ 4.59 → 17.70 (+13.1 pp), AASIST-L từ 6.74 → 19.13 (+12.4 pp), LFCC+LCNN từ 19.64 → 33.81 (+14.2 pp), AASIST3 từ 20.83 → 29.18 (+8.3 pp). XLS-R + Nes2Net tăng nhẹ hơn (0.45 → 2.93, +2.5 pp). Quan sát này gợi ý rằng codec lossy có ảnh hưởng đáng kể tới các mô hình hand-crafted hoặc end-to-end nhỏ; tuy nhiên đây mới là quan sát ở mức tổng EER, cần error analysis theo loại codec để xác nhận giả thuyết.

**ASVspoof 5 — Track 1 (140,950 utterance — 31,334 bonafide / 109,616 spoof).** ASVspoof 5 chứa 32 loại tấn công bao gồm các hệ neural codec TTS và adversarial perturbation. Đáng chú ý, trên dataset này thứ tự các mô hình thay đổi rõ rệt so với ASV19 LA: AASIST (37.81) và AASIST-L (39.47) cao hơn LFCC+LCNN (22.60) và AASIST3 (19.03). XLS-R + AASIST (2.55) và XLS-R + Nes2Net (1.80) tiếp tục giữ EER thấp. Sự đảo thứ tự này cho thấy hành vi của mô hình trên ASV5 phụ thuộc vào loại tấn công cụ thể và có thể không tỉ lệ thuận với hiệu năng trên ASV19 LA — đây là một trong những lý do ASVspoof 5 được chọn làm dataset trọng tâm cho phần error analysis ở §3.4.

**In-the-Wild (31,779 utterance — 19,963 bonafide / 11,816 spoof).** Đây là tập probe cho khả năng tổng quát hoá liên miền: không có mô hình nào trong báo cáo được huấn luyện trên dữ liệu mạng xã hội. Tất cả các mô hình đều cho EER cao hơn so với ASV19 LA: LFCC+LCNN tăng mạnh nhất (19.64 → 70.23), trong khi XLS-R + Nes2Net giữ EER ở mức 5.57. Khoảng cách này gợi ý rằng các mô hình không dùng SSL front-end có thể đang học các đặc trưng phụ thuộc nhiều vào điều kiện huấn luyện; tuy nhiên cũng cần lưu ý In-the-Wild có ground truth đến từ xác minh thủ công và có thể chứa noise label, do đó EER cao có thể phản ánh một phần vấn đề chất lượng nhãn — vấn đề này cần được khảo sát thêm.

### 3.4 Phân tích lỗi trên ASVspoof 5 *(placeholder — sẽ cập nhật)*

Phần này sẽ được hoàn thiện sau khi (i) hoàn thành đánh giá đầy đủ trên tập eval của ASVspoof 5 (hiện báo cáo dùng tập dev với metadata phong phú hơn) và (ii) bổ sung mô hình **Nes2Net không có XLS-R** để cô lập đóng góp của SSL front-end. Khi đó các tiểu mục dự kiến gồm:

- **§3.4.1 Phân phối điểm số** — biểu đồ phân phối điểm *bonafide* theo nhãn cho từng mô hình; quan sát mức độ tách biệt giữa hai phân phối.
- **§3.4.2 Lỗi theo loại tấn công** — EER nhóm theo `attack_tag` của ASVspoof 5; xác định các tấn công nào gây lỗi nhiều nhất cho mỗi mô hình.
- **§3.4.3 Lỗi theo codec** — EER nhóm theo `codec`/`codec_q` để định lượng ảnh hưởng của codec.
- **§3.4.4 Confident errors** — các utterance bị phân loại sai với confidence cao; nếu có audio tương ứng, đính kèm spectrogram đại diện.
- **§3.4.5 Failure overlap giữa các mô hình** — các utterance bị tất cả mô hình phân loại sai (universal hard examples) so với các utterance chỉ một mô hình sai (model-specific weakness).

Các quan sát từ §3.4 sẽ là cơ sở cho phần đề xuất hướng tiếp cận ở Chương 4.

### 3.5 Nhận xét liên dataset

Bảng 3.2 trình bày *generalization gap* — chênh lệch EER giữa từng dataset khó hơn và benchmark cơ sở ASVspoof 2019 LA — như một cách đo định lượng mức độ tổng quát hoá của từng mô hình.

**Bảng 3.2.** Generalization gap (delta EER, điểm phần trăm) so với ASVspoof 2019 LA. Số dương nghĩa là EER tăng khi chuyển sang dataset khó hơn.

| Mô hình | ASV21 DF − ASV19 | ASV5 − ASV19 | ITW − ASV19 |
|---------|------------------|--------------|-------------|
| LFCC + LCNN | +14.17 | +2.96 | +50.59 |
| AASIST | +13.11 | +33.22 | +37.20 |
| AASIST-L | +12.39 | +32.73 | +38.54 |
| AASIST3 | +8.35 | −1.80 | +19.29 |
| XLS-R + AASIST | — | +1.38 | — |
| XLS-R + Nes2Net | +2.48 | +1.35 | +5.12 |

Một số quan sát:

- **Nhóm dùng XLS-R 300M làm front-end** (hai dòng cuối) có generalization gap nhỏ nhất trong phạm vi các dataset được đo. XLS-R + Nes2Net giữ gap dưới 5.5 pp trên cả ba dataset khó.
- **AASIST và AASIST-L** có gap lớn nhất trên ASV5 (+33.22 và +32.73 pp), gợi ý rằng các mô hình này có thể đặc biệt nhạy cảm với loại tấn công trong ASV5; cần error analysis ở §3.4 để xác nhận.
- **AASIST3** có gap âm trên ASV5 (−1.80 pp) — EER trên ASV5 thấp hơn trên ASV19 LA. Quan sát này nhất quán với khả năng AASIST3 được huấn luyện theo phân bố gần ASVspoof 2024 hơn ASVspoof 2019.
- **LFCC+LCNN** có gap lớn nhất trên In-the-Wild (+50.59 pp); kết hợp với khả năng có noise label trong In-the-Wild đã đề cập ở §3.3, cần thận trọng khi diễn giải con số này.

Các nhận xét trên giới hạn trong phạm vi sáu mô hình và bốn dataset được khảo sát, và phụ thuộc vào việc các checkpoint được sử dụng đều ở chế độ pretrained mà không fine-tune. Các quan sát chi tiết hơn về nguyên nhân đứng sau các generalization gap này sẽ được rút ra sau khi hoàn thành error analysis ở §3.4.

---

## Chương 4: Kết luận và Hướng phát triển

### 4.1 Tóm tắt kết quả

*[~1 trang tóm tắt: đã survey gì, reproduce được gì, các phát hiện ban đầu từ EER và error analysis sơ bộ.]*

### 4.2 Đề xuất hướng tiếp cận (mức ý niệm)

*[Dựa trên error analysis ASV5 (khi có), đề xuất 1-2 hướng cải thiện ở mức ý niệm. Mỗi hướng nêu motivation từ pattern lỗi quan sát được, kế hoạch thử nghiệm tổng quan, và kết quả kỳ vọng. Chi tiết cụ thể sẽ được trình bày trong Internship 2.]*

### 4.3 Hướng phát triển (Future work)

*[Liệt kê các bước Internship 2 và Luận văn: cụ thể hoá đề xuất thành experiment plan chi tiết, triển khai và đánh giá đề xuất, mở rộng sang dataset mới (EchoFake), thử các SSL front-end khác (WavLM, multi-lingual XLS-R).]*

---

## Tài liệu tham khảo

*[Sẽ được hoàn thiện dần khi viết các chương; format theo IEEE.]*
