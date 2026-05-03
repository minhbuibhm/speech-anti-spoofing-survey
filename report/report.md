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

**Khả năng tổng quát hoá liên miền (cross-domain generalization).** Nhiều mô hình được báo cáo hiệu năng cao đạt EER rất thấp trên benchmark mà chúng được huấn luyện (ví dụ ASVspoof 2019 LA), nhưng EER tăng đáng kể khi được đánh giá trên dữ liệu có phân bố khác — chẳng hạn audio đã qua nén lossy (ASVspoof 2021 DF) hoặc được scraping từ mạng xã hội (In-the-Wild). Hiện tượng này cho thấy nhiều mô hình đang học các đặc trưng phụ thuộc vào điều kiện thu âm hoặc artifact cụ thể của tập huấn luyện, thay vì các dấu hiệu tổng quát của giọng nói giả mạo.

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

Speech Deepfake Detection (SDD) là bài toán xây dựng một hệ thống *countermeasure* (CM) để phân biệt audio thật (*bonafide*) và audio giả mạo (*spoof*). Ở mức utterance-level, với một đoạn tín hiệu âm thanh đầu vào $x$, mô hình sinh ra một điểm số $s = f(x) \in \mathbb{R}$ thể hiện mức độ tin cậy rằng $x$ là *bonafide*. Quyết định cuối cùng được đưa ra bằng cách so sánh score với một ngưỡng $\tau$: nếu $s \geq \tau$ thì phân loại là *bonafide*, ngược lại là *spoof*. Trong toàn bộ báo cáo này, nhãn được quy ước là *bonafide* = 1 và *spoof* = 0 để thống nhất với cách tính EER ở Chương 3.

Trong bối cảnh Automatic Speaker Verification (ASV), CM thường được đặt trước hoặc song song với hệ thống xác thực người nói. ASV trả lời câu hỏi "người nói có đúng là danh tính đã khai báo không?", còn CM trả lời câu hỏi "tín hiệu đầu vào có phải là giọng thật không?". Hai bài toán liên quan nhưng không đồng nhất: một audio giả có thể bắt chước đúng speaker target và qua được ASV, nhưng vẫn cần bị CM phát hiện là spoof. Vì vậy các challenge ASVspoof đánh giá CM như một thành phần bảo vệ ASV trước các tấn công tổng hợp, chuyển đổi giọng hoặc replay [1], [2].

Một CM system điển hình gồm ba bước. **Front-end** chuyển waveform thành representation: có thể là hand-crafted spectral features như LFCC/CQCC, feature học trực tiếp từ raw waveform, hoặc representation từ self-supervised learning (SSL) model như Wav2Vec2, XLS-R, WavLM. **Back-end** nhận representation này để mô hình hoá quan hệ thời gian, phổ hoặc speaker/utterance-level cue. **Classifier/scoring head** sinh score cuối cùng cho từng utterance. Cách tách `front-end` và `back-end` này giúp so sánh các nhóm kiến trúc ở §2.3: khác biệt chính giữa LFCC+LCNN, AASIST và XLS-R+Nes2Net không chỉ nằm ở classifier, mà nằm ở loại representation mà mô hình được phép khai thác.

Về threat model, các tấn công trong SDD thường được chia thành bốn nhóm chính:

- **Logical Access (LA).** Attacker tiêm audio giả trực tiếp vào pipeline số, không cần phát lại qua loa/micro. Hai dạng phổ biến là **Text-to-Speech (TTS)** và **Voice Conversion (VC)**. TTS sinh giọng nói từ văn bản; VC giữ nội dung hoặc đặc tính ngữ âm của nguồn nhưng biến đổi speaker identity sang giọng đích. LA là kịch bản chính của ASVspoof 2019 LA, ASVspoof 2021 DF và ASVspoof 5 Track 1.
- **Physical Access (PA) / replay.** Attacker phát lại audio qua thiết bị vật lý rồi thu hoặc đưa vào hệ thống qua microphone. Khác với LA, replay chứa thêm dấu vết của loa, phòng, microphone và khoảng cách thu. Do đó một detector huấn luyện cho LA không nhất thiết generalize sang PA, và ngược lại.
- **Deepfake/in-the-wild setting.** Audio được thu thập từ social media, video streaming hoặc nguồn công khai, thường đã qua codec, noise, trimming, post-processing và có metadata không đầy đủ. Đây không phải một attack mechanism đơn lẻ, mà là một deployment-like condition dùng để kiểm tra cross-domain generalization.
- **Adversarial và neural codec attacks.** Adversarial attacks thêm perturbation có chủ đích để làm sai lệch detector. Neural codec attacks liên quan đến các TTS/VC pipeline hiện đại sử dụng codec representation như EnCodec/SNAC/Vocos, khiến artifact không còn giống các spectral artifact truyền thống trong benchmark cũ. ASVspoof 5 đưa các yếu tố này vào benchmark ở quy mô lớn hơn các phiên bản trước [3].

Báo cáo này tập trung thực nghiệm vào nhóm LA và deepfake/in-the-wild vì bốn dataset reproduce ở Chương 3 chủ yếu thuộc hai nhóm này. PA/replay và adversarial robustness được trình bày ở mức cơ sở lý thuyết để làm rõ bức tranh nghiên cứu, nhưng chưa phải trọng tâm thực nghiệm của Internship 1.

### 2.2 Các tập dữ liệu benchmark

Dataset trong SDD không chỉ khác nhau về số lượng sample, mà còn khác nhau về giả định đánh giá. Có thể phân loại theo bốn trục chính.

Thứ nhất là **attack scenario**: Logical Access (TTS/VC được đưa trực tiếp vào hệ thống) và Physical Access/replay (audio phát qua thiết bị vật lý). ASVspoof 2019 có cả LA và PA, nhưng báo cáo này dùng LA vì các mô hình reproduce chủ yếu được công bố cho LA/DF. Thứ hai là **recording condition**: clean lab-recorded data như ASVspoof 2019 LA giúp so sánh kiến trúc trong điều kiện kiểm soát, còn in-the-wild data như In-the-Wild phản ánh deployment condition với codec, noise, channel và source diversity cao hơn. Thứ ba là **attack diversity**: benchmark multi-attack như ASVspoof 2019/2021/5 kiểm tra khả năng phát hiện nhiều hệ TTS/VC khác nhau, trong khi adversarial track kiểm tra trường hợp attacker tối ưu audio để bypass detector. Thứ tư là **language coverage**: nhiều benchmark cũ tập trung tiếng Anh, còn MLAAD và ASVspoof 5 mở rộng câu hỏi sang multi-lingual generalization [3], [8].

Bảng 2.1 tổng hợp các dataset chính. Cột "Kích thước" ưu tiên ghi split hoặc subset được dùng trong báo cáo nếu dataset được reproduce; với dataset chỉ tham chiếu, giá trị được ghi theo paper gốc ở mức tổng quan.

**Bảng 2.1.** Tổng hợp một số dataset SDD tiêu biểu.

| Dataset | Năm | Kịch bản | Ngôn ngữ | Kích thước dùng/tham chiếu | Đặc điểm chính | Vai trò trong báo cáo |
|---------|-----|----------|----------|----------------------------|----------------|-----------------------|
| ASVspoof 2019 LA | 2019 | LA, TTS/VC | Anh | 71,237 utterances trong eval split dùng ở project | Clean lab, 19 attacks, metadata attack rõ | Benchmark baseline để đo hiệu năng trong điều kiện sạch |
| ASVspoof 2021 DF | 2021 | Deepfake/LA + codec | Anh | 458,868 utterances trong eval subset dùng ở project | Audio kế thừa ASV2019, re-encode qua nhiều codec/bitrate | Kiểm tra codec robustness |
| ASVspoof 5 Track 1 | 2024 | LA/deepfake | Multi-lingual | 140,950 utterances trong dev split dùng ở project; eval split lớn hơn nhưng chưa dùng đầy đủ | 32 attacks, có neural codec/diffusion-era systems; có Track 2 adversarial riêng | Dataset trọng tâm cho preliminary error analysis |
| In-the-Wild | 2022 | In-the-wild deepfake | Chủ yếu Anh | 31,779 utterances, 58 speakers trong bản dùng ở project | Scraped từ nguồn công khai, 20.8h bonafide và 17.2h spoofed audio | Probe cho cross-domain generalization |
| MLAAD | 2024 | Multi-lingual TTS | 23 ngôn ngữ | 160.2h synthetic voice theo paper; không reproduce | 52 TTS models, 22 architectures | Hướng mở rộng cho cross-lingual evaluation |
| WaveFake | 2021 | TTS/vocoder | Anh + Nhật | Khoảng 104K utterances theo paper | Tập trung GAN/neural vocoder artifacts | Dataset tham chiếu cho vocoder-specific artifacts |
| FoR | 2019 | TTS | Anh | Khoảng 198K utterances theo paper | Dataset real/fake speech đời đầu, nhiều TTS cũ | Tham chiếu lịch sử, không dùng làm benchmark chính |

**ASVspoof 2019 LA** là điểm xuất phát hợp lý vì điều kiện sạch, protocol rõ và được dùng rộng rãi trong cộng đồng ASV anti-spoofing [1]. Điểm mạnh của nó là metadata attack type đầy đủ, cho phép phân tích từng nhóm TTS/VC. Hạn chế là nhiều attack phản ánh công nghệ synthesis trước 2019 và không có codec/post-processing phức tạp. Vì vậy EER thấp trên ASVspoof 2019 LA không đủ để kết luận mô hình sẽ hoạt động tốt ngoài môi trường benchmark.

**ASVspoof 2021 DF** được thiết kế để đưa yếu tố codec và transmission condition vào đánh giá [2]. Về bản chất, dataset này dùng lại nguồn tấn công từ ASVspoof 2019 nhưng xử lý qua nhiều codec/bitrate. Thiết kế này hữu ích vì cô lập được một câu hỏi cụ thể: khi spectral artifacts bị nén hoặc làm mờ, detector còn phân biệt được bonafide/spoof không? Đây là lý do ASVspoof 2021 DF được dùng trong báo cáo như một stress test cho `codec robustness`.

**ASVspoof 5** mở rộng benchmark sang tập attack hiện đại hơn, có quy mô lớn hơn và có adversarial track riêng [3]. Trong phạm vi Internship 1, báo cáo dùng Track 1 dev split vì split này có metadata phù hợp cho phân tích lỗi sơ bộ. Việc chưa hoàn thành full eval split được ghi rõ ở §3.4; do đó các kết luận từ ASVspoof 5 trong báo cáo cần được hiểu là preliminary diagnosis, không phải đánh giá cuối cùng.

**In-the-Wild** khác với chuỗi ASVspoof ở chỗ nó không cố kiểm soát attack generator. Dataset được thu thập từ nguồn công khai cho 58 public figures, gồm 20.8 giờ bonafide và 17.2 giờ spoofed audio [4]. Điểm mạnh của nó là phản ánh real-world condition: codec không đồng nhất, chất lượng nguồn khác nhau, speaker distribution khác training data và metadata attack không đầy đủ. Điểm yếu cũng nằm ở chính điều này: khó phân tích lỗi theo attack type và có khả năng tồn tại label noise. Trong báo cáo, In-the-Wild được dùng như probe cho `domain shift`, không dùng để rút ra kết luận chi tiết về từng attack mechanism.

Các dataset như MLAAD, WaveFake và FoR giúp hoàn thiện bức tranh dữ liệu nhưng không nằm trong phạm vi reproduce của Internship 1. MLAAD đặc biệt đáng chú ý cho hướng multi-lingual vì paper công bố 23 ngôn ngữ và 52 TTS models [8]. EchoFake và các dataset kết hợp LA/PA hoặc neural codec + replay được xem là hướng theo dõi cho Internship 2, nhưng không đưa vào Bảng 2.1 khi chưa có nguồn chính thức đủ chắc trong phạm vi báo cáo hiện tại.

### 2.3 Các nhóm kiến trúc

Các kiến trúc SDD có thể được nhìn theo câu hỏi: mô hình lấy thông tin gì từ audio, và representation đó có đủ ổn định khi attack/domain thay đổi hay không.

**Hand-crafted features + classifier.** Nhóm này dùng kiến thức signal processing để thiết kế front-end trước, sau đó dùng classifier tương đối nhỏ. LFCC, CQCC, MFCC hoặc IMFCC nén thông tin phổ/thời gian thành cepstral features; back-end có thể là GMM, SVM, LCNN hoặc ResNet. Ưu điểm là rẻ, dễ triển khai và dễ giải thích: nếu một TTS/vocoder tạo ra spectral artifact lặp lại trong một vùng tần số, hand-crafted features có thể bắt được dấu vết đó. Nhược điểm là inductive bias này cũng là điểm yếu. Khi attack generator thay đổi, hoặc khi audio đi qua codec lossy làm mờ high-frequency artifacts, feature thiết kế thủ công có thể không còn giữ được cue phân biệt. LFCC+LCNN trong báo cáo đại diện cho nhóm này.

**End-to-end DNN trên raw waveform.** RawNet2 và AASIST bỏ qua bước feature engineering thủ công, học trực tiếp từ waveform [5], [6]. RawNet2 dùng SincConv/residual blocks để học filter và temporal pattern. AASIST bổ sung heterogeneous graph attention để mô hình hoá quan hệ giữa spectral branch và temporal branch; ý tưởng chính là artifact của spoofed speech có thể xuất hiện không chỉ ở một frame/tần số riêng lẻ, mà trong quan hệ spectro-temporal dài hơn. AASIST đạt kết quả công bố tốt trên ASVspoof 2019 LA với số tham số nhỏ, cho thấy raw waveform model có thể học cue hiệu quả trong benchmark sạch [6]. Tuy nhiên, vì không có large-scale pre-training, nhóm này vẫn có nguy cơ học artifact gắn với training distribution. AASIST-L giữ cùng ý tưởng nhưng giảm capacity để phục vụ lightweight setting.

**SSL front-end + lightweight back-end.** Nhóm này dùng một SSL model đã pre-train trên lượng lớn audio làm representation extractor, sau đó gắn một back-end nhỏ để phân loại spoof/bonafide. Wav2Vec2, XLS-R, WavLM và HuBERT học speech representation từ mục tiêu tự giám sát như masked prediction hoặc contrastive learning, không cần label spoof trong giai đoạn pre-training [7], [9], [10]. Lợi thế kỳ vọng là representation không chỉ chứa artifact cục bộ mà còn chứa thông tin rộng hơn về phonetic structure, speaker/channel variation và acoustic regularity. Vì vậy, khi attack hoặc codec thay đổi, SSL features có thể cung cấp nền tảng ổn định hơn cho back-end. Báo cáo reproduce ba mô hình thuộc nhóm này: AASIST3 (Wav2Vec2 + KAN bridge + AASIST), XLS-R + AASIST, và XLS-R + Nes2Net-X [11], [12], [13]. Hạn chế chính là chi phí: riêng XLS-R 300M đã có khoảng 300 triệu tham số, khiến inference và fine-tuning tốn GPU hơn nhiều so với AASIST hay LFCC+LCNN.

**Foundation model, codec-aware training và ensemble.** Các hướng mới hơn khai thác encoder lớn như Whisper, multi-task learning với ASR/speaker tasks, hoặc training có chủ đích với nhiều codec/bitrate. Ensemble cũng thường xuất hiện trong các hệ thống mạnh ở challenge vì kết hợp nhiều cue khác nhau, nhưng inference cost tăng theo số hệ thống con. Trong scope Internship 1, báo cáo chưa reproduce nhóm này; chúng được xem là hướng mở rộng sau khi error analysis xác định rõ failure mode của các baseline hiện tại.

Bảng 2.2 tổng hợp sáu mô hình được reproduce trong Chương 3. Các giá trị hiệu năng trong cột "Kết quả paper" chỉ để đặt mô hình vào bối cảnh, không dùng thay cho kết quả reproduce.

**Bảng 2.2.** So sánh sáu mô hình được tái lập.

| Mô hình | Input/front-end | Back-end | Compute cost | Điểm mạnh kỳ vọng | Failure mode cần chú ý | Kết quả paper |
|---------|-----------------|----------|--------------|-------------------|------------------------|---------------|
| LFCC + LCNN | LFCC spectral features | LCNN | Thấp | Nhẹ, dễ train, phù hợp baseline | Nhạy với codec và unseen artifacts | LFCC/LCNN variants thường là baseline trong ASVspoof |
| AASIST | Raw waveform | Spectro-temporal graph attention | Thấp-trung bình | Compact, học cue trực tiếp từ waveform | Có thể overfit vào artifact của ASV19-style attacks | 0.83% EER trên ASVspoof 2019 LA [6] |
| AASIST-L | Raw waveform | Lightweight AASIST | Thấp | Giảm tham số, phù hợp constrained setting | Capacity thấp hơn AASIST, vẫn thiếu SSL prior | 0.99% EER trên ASVspoof 2019 LA [6] |
| AASIST3 | Wav2Vec2 SSL features | KAN bridge + AASIST | Cao | Khai thác SSL representation và regularization | Phụ thuộc checkpoint/condition ASVspoof 2024; nặng | Báo cáo cho ASVspoof 2024 [11] |
| XLS-R + AASIST | XLS-R 300M | AASIST | Cao | Cross-lingual SSL prior, back-end đã quen với spoofing | Tốn VRAM; hiệu năng phụ thuộc fine-tuning/checkpoint | Kết quả mạnh trên ASVspoof 2021 LA/DF [12] |
| XLS-R + Nes2Net-X | XLS-R 300M | Nes2Net-X | Cao, back-end nhẹ | Tận dụng multi-layer SSL features với back-end gọn | Front-end vẫn chiếm chi phí chính | Kết quả tốt trên ASVspoof 2021 và In-the-Wild [13] |

### 2.4 Phương pháp đánh giá

Metric chính của báo cáo là **Equal Error Rate (EER)**. Với một ngưỡng $\tau$, mô hình phân loại các utterance có score $s \geq \tau$ là *bonafide* và $s < \tau$ là *spoof*. Khi đó:

$$
\mathrm{FAR}(\tau) = \frac{\mathrm{FP}(\tau)}{\mathrm{FP}(\tau) + \mathrm{TN}(\tau)}
$$

$$
\mathrm{FRR}(\tau) = \frac{\mathrm{FN}(\tau)}{\mathrm{FN}(\tau) + \mathrm{TP}(\tau)}
$$

Ở đây FAR (False Acceptance Rate) là tỉ lệ spoof bị chấp nhận nhầm như bonafide, còn FRR (False Rejection Rate) là tỉ lệ bonafide bị từ chối nhầm như spoof. EER là giá trị tại ngưỡng $\tau^*$ sao cho $\mathrm{FAR}(\tau^*) \approx \mathrm{FRR}(\tau^*)$. Trong thực tế, $\tau^*$ được tìm bằng cách sweep qua các score hoặc nội suy trên ROC/DET curve.

Ưu điểm của EER là không phụ thuộc vào một operating threshold cố định, nên phù hợp để so sánh các mô hình có thang score khác nhau. Đây là lý do EER được dùng làm metric chính trong Chương 3, nơi mục tiêu là reproduce và so sánh tương đối giữa sáu mô hình trên bốn dataset. Tuy nhiên EER cũng có hạn chế: nó giả định hai loại lỗi có chi phí cân bằng, trong khi deployment thực tế có thể xem spoof false acceptance nguy hiểm hơn bonafide false rejection. EER cũng không phản ánh calibration của score; một mô hình có EER thấp vẫn có thể tạo score không tương ứng với xác suất thật, gây khó khi chọn threshold cố định.

Các challenge ASVspoof còn dùng **minimum tandem Detection Cost Function (min t-DCF)** để đánh giá CM khi đặt trong hệ thống ASV hoàn chỉnh [14]. Metric này tính đến prior và cost của nhiều loại lỗi trong pipeline ASV+CM, nên gần deployment hơn EER trong bối cảnh speaker verification. Tuy nhiên min t-DCF cần thông tin về ASV system, ASV score và cost model. Trong các kết quả reproduce của báo cáo, dữ liệu hiện có chỉ gồm CM scores và labels, vì vậy EER là lựa chọn nhất quán hơn cho phạm vi Internship 1.

Ngoài EER, **ROC curve** và **DET curve** là công cụ trực quan để xem trade-off giữa các loại lỗi ở nhiều threshold. ROC biểu diễn True Positive Rate theo False Positive Rate; DET biểu diễn FAR theo FRR trên thang xác suất Gaussian, thường được cộng đồng speaker verification và anti-spoofing dùng vì dễ quan sát vùng lỗi thấp. **Calibration metrics** như log loss hoặc Expected Calibration Error có ý nghĩa khi hệ thống cần score xác suất hoặc threshold cố định, nhưng chưa được phân tích sâu trong báo cáo này.

### 2.5 Các thách thức hiện tại

Từ các survey gần đây và từ thiết kế của các benchmark ASVspoof/In-the-Wild, có thể rút ra bốn thách thức chính làm nền cho phần thực nghiệm ở Chương 3 [3], [4], [15].

**Cross-domain generalization.** Dataset evidence rõ nhất là khoảng cách giữa clean benchmark và in-the-wild benchmark. ASVspoof 2019 LA có điều kiện kiểm soát, attack metadata rõ và ít yếu tố channel phức tạp. In-the-Wild lại chứa audio từ nguồn công khai, speaker distribution khác, codec khác và attack metadata thiếu. Nếu mô hình học artifact phụ thuộc dataset thay vì cue tổng quát của spoofed speech, EER sẽ tăng mạnh khi chuyển domain. Về mặt kiến trúc, hand-crafted features và small end-to-end models dễ bị ảnh hưởng hơn vì representation được học trong phạm vi dữ liệu anti-spoofing hẹp. SSL front-end được kỳ vọng giảm một phần gap này nhờ pre-training trên audio đa dạng, nhưng cần kết quả cross-dataset ở Chương 3 để kiểm chứng trong project cụ thể.

**Codec robustness.** ASVspoof 2021 DF được thiết kế trực tiếp cho vấn đề này: audio từ nguồn ASVspoof 2019 được xử lý qua nhiều codec/bitrate [2]. Codec lossy có thể xoá high-frequency components, làm mờ phase/spectral artifacts hoặc tạo artifact mới không liên quan đến synthesis. Điều này ảnh hưởng đặc biệt tới các mô hình dựa vào spectral cues cục bộ như LFCC+LCNN, và cũng có thể ảnh hưởng tới raw waveform models nếu chúng học artifact ở waveform distribution gốc. Nếu SSL representation học được cấu trúc speech ổn định hơn, các mô hình XLS-R-based trong Chương 3 nên có generalization gap nhỏ hơn trên ASVspoof 2021 DF.

**Modern attacks và adversarial setting.** Các hệ TTS/VC hiện đại dùng neural codec, diffusion hoặc pipeline nhiều tầng có artifact khác với các hệ cũ trong ASVspoof 2019. ASVspoof 5 đưa thêm các attack gần thời điểm hiện tại hơn và có adversarial track riêng [3]. Thách thức ở đây không chỉ là "thêm nhiều attack", mà là thay đổi bản chất của cue: detector có thể không còn tìm thấy dấu hiệu vocoder/spectral quen thuộc. Vì vậy thứ hạng mô hình trên ASVspoof 2019 LA không nhất thiết dự đoán thứ hạng trên ASVspoof 5. Đây là lý do §3.4 chọn ASVspoof 5 làm trọng tâm error analysis.

**Multi-linguality, fairness và deployment constraints.** Phần lớn benchmark kinh điển tập trung tiếng Anh hoặc một số corpus giới hạn. MLAAD mở rộng câu hỏi sang 23 ngôn ngữ, cho thấy SDD không nên chỉ được đánh giá trên English-centric data [8]. Ngoài ra, deployment thực tế còn bị ràng buộc bởi latency, VRAM, privacy và khả năng chạy trên edge device. AASIST/AASIST-L có lợi thế compute, nhưng có thể kém ổn định khi domain thay đổi; XLS-R/Nes2Net có lợi thế representation nhưng chi phí cao. Trade-off này không được giải quyết chỉ bằng một bảng EER, mà cần phân tích đồng thời accuracy, robustness và resource cost trong các bước tiếp theo.

Các thách thức trên tạo khung diễn giải cho Chương 3: ASVspoof 2019 LA kiểm tra clean benchmark performance, ASVspoof 2021 DF kiểm tra codec robustness, ASVspoof 5 kiểm tra modern attack robustness ở mức sơ bộ, và In-the-Wild kiểm tra cross-domain generalization.

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

*[Liệt kê các bước Internship 2 và Luận văn: cụ thể hoá đề xuất thành experiment plan chi tiết, triển khai và đánh giá đề xuất, cân nhắc mở rộng sang dataset mới sau khi verify nguồn chính thức, thử các SSL front-end khác (WavLM, multi-lingual XLS-R).]*

---

## Tài liệu tham khảo

[1] X. Wang, J. Yamagishi, M. Todisco, H. Delgado, A. Nautsch, N. Evans, et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech & Language*, vol. 64, 2020.

[2] J. Yamagishi, X. Wang, M. Todisco, M. Sahidullah, J. Patino, A. Nautsch, et al., "ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild," *Proceedings of the ASVspoof 2021 Workshop*, 2021.

[3] X. Wang, J. Yamagishi, et al., "ASVspoof 5: Crowdsourced speech data, deepfakes, and adversarial attacks at scale," arXiv:2408.09391, 2024.

[4] N. M. Müller, P. Czempin, F. Dieckmann, A. Froghyar, and K. Böttinger, "Does audio deepfake detection generalize?," *Interspeech*, 2022.

[5] H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher, "End-to-end anti-spoofing with RawNet2," *ICASSP*, 2021.

[6] J.-w. Jung, H.-S. Heo, H. Tak, H.-j. Shim, J. S. Chung, B.-J. Lee, H.-J. Yu, and N. Evans, "AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks," *ICASSP*, 2022.

[7] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," *NeurIPS*, 2020.

[8] N. M. Müller, P. Kawa, W. H. Choong, E. Casanova, E. Gölge, T. Müller, P. Syga, P. Sperl, and K. Böttinger, "MLAAD: The Multi-Language Audio Anti-Spoofing Dataset," arXiv:2401.09512, 2024.

[9] A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, et al., "XLS-R: Self-supervised cross-lingual speech representation learning at scale," *Interspeech*, 2022.

[10] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, et al., "WavLM: Large-scale self-supervised pre-training for full stack speech processing," *IEEE Journal of Selected Topics in Signal Processing*, vol. 16, no. 6, pp. 1505-1518, 2022.

[11] K. Borodin, V. Kudryavtsev, D. Korzh, A. Efimenko, G. Mkrtchian, M. Gorodnichev, and O. Rogov, "AASIST3: KAN-enhanced AASIST speech deepfake detection using SSL features and additional regularization for the ASVspoof 2024 Challenge," arXiv:2408.17352, 2024.

[12] H. Tak, M. Todisco, X. Wang, J.-w. Jung, J. Yamagishi, and N. Evans, "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation," *Odyssey*, 2022.

[13] T. Liu, D.-T. Truong, R. K. Das, K. A. Lee, and H. Li, "Nes2Net: A lightweight nested architecture for foundation model driven speech anti-spoofing," *IEEE Transactions on Information Forensics and Security*, vol. 20, pp. 12005-12018, 2025.

[14] T. Kinnunen, H. Delgado, N. Evans, K. A. Lee, V. Vestman, A. Nautsch, et al., "t-DCF: A detection cost function for the tandem assessment of spoofing countermeasures and automatic speaker verification," *Odyssey*, 2018.

[15] Z. Li, J. Yi, X. Wang, and H. Zhao, "A survey on speech deepfake detection," *ACM Computing Surveys*, 2024.
