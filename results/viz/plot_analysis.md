# Phân tích định tính các confident error trên ASVspoof 5

## Loại lỗi cần phân tích nhất

Theo report.md (§3.4.4 và F1 ở §3.6), hai loại lỗi đáng được soi định tính nhất là:

1. **Confident false positive trên các attack hiện đại nguy hiểm A26/A17/A28** (§3.4.4). Đây là loại lỗi mà spoof bị model gán score bonafide rất cao — nguy hiểm trong deployment vì audio giả được chấp nhận như thật. Mở rộng tự nhiên sang các attack mới khác trong ASVspoof 5 (A18/A19/A20/A30) cũng đáng quan tâm.
2. **Confident error trên codec C04 (Encodec) và C07 (MP3+Encodec)** (F1). EER của nhóm XLS-R nhảy vọt 35-42% ở hai codec này so với 2.71-3.78% trên `nocodec`, chứng tỏ codec robustness gap là failure mode trọng yếu.

## Các audio trong tập visualize thuộc nhóm cần phân tích

Cross-reference `selected_hard_errors.csv` (30 audio đã vẽ) với hai tiêu chí trên:

**Nhóm 1 — FP trên attack A17/A26/A28 (deployment-dangerous):**
- E_0008501537 — A28, C11, XLS-R+AASIST, conf 0.99998
- E_0009991371 — A17, C06, LFCC+LCNN, conf 1.0

**Nhóm 1' — FP trên attack hiện đại khác (A18/A19/A20/A30):**
- E_0001305125 — A30, **C04**, LFCC+LCNN, conf 1.0 *(giao với nhóm 2)*
- E_0001304530 — A18, C06, LFCC+LCNN
- E_0001306028, E_0001307071 — A18, nocodec, LFCC+LCNN
- E_0001306097, E_0001307106 — A30, nocodec, LFCC+LCNN
- E_0001307935 — A20, nocodec, LFCC+LCNN
- E_0009983663 — A18, C06, LFCC+LCNN
- E_0009992180 — A19, C06, LFCC+LCNN

**Nhóm 2 — bonafide FN trên C04/C07 (XLS-R group, codec dominance):**
- E_0000430717, E_0001015980, E_0002108564, E_0008322877 — bonafide, **C04**, XLS-R+Nes2Net
- E_0001333311, E_0004419784, E_0006647216, E_0008117010, E_0008144462 — bonafide, **C07**, XLS-R+Nes2Net

**Năm audio được phân tích chi tiết dưới đây** (chọn để cover cả hai trục):

1. `E_0008501537` — FP A28/C11 (XLS-R+AASIST) — nhóm 1
2. `E_0009991371` — FP A17/C06 (LFCC+LCNN) — nhóm 1
3. `E_0001305125` — FP A30/C04 (LFCC+LCNN) — giao nhóm 1' và 2
4. `E_0002108564` — bonafide FN/C04 (XLS-R+Nes2Net) — nhóm 2
5. `E_0001333311` — bonafide FN/C07 (XLS-R+Nes2Net) — nhóm 2

---

## E_0008501537 — A28 spoof, codec C11, XLS-R+AASIST FP (conf ≈ 0.99998)

**Analysis:** Đây là confident false positive nguy hiểm nhất trong bộ visualize: A28 (một trong những attack hiện đại được report nêu tên ở §3.4.4) đi qua C11 (varied calling/PSTN-like 8 kHz) bị XLS-R+AASIST gán bonafide score ≈ 1. LTAS thể hiện cắt brick-wall sắc nét ở ~4 kHz, hai đường spoof (xanh) và bonafide reference E_0004041591 (cam) gần như chồng khít trong dải 0-4 kHz — toàn bộ dải tần >4 kHz, nơi neural vocoder thường để lại artifact aliasing/spectral smoothing, đã bị codec xoá sạch. Waveform có một quãng silence/nhiễu rất nhẹ kéo dài 0-3.5 s rồi mới đến speech voiced từ ~4 s, gợi ý lead-in padding kiểu TTS/VC alignment. STFT và mel cho thấy harmonic stripes thẳng tắp, formant đầy đủ — speech "nghe có vẻ tự nhiên" sau khi codec bóp băng tần. LFCC gần như phẳng (chỉ vài coefficient đầu có biên độ) vì băng tần hữu ích quá hẹp. Centroid bám quanh 500-800 Hz, rolloff ~3.5 kHz — đặc trưng narrowband telephony. Kết luận: model mất hết cue HF nên fallback vào thống kê sub-4 kHz vốn quá giống bonafide → chấp nhận attack hiện đại như thật. Đây chính là failure mode "audio giả bị chấp nhận trong deployment" mà §3.4.4 cảnh báo.

## E_0009991371 — A17 spoof, codec C06 (M4A 16 kHz), LFCC+LCNN FP (conf = 1.0)

**Analysis:** A17 là attack được report đặc biệt nhắc tên ở §3.4.4. C06 là M4A 16 kHz — về lý thuyết "dễ" hơn nhóm C04/C07 (báo cáo dùng C06 làm "điểm tương phản"), nhưng LFCC+LCNN vẫn fail ở mức confidence tuyệt đối 1.0. LTAS giữ băng tần đến gần 8 kHz, nhưng trên 4 kHz đường spoof (xanh) cao hơn bonafide reference (cam) đáng kể — A17 để lại energy excess ở HF kiểu vocoder, *cue rõ nhưng LFCC+LCNN không khai thác được*. Spectrogram và mel cho thấy harmonic stripes mạnh ở vùng formant 1-3 kHz, một số "banding" theo phương ngang ở dải 3-5 kHz có thể là artifact aliasing/quantization của neural vocoder qua M4A. LFCC panel hoàn toàn phẳng (toàn vùng đỏ, không có texture) — feature design của LFCC chia đều theo linear filter bank không bắt được cấu trúc fine-grained ở các sub-band mà attack hiện đại như A17 hay rò rỉ artifact. Centroid và rolloff dao động lớn (centroid 1.5-2.5 kHz, rolloff đạt đỉnh ~5-6 kHz) — tín hiệu vẫn còn HF content, nhưng feature hand-crafted không tận dụng. Kết luận: hai vấn đề chồng nhau — (i) LFCC+LCNN bị giới hạn bởi feature design ngay cả ở codec không quá khắc nghiệt, (ii) A17 là attack hiện đại tạo cue tinh tế nằm ngoài "vùng quan sát" của hand-crafted feature. Phù hợp với nhận định ở §3.4.3 rằng LFCC+LCNN bị giới hạn rộng hơn chỉ một vài codec khó.

## E_0001305125 — A30 spoof, codec C04 (Encodec 16 kHz), LFCC+LCNN FP (conf = 1.0)

**Analysis:** Mẫu này nằm ở giao điểm hai trục báo cáo quan tâm — modern attack (A30) trên codec khó nhất của benchmark (C04 Encodec). LTAS cho thấy đường spoof (xanh) *thấp hơn* bonafide reference (cam) khoảng 5-10 dB suốt dải 1-7 kHz, kèm vài đỉnh trồi sụt ở 1 kHz và 4-5 kHz — đặc trưng "neural codec output đi qua neural codec lần nữa": Encodec đã quantize tín hiệu spoof và làm phổ trung bình tụt xuống. Waveform có cấu trúc burst-pause rõ rệt với các pulse mạnh tại t ≈ 1.0, 1.6, 4.5 s xen kẽ silence — kiểu prosody không tự nhiên, có thể là A30 attack tạo speech ngắt quãng. Spectrogram thể hiện harmonic structure khá thưa, năng lượng tập trung dưới 1 kHz, và xuất hiện *vertical banding* ở dải 3-6 kHz đáng nghi (artifact quantization của Encodec). LFCC panel có cấu trúc rõ hơn so với mẫu C11/C06 (do còn băng tần đầy đủ 0-8 kHz), nhưng vẫn không đủ giúp LFCC+LCNN. Centroid và rolloff cao bất thường (centroid ~3-4 kHz, rolloff đạt 6-7 kHz) — phổ "phẳng quá mức", một dấu hiệu của codec compression. Kết luận: đây là hai-tầng-artifact (A30 + Encodec) — Encodec đã "làm sạch" cue mà LFCC+LCNN từng học được trong training (chủ yếu là ASV2019 LA clean), nên model fall back sang quyết định mù dựa trên prior. Phù hợp với F1: C04 là nơi gap codec lớn nhất.

## E_0002108564 — Bonafide, codec C04 (Encodec 16 kHz), XLS-R+Nes2Net FN (conf ≈ 0.99999)

**Analysis:** Đây là mẫu tiêu biểu của F1 codec dominance: speech bonafide thật nhưng XLS-R+Nes2Net — mô hình tốt nhất overall — gán spoof score 0.99999. LTAS đường hard error (xanh) và đường bonafide reference E_0000125404 cùng codec (cam) gần như chồng nhau hoàn toàn trên toàn dải 0-8 kHz, hai đường cùng có một dốc nhẹ giảm dần và cùng cắt sharp tại ~7.5 kHz (Encodec cutoff). Waveform liên tục, voiced energy đều suốt 0-7 s, không có silence padding hay burst lạ. STFT và mel hiển thị harmonic stripes thẳng, formant đầy đủ, energy phân bố tự nhiên — đây là speech rõ ràng natural. LFCC panel có một số texture nhẹ ở các coefficient bậc cao (do còn băng tần rộng), nhưng tổng thể vẫn nghèo cấu trúc. Centroid 1-2 kHz, rolloff 3-5 kHz, RMS biến thiên tự nhiên theo voicing, ZCR ổn định — tất cả descriptor đều "trong khoảng bonafide". Kết luận: không có dấu hiệu spoof nào nhìn thấy được; XLS-R+Nes2Net phán sai *vì* representation học từ XLS-R 300M chưa từng thấy Encodec output trong pretraining/fine-tuning, nên Encodec artifact rơi vào vùng OOD và bị nhầm với "spoof manifold". Đây chính là điểm mà §4.4 đề xuất codec-aware training và §3.4.3 chỉ ra C04 chiếm phần lớn EER.

## E_0001333311 — Bonafide, codec C07 (MP3+Encodec 16 kHz), XLS-R+Nes2Net FN (conf ≈ 0.99999)

**Analysis:** Bằng chứng đối xứng với E_0002108564 nhưng trên C07 — chuỗi nén kép MP3+Encodec, codec khó nhất trên benchmark. LTAS hai đường (hard error xanh, bonafide ref E_0008104077 cùng C07 cam) lại chồng khít, không có sai khác phổ nào đáng kể; cả hai cùng có lưng dốc đều và cutoff mềm ở ~7.5 kHz. Waveform speech liên tục, energy phân bố tự nhiên theo phonetic structure, không có pulse hay artifact thị giác nào. STFT thể hiện harmonic stripes rõ và formant trajectory hợp lý; mel-spectrogram cho thấy band-level energy phân bố bình thường, không có "hole" hay "spike" lạ. LFCC panel phẳng — đặc trưng của tín hiệu đã đi qua hai tầng codec làm mất chi tiết phổ. Centroid quanh 1-2 kHz, rolloff 3-4.5 kHz, RMS đều — không có lý do hand-crafted nào để gọi đây là spoof. Tuy nhiên XLS-R+Nes2Net tự tin tuyệt đối nó là spoof. Cơ chế lỗi giống mẫu trước: chuỗi MP3→Encodec để lại dấu vết quantization và pre-echo đặc biệt mà XLS-R chưa từng thấy lúc pretraining, đẩy embedding của bonafide thật vào vùng "spoof-like". Hai mẫu C04 và C07 cùng nhau xác nhận: gap không nằm ở attack family hay phổ macroscopic, mà nằm ở *codec footprint dưới ngưỡng cảm nhận thị giác* — phát hiện này khớp với F2 (ASV21 DF không đại diện cho codec robustness thực tế) và là motivation cho codec-aware training/CodecFake-style data ở §4.4.

---

---

## Bổ sung: focus trên XLS-R+Nes2Net × C04/C07

Vì XLS-R+Nes2Net là baseline mạnh nhất overall (dẫn đầu ASV19 LA, ASV21 DF, ITW theo Chương 4) nhưng lại bị C04/C07 đẩy EER lên 41-42% (F1), nhóm lỗi này là điểm yếu cụ thể, có thể tấn công bằng codec-aware training. Bốn mẫu C04 và bốn mẫu C07 dưới đây đều là bonafide bị model gán spoof score ≈ 1 — failure mode đối xứng với "deployment-dangerous FP" nhưng ở chiều ngược lại (bonafide bị reject), gây tổn thất usability.

### E_0000430717 — bonafide C04 (codec_q=4, speaker E_4792, conf ≈ 0.999996)

**Analysis:** Speech nữ liên tục ~5.4 s, tách thành hai burst bởi pause ngắn ~2.4-2.7 s. LTAS hard error (xanh) và bonafide ref E_0002699667 cùng C04 (cam) gần như chồng khít trên toàn dải, chỉ chênh 1-2 dB trong vùng 4-7 kHz và một dip nhẹ của hard error ở ~5.5 kHz — không có khác biệt phổ vĩ mô. STFT/mel cho thấy F0 nữ ~200 Hz rất ổn định, harmonic stripes thẳng và đậm đến ~5-6 kHz rồi suy giảm tự nhiên; formant trajectory rõ ràng, không có "smear" ngang lạ. LFCC panel chỉ có hoạt động yếu ở vài coefficient bậc cao. Centroid 1-2.5 kHz, rolloff đạt đỉnh ~5 kHz ở các âm tiết stressed, ZCR thấp — đều nằm trong khoảng bonafide. Không có dấu hiệu spoof bằng mắt; model phán sai vì Encodec footprint nằm ngoài training distribution của XLS-R.

### E_0008322877 — bonafide C04 (codec_q=5, conf ≈ 0.99999)

**Analysis:** Utterance dài 9.5 s với nhiều speech burst đều nhau. LTAS hai đường (hard error xanh và bonafide ref E_0007132163 cùng C04, cam) chồng khít gần như hoàn hảo, kể cả vùng peak 200-400 Hz và phần dốc lên 7 kHz; chỉ một offset rất nhỏ ~1 dB ở 4-6 kHz. Spectrogram cho thấy harmonic structure chuẩn, formant chuyển động hợp lý theo nguyên âm, không có band-suppression hay vertical banding nghi vấn nào. LFCC panel mịn, không có cấu trúc nổi bật. Centroid và rolloff có nhịp lên/xuống đúng theo voicing, không "đều quá" như speech synthesis thường gặp. Đây là một bonafide chuẩn mực — sự thật model gán spoof confidence ~1 cho thấy XLS-R+Nes2Net hoàn toàn không "nhìn vào" phổ; quyết định dựa trên embedding mà C04 đẩy ra khỏi manifold bonafide đã học.

### E_0001015980 — bonafide C04 (codec_q=4, conf ≈ 0.9999931)

**Analysis:** Speech nữ ~5.8 s, energy tập trung 0.3-5 s. LTAS có hiện tượng đáng chú ý: đường hard error (xanh) *cao hơn* bonafide ref E_0005620350 (cam) khoảng 1-3 dB trong dải 3-7 kHz, đặc biệt một bump ở ~4 kHz — đây có thể là spectral leakage do Encodec quantization ở codec_q=4 (bitrate thấp), nhưng đó *vẫn là speech thật*. STFT/mel cho thấy harmonic clear, formant ổn định, F0 ~200 Hz tự nhiên. LFCC panel có texture nhẹ nhưng tổng thể vẫn rất thưa. Centroid 1.5-3 kHz, rolloff peaks ~5 kHz, RMS variation tự nhiên. Điểm thú vị: LTAS *cùng codec* mà hai utterance khác nhau (cùng C04, cùng speaker female class) đã có offset nhỏ → confirm là *codec_q* và nội dung phát âm gây offset, không phải sự khác biệt spoof/bonafide. XLS-R+Nes2Net dường như phản ứng quá mạnh với cái bump 4 kHz đó.

### E_0008117010 — bonafide C07 (codec_q=3, conf ≈ 0.9999864)

**Analysis:** Utterance dài 9.7 s với một silence gap rõ ràng 4.7-5.5 s chia thành hai đoạn speech. LTAS hard error và bonafide ref *gần như chồng hoàn toàn* — kể cả tail trên 7 kHz nơi cả hai cùng cắt cùng độ dốc (Encodec cutoff đặc trưng). Spectrogram thể hiện hai burst với harmonic structure rõ ràng, formant dynamics tự nhiên. Hai burst có energy profile khác nhau (burst 1 mạnh hơn ở 0.5-4 s, burst 2 ở 6-9 s) — kiểu speech tự nhiên, không phải lặp synthesized. LFCC gần như hoàn toàn phẳng (codec C07 = MP3+Encodec chain bóp băng tần và quantization fine detail). Centroid 1-2 kHz, rolloff 3-5 kHz; RMS có nhịp tự nhiên với silence ở giữa. Không thể tìm ra cue spoof bằng mắt — XLS-R+Nes2Net nhầm hoàn toàn do chain MP3→Encodec để lại "double-quantization watermark" mà model coi như spoof signature.

### E_0008144462 — bonafide C07 (codec_q=4, conf ≈ 0.9999853)

**Analysis:** Utterance ~9.7 s với pause khá dài 4-5.5 s, chia rõ thành hai phần. LTAS bonafide ref E_0005493435 (cam) và hard error (xanh) chồng khít trên toàn dải, hard error chỉ suy giảm 0.5-1 dB ở 4-6 kHz — gần như không có hiệu ứng phổ macroscopic. Spectrogram cho thấy harmonic stripes thẳng, formant rõ, energy phân bố tự nhiên theo voicing. Mel-spectrogram có dynamic range tốt, không có "checkerboard" hay banding của neural codec artifact ở mức nhìn thấy. LFCC panel mịn không có cấu trúc bậc cao. Spectral descriptors biến động lớn theo nội dung phát âm (centroid lên đến 3 kHz tại các âm sat/voiceless, RMS spikes ở consonant) — phản ứng tự nhiên. Đây lại là bonafide hoàn toàn "trông như bonafide" nhưng XLS-R+Nes2Net gán spoof score 1 — cùng cơ chế: codec chain C07 đẩy embedding khỏi manifold.

### E_0006647216 — bonafide C07 (codec_q=5, conf ≈ 0.9999835)

**Analysis:** Mẫu ngắn nhất trong nhóm, ~4.8 s. LTAS có khác biệt rõ nhất trong sáu mẫu này: hard error (xanh) thấp hơn bonafide ref E_0008852743 (cam) đáng kể trong dải 1-3 kHz (chênh 3-5 dB), và *cao hơn* ref ở dải 4-7 kHz — pattern "spectral tilt nghiêng ngược". Tuy vậy, hai đường cùng cắt cùng độ dốc ở ~7.5 kHz (cùng codec C07). Waveform thể hiện speech nữ với F0 cao, các burst voiced mạnh tại t ≈ 1.0, 2.0, 3.0, 4.2 s. Spectrogram cho thấy harmonic stripes rất rõ, F0 ổn định ~250 Hz, formant trajectory hợp lý. LFCC vẫn phẳng. Centroid spikes lên 3-4 kHz tại các âm bật, rolloff đạt 6 kHz ở các đỉnh — tín hiệu hoàn toàn natural. Spectral tilt khác biệt ở đây có thể giải thích bằng nội dung phát âm (nhiều âm vô thanh / fricative) chứ không phải spoof artifact; XLS-R+Nes2Net không phân biệt được hai khả năng này khi đứng sau C07 chain.

### Tổng kết riêng cho nhóm XLS-R+Nes2Net × C04/C07

Tám mẫu (bao gồm hai mẫu đã phân tích chi tiết ở phần chính: E_0002108564, E_0001333311) cho ra một mẫu hình rất nhất quán:

1. **Phổ vĩ mô bonafide hard-error nhìn như bonafide ref cùng codec** — không có cue spoof bằng mắt nào ở LTAS, STFT, mel hay LFCC.
2. **Khác biệt nhỏ chủ yếu đến từ codec_q và nội dung phát âm**, không phải nguồn gốc spoof/bonafide.
3. **Model vẫn tự tin gán spoof với score ≈ 1**, chứng tỏ quyết định dựa trên fingerprint codec ở mức embedding chứ không phải đặc trưng âm học có thể giải thích bằng phổ.
4. **Hai codec neural-based (C04 Encodec, C07 MP3+Encodec) tạo cùng failure mode** dù khác cơ chế nén → đề xuất codec-aware augmentation cần bao gồm cả pure neural codec lẫn chain codec (CodecFake/CodecFake+ kiểu data).
5. **codec_q dao động trong nhóm này (3-5)** nhưng không loại bỏ được lỗi → vấn đề không phải bitrate; vấn đề là *sự có mặt của Encodec footprint*.

Hệ quả thực tiễn cho roadmap (khớp với §4.4 trong report):
- Codec-aware fine-tuning với Encodec/MP3+Encodec làm augmentation trong loại trừ tự tin sai
- Consistency regularization giữa cùng utterance qua các codec khác nhau, ép embedding ổn định
- Score calibration trên dev split có codec để tránh model "đè quá tay" về spoof khi gặp neural codec

---

## Bổ sung: confident False Positive trên modern attack (A28, A17)

FP là loại lỗi có ý nghĩa security: spoof bị model gán bonafide. Verify từ `error_analysis.pkl` cho 5 mô hình trên ASV5 eval (FP rate per attack):

- **A28** = #1 FP cho cả hai mô hình XLS-R: XLS-R+AASIST 0.17%, XLS-R+Nes2Net 3.01%.
- **A17** = #1 FP cho AASIST (11.96%), #2 cho AASIST-L (8.98%); đồng thời nằm top-2 của cả hai XLS-R model (0.06% và 0.69%) → là attack cross-cutting khó nhất qua hai họ kiến trúc.

Bốn mẫu dưới đây thuộc nhóm FP confident trên A28/A17 với codec = nocodec (loại bỏ biến codec, để hiệu ứng còn lại thuần là "attack quality vs model representation").

### E_0004782150 — A28 nocodec, XLS-R+Nes2Net FP (conf ≈ 0.99999)

**Analysis:** Speech ~8 s với nhiều burst voiced, một silence gap ngắn 1.5-2.5 s. LTAS hard error (xanh A28) và bonafide reference E_0009676852 (cam) chồng khít gần như hoàn hảo trên toàn dải 0-8 kHz, kể cả phần roll-off mềm ở ~7.5 kHz. Không có offset spectral tilt, không có bump/dip ở bất kỳ dải tần nào — A28 đã tạo được spectral envelope đồng nhất với bonafide. STFT và mel cho thấy harmonic structure dày, formant trajectory ổn định theo nguyên âm, không có vertical banding hay replication artifact nhìn thấy. LFCC mịn, chỉ vài coefficient bậc thấp có hoạt động. Centroid 1-2.5 kHz dao động hợp lý theo voicing, rolloff đạt 5 kHz ở consonant, ZCR thấp — tất cả descriptor đều rơi vào khoảng bonafide. → Không có cue phổ vĩ mô để model bắt; XLS-R+Nes2Net FP với conf ≈ 1 cho thấy attack đã tiệm cận manifold bonafide ở mức representation. Đây là kịch bản failure mode đáng lo nhất: attack hiện đại không để lại fingerprint phân biệt bằng mắt.

### E_0009392843 — A28 nocodec, XLS-R+Nes2Net FP (conf ≈ 0.99996)

**Analysis:** Pattern waveform đáng chú ý: 0-3.5 s gần như im lặng / nhiễu rất nhẹ, rồi speech voiced từ 4 s trở đi — đây là dạng "lead-in padding" rất đặc trưng pipeline TTS/VC (alignment đệm silence ở đầu). Đây là *cue tiềm năng* mà XLS-R+Nes2Net không khai thác. LTAS hard error (xanh) chồng khít bonafide reference E_0001292361 (cam) trên dải 0-6 kHz, hard error chỉ tụt 1-3 dB ở dải 5-7 kHz. STFT/mel: harmonic và formant tự nhiên trong segment voiced; vùng silence đầu utterance không có hiện tượng "phantom harmonics" hay buzz, gợi ý A28 generator giữ silence sạch (không vocoder bleed). LFCC flat. Centroid và rolloff biến thiên hợp lý. → Cue duy nhất khả nghi là cấu trúc thời gian (silence padding); cue phổ thì không có. Confirm A28 ở chế độ nocodec là điểm yếu thuần ở mức temporal/representation, không phải spectral.

### E_0002841318 — A17 nocodec, AASIST FP (conf ≈ 0.99999)

**Analysis:** *Đây là mẫu thú vị nhất trong bốn mẫu mới — A17 để lại cue phổ rất rõ mà AASIST vẫn bỏ qua.* Speech ~7.8 s với prosody biến thiên tự nhiên. LTAS hard error (xanh A17) **cao hơn** bonafide reference E_0008820523 (cam) khoảng 5-10 dB trong dải 1-7 kHz — spectral tilt của A17 *nông hơn* hẳn bonafide (high frequencies relatively boosted). Đây là fingerprint synthesizer rõ rệt: pipeline TTS thường tạo speech có HF content phẳng/đậm hơn natural speech. STFT thể hiện vertical banding nhẹ ở dải 2-5 kHz tại các consonant — artifact aliasing/quantization tiềm năng của vocoder. Mel có cấu trúc "frame-like" đều đặn hơn natural. LFCC vẫn flat (vì AASIST không dùng LFCC). Centroid và rolloff cao bất thường (centroid spike ~3-4 kHz, rolloff peaks ~5-6 kHz), khớp với LTAS tilt nông. → **Cue tồn tại rõ ở mức spectral** (LTAS slope, vertical banding, centroid cao), nhưng AASIST raw-waveform end-to-end **bỏ qua hoàn toàn** và FP với conf ≈ 1. Đây là evidence trực quan rằng AASIST trained on ASV19 LA đã học artifact phụ thuộc training distribution, không phải tilt/spectral signature tổng quát.

### E_0002067090 — A17 nocodec, AASIST FP (conf ≈ 0.99999)

**Analysis:** Pattern lặp lại của E_0002841318, xác nhận đặc trưng phổ của A17 là systematic chứ không phải single-sample fluke. Speech ~7.5 s, voiced liên tục. LTAS hard error (xanh) **cao hơn** bonafide reference E_0002328452 (cam) ~5-10 dB trên dải 2-7 kHz, cùng pattern "spectral tilt nông" như mẫu trước. STFT có harmonic structure rõ kèm vertical banding nhẹ; mel cho thấy energy phân bố đều trong dải mid-high khác với bonafide thường tập trung dưới 3 kHz. LFCC có texture rõ hơn các mẫu khác (vì nocodec giữ băng tần đầy đủ + A17 tạo spectral spread). Centroid và rolloff đạt đỉnh 5-6 kHz ở consonant — confirm spectrum nông. Đáng chú ý theo verify từ error_analysis.pkl, mẫu này cũng có score FP cao trên cả hai XLS-R model (rank top FP của A17 cho XLS-R+Nes2Net) — tức A17 *cũng đánh được* SSL representation, dù FP rate tổng thấp hơn nhiều so với AASIST (0.69% vs 11.96%). → AASIST bỏ cue spectral của A17 hoàn toàn; XLS-R+Nes2Net giảm mạnh nhưng chưa loại bỏ.

### Tổng kết riêng cho nhóm FP × A28/A17

Bốn mẫu nocodec (loại bỏ biến codec) cho ra hai failure mode rất khác nhau giữa hai họ kiến trúc:

1. **A28 vs XLS-R+Nes2Net (3.01% FP rate)** — *No visible spectral cue*. LTAS gần như chồng bonafide; cue khả nghi duy nhất là silence padding cấu trúc thời gian. SSL representation 300M vẫn FP. Đây là kịch bản "attack thực sự gần manifold bonafide" — khó giải quyết bằng spectral feature thuần.

2. **A17 vs AASIST (11.96% FP rate)** — *Visible spectral cue ignored*. LTAS hard error cao hơn bonafide ~5-10 dB ở dải 1-7 kHz, vertical banding ở STFT, centroid/rolloff cao bất thường. AASIST raw-waveform CNN bỏ qua hoàn toàn. Đây là kịch bản "model học sai feature" — có thể cải thiện bằng spectral-aware augmentation hoặc front-end mạnh hơn.

3. **A17 trên XLS-R+Nes2Net** (0.69% FP rate, giảm ~17× so với AASIST): SSL representation *có bắt được* spectral cue của A17 hơn rõ rệt, nhưng chưa hoàn toàn. Câu này confirm SSL front-end *giúp* trên A17 (cue spectral) nhưng *không giúp* trên A28 (cue ẩn).

Hai điểm này tạo asymmetry quan trọng cho roadmap §4.4:
- **Codec-aware augmentation** (đã đề xuất) chỉ giải quyết một nửa C04/C07 FN, không giúp A28 FP vì A28 ở nocodec đã đánh được rồi.
- **Attack-aware augmentation / hard-attack mining** cho A28 và A17 là hướng cần thêm vào: A28 vì SSL 300M chưa đủ; A17 vì raw-waveform end-to-end ignore spectral cue.
- **Front-end/back-end ablation** (F5) có evidence cụ thể: XLS-R giảm FP rate trên A17 từ 11.96% xuống 0.69% so với AASIST → front-end thực sự đóng góp; XLS-R+AASIST giảm FP trên A28 từ 3.01% xuống 0.17% so với XLS-R+Nes2Net (same front-end, khác back-end) → back-end cũng đóng góp đáng kể.

> ⚠ Lưu ý wording cho report: không claim "XLS-R+AASIST tốt hơn XLS-R+Nes2Net chủ yếu vì xử lý A28 tốt hơn" vì chưa decomposition tổng lỗi. Diễn đạt an toàn: *"A28 là FP bottleneck rõ nhất của cả hai XLS-R; XLS-R+AASIST giảm mạnh FP trên A28 (0.17%) so với XLS-R+Nes2Net (3.01%)"*.

## Tóm tắt cross-cutting

- **C11 (narrowband 8 kHz)**: xoá band >4 kHz → cue HF biến mất → lỗi đi cả hai chiều (E_0008501537 FP, E_0008571310 FN) trên cùng codec.
- **C06 (M4A 16 kHz)**: codec "dễ" nhưng LFCC+LCNN vẫn fail (E_0009991371) → giới hạn feature design, không phải codec.
- **C04/C07 (Encodec / MP3+Encodec)**: phổ vĩ mô không khác bonafide, nhưng XLS-R nhầm bonafide thành spoof (E_0002108564, E_0001333311) → footprint codec rơi ngoài training distribution của SSL front-end, không phát hiện bằng mắt.
- Tất cả các FP confident đều có hai đặc điểm chung: (i) phổ "trông quá giống bonafide" sau codec, (ii) cue artifact nằm ở dải tần đã bị nén/cắt → motivation rõ cho codec-aware training và khai thác feature sub-4 kHz như jitter/shimmer/F0/prosody.
