# Report TODOs Checklist

Mục tiêu: hoàn thiện `report/report.md` — báo cáo Internship 1, bước đầu trong pipeline luận văn.

**Vị trí trong pipeline luận văn (rất quan trọng để giữ đúng scope):**
- **Internship 1 (kỳ này)** = literature survey + reproduction + preliminary diagnosis (error analysis sơ bộ) + conceptual proposal (đề xuất hướng tiếp cận ở dạng ý niệm).
- **Internship 2** = detailed proposal + experiment plan (cụ thể hoá đề xuất, thiết kế thí nghiệm chi tiết, baseline cụ thể, metrics, lịch trình).
- **Luận văn** = implement + evaluate + analyze proposed approach (triển khai và đánh giá đề xuất).

→ Hệ quả cho report Internship 1: phần "đề xuất" chỉ ở mức ý niệm/hướng đi (Ch4), KHÔNG cần experiment plan chi tiết. Phần error analysis chỉ cần đủ để dẫn dắt motivation cho đề xuất, không cần bao trùm hết các datasets.

Phạm vi cụ thể: survey (datasets + architectures), reproduce 6 models × 4 datasets, error analysis trên ASVspoof 5, đề xuất hướng tiếp cận sơ bộ.

Cấu trúc report:
- Lời cam đoan / Lời ngỏ / Tóm tắt
- Ch1 Giới thiệu — motivation, thách thức, cấu trúc báo cáo
- Ch2 Cơ sở lý thuyết — datasets, attacks, metrics, architectures, challenges
- Ch3 Reproduce và phân tích kết quả — setup, EER overview 4 datasets, error analysis ASV5
- Ch4 Kết luận và Future work — đề xuất hướng tiếp cận ở dạng ý niệm

Quy ước:
- Research notes (kết quả search/đọc paper) → lưu vào `report/notes/<topic>.md`, KHÔNG paste vào `report.md`
- Mỗi TODO ở dưới phải tự đủ ngữ cảnh để load lại phiên sau và làm độc lập
- Trạng thái: `[ ]` chưa làm, `[~]` đang làm, `[x]` xong, `[!]` blocked/cần chạy thêm thực nghiệm
- **Văn phong**: dùng ngôn ngữ mô tả/quan sát ("kết quả ban đầu cho thấy...", "có xu hướng...", "duy trì EER thấp hơn..."). Tránh assert mạnh ("sụp đổ", "tốt nhất tuyệt đối", "thất bại hoàn toàn") khi error analysis chưa đầy đủ — báo cáo Internship 1 chỉ là preliminary diagnosis.
- **Encoding**: lưu UTF-8 (không cần BOM trên VS Code; nếu IDE hiển thị sai tiếng Việt, kiểm tra setting `files.encoding = utf8`).

---

## A. Khung sườn report

- [ ] **A1. Tạo skeleton report.md với đầy đủ heading + một dòng mô tả cho từng section**
  - Output: `report/report.md` có 4 chương + các phần phụ với heading tiếng Việt, mỗi heading kèm 1 dòng mô tả nội dung sẽ viết. Không viết nội dung thật ở bước này.
  - Mục đích: nhìn được toàn cảnh trước khi điền nội dung; dễ navigation khi viết từng phần.

- [ ] **A2. Tạo folder `report/notes/` cho research notes**
  - Output: `report/notes/datasets.md`, `report/notes/architectures.md`, `report/notes/attacks.md`, `report/notes/metrics.md` — đều skeleton rỗng, kèm 1-2 dòng mô tả nội dung.
  - Mục đích: tách kết quả research khỏi report cuối cùng; report.md chỉ chắt lọc lại.

---

## B. Chương 1 — Giới thiệu

- [ ] **B1. Viết motivation + bài toán speech deepfake detection (~1 trang)**
  - Output: §1.1 trong `report.md`. Nội dung: voice cloning/TTS phát triển nhanh → rủi ro thực tiễn (lừa đảo, mạo danh, disinformation); bài toán: phân loại bonafide vs spoof từ tín hiệu audio.
  - Tham chiếu: trích các con số/case study từ "A Survey on Speech Deepfake Detection" (Li et al., 2024) hoặc Zhang et al., 2025 nếu cần.
  - Yêu cầu: viết theo phong cách báo cáo học thuật tiếng Việt, không bullet.

- [ ] **B2. Viết các thách thức chính (high level → cụ thể)**
  - Output: §1.2. Danh sách 3-4 thách thức: (i) generalization sang attack type/codec/môi trường mới, (ii) tốc độ phát triển TTS/VC vượt tốc độ ra dataset benchmark, (iii) real-world condition (codec, noise, scraping) khác xa benchmark, (iv) imbalance bonafide/spoof.
  - Yêu cầu: dẫn chứng bằng trend EER theo thời gian (ASV19 → ASV21 → ASV5/ITW) ở mức high-level — chưa cần số liệu chi tiết, đó là việc của Ch3.

- [ ] **B3. Viết hướng giải quyết tổng quan + cấu trúc báo cáo**
  - Output: §1.3. Nói ngắn về cách tiếp cận của báo cáo: literature review → reproduce → error analysis → đề xuất sơ bộ. Sau đó liệt kê cấu trúc 4 chương.

---

## C. Chương 2 — Cơ sở lý thuyết

### C.1 Research notes (làm trước, lưu vào `report/notes/`)

- [ ] **C1. Research datasets — phân loại nhóm, lưu vào `report/notes/datasets.md`**
  - Mục đích: trả lời câu "speech deepfake datasets có bao nhiêu nhóm?" — tổ chức theo dạng cây từ high level (logical access vs physical access; clean lab vs in-the-wild; single-attack vs multi-attack vs adversarial).
  - Output: notes chi tiết về ASVspoof 2019/2021/5, In-the-Wild, MLAAD, FoR, WaveFake, EchoFake (2025), v.v. — mỗi dataset: năm, kích thước, attack types, ngôn ngữ, special properties, pros/cons.
  - Sau đó: identify nhóm "thách thức hiện nay" (codec-heavy, in-the-wild, adversarial, multi-lingual) → đó là motivation cho phần evaluation.
  - Phương pháp: dùng Agent (Explore hoặc general-purpose) để search papers 2024-2026; ưu tiên các survey gần đây.

- [ ] **C2. Research architectures — phân loại approach, lưu vào `report/notes/architectures.md`**
  - Mục đích: trả lời "có những hướng tiếp cận nào cho SDD?" — tổ chức cây từ high level (hand-crafted features + classifier; end-to-end DNN; SSL front-end + back-end; ensemble/foundation model).
  - Output: notes về LFCC/CQCC + GMM/LCNN/ResNet, RawNet2, AASIST family, SSL (Wav2Vec2/XLS-R/WavLM) + back-end (AASIST, Nes2Net, MFA), foundation model approaches 2025-2026.
  - Mỗi nhóm: pros, cons, trade-offs (params, generalization, codec robustness).
  - Sau đó: identify nhóm "approach tiềm năng" (SSL+robust back-end) → motivation cho 6 models đã reproduce.

- [ ] **C3. Research attack types & evaluation metrics, lưu vào `report/notes/attacks.md` + `notes/metrics.md`**
  - Attack types: TTS, VC, replay, adversarial, neural codec attacks — mỗi loại 2-3 dòng.
  - Metrics: EER (định nghĩa, công thức), min t-DCF (khi nào dùng), ROC/DET curve, calibration metrics.

### C.2 Viết vào report.md (chắt lọc từ notes)

- [x] **C4. §2.1 Bài toán và các loại tấn công**
  - Output: định nghĩa bài toán phân loại nhị phân, liệt kê các loại tấn công chính, mô tả pipeline tổng quát của một SDD system (frontend → backend → classifier).
  - Source: chắt lọc từ `notes/attacks.md`.

- [x] **C5. §2.2 Datasets — phân loại và mô tả các benchmark chính**
  - Output: bảng tổng hợp datasets (tên, năm, kích thước, attack types, đặc điểm), giải thích tại sao ASVspoof 2019/2021/5 và In-the-Wild được chọn cho phần reproduce.
  - Yêu cầu: nhấn mạnh trend thách thức theo thời gian (clean → codec → adversarial → in-the-wild).
  - Source: chắt lọc từ `notes/datasets.md`.

- [x] **C6. §2.3 Architectures — phân loại và mô tả các approach**
  - Output: phân loại theo cây (hand-crafted, end-to-end DNN, SSL+back-end), với mỗi nhóm có pros/cons/trade-off.
  - Yêu cầu: tập trung kỹ vào nhóm SSL + back-end vì đó là nhóm tiềm năng và được reproduce.
  - Source: chắt lọc từ `notes/architectures.md`.

- [x] **C7. §2.4 Evaluation metrics**
  - Output: định nghĩa EER (kèm công thức), giải thích min t-DCF, mô tả ROC/DET curve.
  - Source: `notes/metrics.md`.

- [x] **C8. §2.5 Thách thức hiện tại**
  - Output: chắt lọc thành 3-4 thách thức cụ thể (cross-dataset generalization, codec robustness, adversarial attacks, real-world conditions) — kèm reference đến paper survey.

- [x] **C9. Mở rộng §2.1 thành định nghĩa bài toán + threat model đầy đủ**
  - Output: §2.1 không chỉ liệt kê attack types mà còn giải thích rõ CM pipeline, score/threshold, quan hệ giữa Speech Deepfake Detection và ASV anti-spoofing, khác biệt giữa Logical Access, Physical Access, Deepfake/in-the-wild và adversarial setting.
  - Yêu cầu: giữ nguyên các thuật ngữ chuyên ngành tiếng Anh khi tự nhiên hơn: `countermeasure`, `front-end`, `back-end`, `bonafide`, `spoof`, `Logical Access`, `Physical Access`, `replay`, `threat model`.

- [x] **C10. Viết lại §2.2 theo cấu trúc taxonomy → bảng → mô tả benchmark chính**
  - Output: §2.2 có taxonomy datasets theo các trục LA/PA, clean/in-the-wild, multi-attack/adversarial, mono/multi-lingual; sau đó là bảng tổng hợp và đoạn mô tả vai trò của từng benchmark chính.
  - Yêu cầu: phân biệt rõ `official dataset size` và `subset/split used in this project`; không đưa claim chưa verify vào giọng văn chắc chắn.

- [x] **C11. Chuẩn hoá số liệu dataset**
  - Output: các số liệu ASVspoof 2019/2021/5, In-the-Wild, MLAAD, WaveFake, FoR trong `report.md` thống nhất với notes hoặc được ghi rõ là số liệu dùng trong project.
  - Yêu cầu: EchoFake hoặc dataset quá mới chỉ ghi như hướng mở rộng cần verify, không đưa số lượng ước tính như một fact chính thức nếu chưa có nguồn chắc.

- [x] **C12. Mở rộng §2.3 architectures theo nguyên lý + inductive bias + failure mode**
  - Output: mỗi nhóm kiến trúc có mô tả cách hoạt động, ưu/nhược điểm, chi phí tính toán, lý do có thể generalize tốt/kém, và liên hệ với 6 mô hình reproduce.
  - Yêu cầu: tập trung sâu hơn vào nhóm `SSL front-end + lightweight back-end` vì đây là nhóm dẫn dắt kết quả Chương 3 và đề xuất Chương 4.

- [x] **C13. Nâng cấp Bảng 2.2**
  - Output: bảng so sánh 6 mô hình có thêm input representation, compute cost, điểm mạnh, hạn chế/failure mode, thay vì chỉ có params và EER paper.
  - Yêu cầu: không kết luận tuyệt đối; dùng "kỳ vọng", "có xu hướng", "trong các benchmark đã công bố".

- [x] **C14. Mở rộng §2.4 metrics**
  - Output: giải thích FAR, FRR, EER bằng công thức; nói rõ cách sweep threshold; bổ sung hạn chế của EER, ý nghĩa min t-DCF, DET/ROC và calibration.
  - Yêu cầu: giải thích vì sao report dùng EER là metric chính trong phạm vi Internship 1.

- [x] **C15. Viết lại §2.5 thành synthesis section**
  - Output: mỗi challenge nối được `dataset evidence → architecture weakness → expectation for experiments`; các challenge chính gồm cross-domain generalization, codec robustness, modern attacks/adversarial, multilingual/fairness ở mức định hướng.
  - Yêu cầu: phần này phải dẫn tự nhiên sang Chương 3, không chỉ lặp lại Chương 1.

- [x] **C16. Thêm citation và references cho toàn bộ Chương 2**
  - Output: các claim chính trong §2.1-§2.5 có citation; phần `Tài liệu tham khảo` có danh sách paper/dataset/model liên quan theo format nhất quán.
  - Yêu cầu: ưu tiên paper gốc/survey chính; hạn chế nguồn không chính thức.

- [x] **C17. Rà lại wording để tránh overclaim**
  - Output: các cụm như "SOTA", "dominate", "robust hơn đáng kể", "hiện đại nhất" được thay bằng ngôn ngữ học thuật thận trọng hoặc có citation rõ.
  - Yêu cầu: phù hợp scope Internship 1 — literature survey + reproduce + preliminary diagnosis.

- [x] **C18. Refresh literature review datasets theo nguồn 2024-2026**
  - Output: `report/notes/datasets.md` và §2.2 trong `report.md` được cập nhật thêm các nhóm/dataset còn thiếu: ASVspoof 5 design/evaluation papers, MLAAD bản mới, CodecFake/CodecFake+, EchoFake, ADD 2022/2023, PartialSpoof.
  - Yêu cầu: phân biệt rõ `reproduced benchmark` và `relevant but out-of-scope benchmark`; mỗi dataset ngoài scope phải có lý do không reproduce (lệch task, quá mới, chi phí lớn, thiếu protocol phù hợp, hoặc không phục vụ câu hỏi chính của Internship 1).

- [x] **C19. Làm rõ lý do chọn/không chọn dataset trong §2.2**
  - Output: Bảng dataset trong §2.2 có cột vai trò/lý do; sau bảng có đoạn tổng hợp giải thích vì sao 4 dataset reproduce là đủ cho mục tiêu hiện tại nhưng chưa bao trùm toàn bộ landscape.
  - Yêu cầu: tránh viết như thể ASVspoof 2019/2021/5 + In-the-Wild là toàn bộ SDD; dùng wording "đại diện cho bốn stress test chính trong phạm vi báo cáo".

- [x] **C20. Refresh literature review architectures theo taxonomy ít nhóm ở highest level**
  - Output: `report/notes/architectures.md` được tổ chức lại theo vài nhóm lớn trước, trong mỗi nhóm mới chia nhánh nhỏ: (i) signal/task-specific supervised detectors, (ii) raw waveform/end-to-end detectors, (iii) SSL/foundation front-end + back-end, (iv) system-level robustness/fusion.
  - Yêu cầu: không mở đầu bằng danh sách dài model; tên paper/model cụ thể chỉ xuất hiện sau khi đã giải thích nguyên lý của nhóm.

- [x] **C21. Bổ sung các hướng architecture/paper nổi bật 2024-2026 vào §2.3**
  - Output: §2.3 có nhắc rõ WavLM+MFA, WavLM back-ends/fusion trong ASVspoof 5, Nes2Net, AASIST3, codec-aware/CodecFake-aware training, Whisper-based approach ở mức emerging.
  - Yêu cầu: giữ trọng tâm vào nhóm SSL + back-end vì liên quan trực tiếp đến 6 mô hình reproduce, nhưng không bỏ qua fusion/calibration và codec-aware training vì đây là đặc điểm của hệ thống mạnh gần đây.

- [x] **C22. Thêm bảng paper/dataset đại diện vào literature review**
  - Output: §2.2/§2.3 hoặc đoạn cuối §2.3 có bảng ngắn liệt kê paper/dataset/model đại diện, vai trò của chúng, và lý do có/không đưa vào reproduce.
  - Yêu cầu: bảng phục vụ review nhanh, không biến Chương 2 thành bibliography dài.

- [x] **C23. Cập nhật references Chương 2 cho các nguồn mới**
  - Output: `Tài liệu tham khảo` có thêm ASVspoof 5 design/evaluation, ADD 2022/2023, PartialSpoof, CodecFake, CodecFake+, EchoFake, WavLM+MFA, WavLM back-ends, WavLM ensemble, Whisper+AASIST/Scalable AASIST nếu được nhắc trong body.
  - Yêu cầu: ưu tiên paper gốc/arXiv/ISCA/IEEE; chỉ dùng nguồn dataset page khi paper chưa đủ thông tin.

---

## D. Chương 3 — Reproduce và phân tích kết quả

- [x] **D1. §3.1 Setup thí nghiệm**
  - Output: mô tả high-level cách reproduce: Kaggle (P100/T4), single notebook `survey.ipynb` per training/inference, các pretrained checkpoints lấy từ đâu (HuggingFace, repo gốc), datasets từ Kaggle public datasets nào.
  - Yêu cầu: KHÔNG đi quá low level (không paste code), tập trung vào reproducibility — người đọc cần biết "có thể chạy lại được" và "muốn mở rộng sang model/dataset mới thì làm sao".
  - Đề cập folder structure: `notebooks/eval_<dataset>.ipynb`, `results/<dataset>/results.pkl`, `results/checkpoints/`.

- [x] **D2. §3.2 Bảng EER tổng hợp 6 models × 4 datasets**
  - Output: bảng markdown với rows = 6 models (AASIST, AASIST-L, AASIST3, LFCC+LCNN, XLS-R+AASIST, XLS-R+Nes2Net), cols = 4 datasets (ASV19 LA, ASV21 DF, ASV5, In-the-Wild).
  - Số liệu: lấy trực tiếp từ `results/README.md` (đã có sẵn). Đánh dấu ô "pending" (XLS-R+AASIST trên ASV21 DF).
  - Sau bảng: 2-3 đoạn nhận xét high-level **theo phong cách mô tả số liệu, không kết luận mạnh**. Ví dụ: "kết quả ban đầu cho thấy xu hướng EER tăng từ ASV19 → ASV21/ASV5/ITW", "các model dùng SSL front-end (XLS-R) duy trì EER thấp hơn trên các dataset khó", "LFCC+LCNN có EER tăng đáng kể trên các điều kiện codec/in-the-wild". Tránh các từ như "sụp đổ", "tốt nhất", "thất bại" khi chưa có error analysis chi tiết.

- [x] **D3. §3.3 Phân tích EER theo từng dataset (overview)**
  - Output: mỗi dataset 1 đoạn ngắn (~100 từ): mô tả tính chất dataset → EER ranking → giải thích sơ bộ tại sao một số model fail.
  - Yêu cầu: giữ ở mức overview, không đi vào confusion matrix / score distribution — đó là việc của §3.4.
  - Source: `results/error_analysis/synthesis/eer_table.csv` + `results/README.md`.

- [x] **D4. §3.4 Error analysis trên ASVspoof 5**
  - Output: §3.4 đã được cập nhật bằng kết quả ASVspoof 5 Track 1 eval split (680,774 utterance) với phân tích score/EER, attack, codec, confident errors và failure overlap.
  - Yêu cầu còn lại: nếu cần cô lập đóng góp của SSL front-end, vẫn nên bổ sung model Nes2Net không có XLS-R ở Internship 2 hoặc phần future work; không bắt buộc để hoàn thiện Ch3 Internship 1.

- [x] **D5. §3.5 Nhận xét cross-dataset (ngắn)**
  - Output: 1-2 đoạn mô tả generalization gap (EER delta giữa ASV19 và ITW/ASV5), nhận xét pattern theo phong cách quan sát: "các model dựa trên SSL front-end có gap nhỏ hơn", "model dùng hand-crafted feature có gap lớn hơn rõ rệt".
  - Yêu cầu: dùng ngôn ngữ mô tả số liệu, tránh kết luận tuyệt đối khi chưa có error analysis đầy đủ trên cả 4 datasets.
  - Source: `results/error_analysis/synthesis/generalization_gap.csv`.

---

## E. Chương 4 — Kết luận và Future work

- [ ] **E1. §4.1 Tóm tắt kết quả**
  - Output: 1 trang tóm tắt: đã survey gì, reproduce gì, phát hiện chính từ EER + error analysis sơ bộ.

- [ ] **E2. §4.2 Đề xuất hướng tiếp cận (dạng ý niệm)**
  - Output: dựa trên error analysis ASV5 ở §3.4, đề xuất 1-2 hướng improvement (ví dụ: codec augmentation, ensemble SSL+hand-crafted, fine-tune SSL trên multi-codec). Mỗi hướng: motivation từ error pattern, plan thử nghiệm cao cấp, expected outcome.
  - Yêu cầu: ý niệm thôi, chi tiết sẽ ở Internship 2 (đề cương luận văn).

- [ ] **E3. §4.3 Future work**
  - Output: liệt kê các bước Internship 2 sẽ làm: implement đề xuất, evaluate, analyze; cân nhắc mở rộng sang dataset mới sau khi verify nguồn chính thức; thử các SSL khác (WavLM, multi-lingual XLS-R).

---

## F. Phần đầu/cuối báo cáo

- [ ] **F1. Tóm tắt nội dung (abstract, ~250 từ)** — viết SAU CÙNG khi 4 chương đã xong.
- [ ] **F2. Lời cam đoan / Lời ngỏ** — bỏ trống, user tự điền.
- [ ] **F3. Tài liệu tham khảo** — collect dần khi viết các chương; format theo IEEE hoặc APA.

---

## G. Việc chạy thực nghiệm song song (user thực hiện)

- [x] **G1. Eval ASVspoof 5 trên eval track (Track 1)** — đã có `results/asvspoof5/results.pkl` với 680,774 utterance và đủ 6 models.
- [ ] **G2. Train + eval Nes2Net (không XLS-R) trên ASV5 [deferred]** — chuyển sang Internship 2 nếu muốn cô lập hiệu quả của SSL frontend; không còn chặn Ch3 Internship 1.
- [!] **G3. Eval XLS-R+AASIST trên ASV21 DF** — ASV21 DF vẫn pending; In-the-Wild đã có kết quả 10.91% EER.
- [x] **G4. Chạy lại error analysis trên ASV5 → fill §3.4** — đã có `results/error_analysis/asvspoof5/` và §3.4 đã được cập nhật.

---

## Thứ tự đề xuất (lần đầu)

1. A1, A2 (tạo skeleton)
2. B1, B2, B3 (Ch1 — không cần research nặng)
3. C1, C2, C3 (research notes — có thể delegate cho Agent, song song)
4. C4-C8 (Ch2 — chắt lọc từ notes)
5. D1, D2, D3, D4, D5 (Ch3 — số liệu ASV5 eval và error analysis đã có)
6. E1, E2, E3
7. F1 (cuối cùng)

Mỗi lần load lại session, chỉ cần: "tiếp tục TODO `<id>` trong `report/TODO.md`".
