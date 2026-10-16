# Video Sub VN — Phụ đề Karaoke tiếng Việt (burn-in MP4)

Ứng dụng desktop Windows: nhận dạng giọng nói tiếng Việt bằng faster-whisper,
highlight **từng từ đang nói màu vàng** (chữ thường màu trắng, viền đen),
rồi burn cứng phụ đề vào MP4 bằng FFmpeg + libass.

## Tính năng

- Chọn video → hiện tên file, phân giải, thời lượng.
- Nhận dạng tiếng Việt (`language="vi"`, `word_timestamps=True`), có VAD tùy chọn.
- Timestamp từng từ + làm mịn khoảng lặng (không highlight khi im lặng).
- Khung **soát lỗi chính tả** trước khi render; sửa text không cần chạy lại Whisper
  (mốc thời gian được chia lại theo tỉ lệ trong đoạn).
- Tùy chỉnh: font, cỡ chữ, màu chữ/màu highlight (mặc định trắng/vàng),
  viền đen, vị trí, cách mép dưới, số từ/dòng.
- Render MP4 burn-in (libx264 + AAC, giữ nguyên phân giải/audio gốc),
  kèm file `.srt` + JSON timestamps + file `.ass` trung gian.
- Chạy nền bằng QThread/worker — GUI không đơ; có nút Hủy.

## Cài đặt (Windows)

1. Cài Python 3.11 (64-bit) + FFmpeg full:
   - `winget install Gyan.FFmpeg` (gồm cả `ffmpeg` và `ffprobe`), hoặc tải tại
     https://www.gyan.dev/ffmpeg/builds/ rồi thêm thư mục `bin` vào PATH.
2. Mở terminal trong thư mục dự án:
   ```bat
   setup.bat
   ```
   hoặc thủ công:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Kiểm tra: `ffmpeg -version` và `ffprobe -version` phải chạy được.

### GPU NVIDIA (tùy chọn)

- Mặc định chế độ `auto`: thử CUDA (`float16`), lỗi thì tự về CPU (`int8`).
- Để dùng GPU cần cuDNN/cuBLAS tương thích với `ctranslate2`; nếu báo lỗi CUDA,
  chọn `cpu` trong giao diện — vẫn chạy bình thường, chỉ chậm hơn.

### Model & cache

- faster-whisper tự tải model (`tiny`→`large-v3`, mặc định `medium`) từ Hugging Face
  về cache (thường `%USERPROFILE%\.cache\huggingface`).
- Model `medium` ≈ 1.5 GB; lần đầu cần mạng, lần sau chạy offline.
- Audio trung gian 16 kHz lưu ở `%TEMP%\subvn_cache` (tự tạo).

## Chạy

```bat
run.bat
```
hoặc:
```bat
.venv\Scripts\activate
python main.py
```

Quy trình: **Chọn video → Bắt đầu nhận dạng → Soát lỗi → Render video karaoke**.

## Cách highlight từng từ (quan trọng)

- Không dùng karaoke `\k/\kf` tràn (nó giữ màu các từ đã nói).
- Với mỗi đoạn, sinh **một event ASS cho mỗi khoảng thời gian của từng từ**:
  trong event đó chỉ từ đang nói được bọc `{\1c<yellow>}...{\1c<white>}`,
  các từ khác giữ trắng. Khoảng lặng ngắn (≤ `silence_hide_s`, mặc định 0.6s)
  hiện cả dòng trắng; khoảng lặng dài thì **không hiện gì** (tránh phụ đề ma).
- Vì vậy từ trước đó **trở về trắng** sau khi nói xong, đúng yêu cầu.

## Căn chỉnh (alignment) tiếng Việt — đánh giá trung thực

- Các forced-aligner phổ biến (whisperx, wav2vec2-large-960h) **không hỗ trợ
  tiếng Việt** (chỉ en/es/fr/de/...) nên dự án **không phụ thuộc** chúng.
- Pipeline dùng word timestamps của faster-whisper (có hỗ trợ `vi`) +
  `core/alignment.py`: sắp xếp, tách chồng lấn, vá khe micro-gap ≤ 0.15s,
  giữ nguyên khoảng lặng lớn. Muốn cắm aligner Việt thật sau này thì
  implement `align_with_external()` — không cần đổi GUI/renderer.

## Chẩn đoán render chậm / treo (vụ 20s video kẹt 2183s)

Nguyên nhân tìm thấy khi đọc code (không đoán mò):

1. **Deadlock pipe stderr (nghi phạm chính gây treo vô hạn):** `render_burned`
   và `extract_audio` cũ gọi ffmpeg với `stderr=PIPE` nhưng **không bao giờ đọc**
   trong lúc chạy. FFmpeg log ra stderr; khi bộ đệm pipe (~64KB) đầy — rất dễ
   xảy ra nếu libass spam cảnh báo `Glyph ... not found` theo từng frame khi
   font thiếu chữ Việt — ffmpeg kẹt ở `write()` mãi mãi, GUI đợi mãi mãi.
   → Đã sửa: `core/ffmpeg_runner.py` xả stderr bằng thread riêng ra
   `logs/ffmpeg_<ts>.log`, pipe không bao giờ đầy.
2. **Encode chậm:** preset `medium` hardcoded + luôn re-encode AAC.
   → Mặc định `veryfast` (tùy chọn ultrafast/fast/medium), copy audio (`-c:a copy`)
   khi codec gốc tương thích MP4 (aac/mp3/ac3).
3. **Tiến trình giả:** vòng render cũ chỉ hiển thị "heartbeat" theo thời gian tường,
   không phản ánh thực tế. → Dùng `-progress pipe:1` của ffmpeg: frame, fps,
   `out_time`, speed, dung lượng hiện live trên GUI.
4. **Tải lại model mỗi lần chạy, không cache transcript.**
   → Model cache toàn tiến trình theo (model, device); transcript cache theo
   hash (path, size, mtime, model, vad) — đổi style không chạy lại Whisper.
5. **Không có timing/log theo stage** nên không biết kẹt ở đâu.
   → Mỗi run ghi `logs/transcribe_<ts>.jsonl/.log` và `logs/render_<ts>.jsonl/.log`;
   GUI hiện thời gian thực từng stage + nút "Mở log".

## Sửa lỗi "render xong nhưng không thấy chữ"

Nguyên nhân tìm thấy khi đọc code:

1. **File ASS ghi kèm BOM (`utf-8-sig`):** một số bản libass không nhận ra header
   `[Script Info]` khi có BOM nên parse được 0 event → video render bình thường
   nhưng không có chữ, **không báo lỗi**. → Đã sửa: ghi UTF-8 thuần (`core/subtitle_renderer.py`).
2. **Không kiểm tra ASS trước render:** file rỗng, event ngoài thời lượng video,
   mất tag highlight đều cho ra video "sạch chữ". → Đã thêm `validate_ass()`:
   lỗi fatal (không Dialogue, timestamp ngoài video, mất màu vàng...) thì **dừng
   render ngay** với thông báo rõ; font quá nhỏ/margin quá lớn chỉ ghi cảnh báo
   vào log render.
3. **Không kiểm tra ffmpeg có libass:** GUI cũ chỉ cảnh báo. → `assert_subtitles_filter()`
   chặn render ngay nếu bản FFmpeg thiếu filter `subtitles`, kèm hướng dẫn cài bản full.
4. **Escape đường dẫn filter thiếu `;[]`:** dấu `;` trong tên thư mục sẽ cắt
   filtergraph và lặng lẽ bỏ filter phụ đề. → Đã escape đầy đủ.
5. **Exit code 0 ≠ có chữ:** thêm bước validate sau render — so sánh frame nguồn
   và frame output tại giữa-các-từ (pixel trắng/vàng ở dải phụ đề), verdict
   `PASS/FAIL/INCONCLUSIVE` + lưu ảnh PNG preview vào `previews/` để mắt thường
   kiểm tra. Verdict `FAIL` chặn ngay với hướng dẫn xử lý.

Bổ sung xác minh màu vàng: `check_karaoke_colors()` đếm pixel vàng/trắng ở dải
phụ đề trên frame nguồn vs output tại giữa-các-từ — `FAIL` (chữ hiện mà không
có vàng) chặn render ngay. Câu lệnh FFmpeg chính xác được ghi dòng `[CMD]` đầu
mọi `logs/render_*.log` và `logs/diag_*.log`. Script chẩn đoán tối thiểu:
`python tools/diagnose_burnin.py` (dựng video đen 6s + ASS thật → in số liệu
pixel từng từ + lưu PNG preview, exit 0 = đã chứng minh highlight vàng trên frame).

Chạy regression test: `pytest -q` (`test_ass_validation.py` chạy thuần Python;
`test_burnin.py` dựng video đen 5s + ASS tiếng Việt, burn thật và check frame,
tự skip khi thiếu ffmpeg/libass).

## Chế độ test nhanh

- Tích "Chỉ xử lý đoạn đầu video" + chọn N giây (mặc định 20s): chỉ trích/render
  N giây đầu (`-t`), vẫn giữ highlight vàng đúng timeline.
- "Dùng lại transcript đã cache": bỏ qua Whisper/alignment khi video+model không
  đổi; GUI đánh dấu stage bị bỏ qua (⏭ skipped) thay vì báo xong giả.
- Dùng preset `veryfast`/`ultrafast` khi lặp style.

## Kiểm thử

```bat
pytest -q
```

`tests/test_subtitles.py` bao phủ: Unicode/dấu tiếng Việt, timestamp âm,
thứ tự thời gian, chồng lấn, khoảng lặng dài, highlight vàng di chuyển theo
từng từ, wrap dòng, header ASS portrait 9:16, path Windows, video không lời.

> Trạng thái xác minh: terminal thực thi trong môi trường này bị lỗi
> (`spawn EFTYPE`) nên **pytest chưa chạy được ở đây** — hãy chạy `pytest -q`
> trên máy Windows thật và báo kết quả. Logic renderer/alignment là hàm thuần
> nên dễ kiểm thử độc lập.

## Giới hạn đã biết

- Word timestamps của Whisper không hoàn hảo tuyệt đối; hãy soát lỗi + nghe lại.
- Video không có audio track → báo lỗi rõ ràng, không sinh phụ đề giả.
- VAD quá mạnh có thể cắt tiếng nhỏ — tắt checkbox VAD nếu mất chữ.
- Cần FFmpeg bản **full** có filter `subtitles` (libass); bản thiếu sẽ báo lỗi khi render.
- Chưa test end-to-end GPU/CPU render video thật trong môi trường này
  (không có terminal/media tools) — xem `pytest` + chạy thử 1 video ngắn.
