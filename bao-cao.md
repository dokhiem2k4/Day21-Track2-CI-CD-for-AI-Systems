# Báo Cáo Lab MLOps — Day 21: CI/CD cho AI Systems

**Sinh viên:** Đỗ Minh Khiêm - 2A202600463  
**Repo:** https://github.com/dokhiem2k4/Day21-Track2-CI-CD-for-AI-Systems  
**Ngày:** 07/05/2026

---

## 1. Bộ Siêu Tham Số Đã Chọn (Kết Quả Bước 1)

Qua 5 thí nghiệm trên MLflow UI với tập `train_phase1.csv` (2998 mẫu), bộ tham số cuối cùng được chọn:

| Tham số | Giá trị | Lý do chọn |
|---|---|---|
| `model_type` | `random_forest` | Ổn định nhất trên dữ liệu tabular so với GBT và LR |
| `n_estimators` | 500 | Tăng từ 100 → 200 → 500; độ chính xác tăng dần, không overfitting |
| `max_depth` | 25 | Độ sâu lớn hơn cho phép học được pattern phức tạp của dữ liệu rượu vang |
| `min_samples_split` | 2 | Giá trị mặc định; thay đổi không cải thiện đáng kể |
| `class_weight` | `balanced` | Phân phối nhãn lệch (lớp 1 chiếm ~60%), `balanced` cải thiện F1 |

**Kết quả tốt nhất:** `accuracy = 0.758`, `f1_score ≈ 0.751` trên tập eval (500 mẫu).  
Ngưỡng đánh giá của pipeline là **0.70** — bộ tham số này vượt qua với biên an toàn.

---

## 2. Khó Khăn Gặp Phải và Cách Giải Quyết

### 2.1 Deploy step bị treo vô hạn

**Vấn đề:** `sudo systemctl restart mlops-serve` trong SSH session khiến process uvicorn mới kế thừa file descriptor của SSH, giữ session mở mãi không thoát.

**Giải quyết:** Bỏ hoàn toàn SSH trong Deploy job, thay bằng vòng lặp `curl` từ runner kiểm tra trực tiếp endpoint `/health` trên port 8000. Model đã được upload lên GCS trong Train job; systemd với `Restart=always` tự khởi động lại service.

### 2.2 Sklearn version mismatch gây load model 8 phút

**Vấn đề:** `requirements.txt` pin `scikit-learn==1.4.2` nhưng VM có `scikit-learn==1.7.2` (do `pip install mlflow` upgrade). Khi service load `model.pkl`, mỗi cây quyết định phải qua cảnh báo unpickle → 500 cây × vài giây = ~8 phút.

**Giải quyết:** Bỏ pin version (`scikit-learn` không có `==`), để GitHub Actions runner cài cùng version với VM. Model được train và serve trên cùng version → load tức thì.

### 2.3 MLflow remote tracking (Bonus 1)

**Vấn đề:** Artifact logging (`mlflow.sklearn.log_model`) thất bại vì artifact root là đường dẫn local trên VM, GitHub Actions runner không thể ghi.

**Giải quyết:** Chỉ log params và metrics (không log artifact). Model được lưu riêng qua `joblib.dump` + upload lên GCS — đây là cách vận hành thực tế hơn.

### 2.4 Eval gate thất bại ở Bước 3

**Vấn đề:** Sau khi gộp thêm 2998 mẫu `train_phase2.csv`, accuracy ban đầu chỉ đạt ~0.68 (dưới ngưỡng 0.70).

**Giải quyết:** Tune lại params (tăng `n_estimators` và `max_depth`, thêm `class_weight=balanced`) để đưa accuracy lên 0.758, đảm bảo eval gate luôn qua ở cả Bước 2 và Bước 3.

---

## 3. Kết Quả Đạt Được

| Hạng mục | Kết quả |
|---|---|
| MLflow UI ≥ 3 runs | ✅ 5 runs với params và metrics đầy đủ |
| 4 GitHub Actions jobs xanh | ✅ Unit Test → Train → Eval → Deploy |
| Eval gate (accuracy ≥ 0.70) | ✅ accuracy = 0.758 |
| VM `/health` | ✅ `{"status": "ok"}` |
| VM `/predict` | ✅ `{"prediction": 0, "label": "thap"}` |
| Bước 3 tự động hóa | ✅ Commit `.dvc` kích hoạt toàn pipeline |
| Bonus 1 — MLflow remote | ✅ Tracking lên MLflow server trên GCE VM |
| Bonus 2 — Multi-algorithm | ✅ RF, GradientBoosting, LogisticRegression |
| Bonus 3 — Performance report | ✅ `report.txt` (confusion matrix + classification report) |
| Bonus 4 — Rollback | ✅ So sánh old vs new accuracy, hủy deploy nếu model mới kém hơn |
| Bonus 5 — Data drift warning | ✅ Cảnh báo nếu lớp < 10%, ghi `class_distribution` vào metrics.json |
