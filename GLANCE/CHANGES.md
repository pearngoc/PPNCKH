# Nhật ký thay đổi — TabGLANCE & TabCF Integration

Tài liệu này ghi lại tất cả các thay đổi được thực hiện trong quá trình tích hợp TabCF vào framework GLANCE và bổ sung các metric chất lượng mới.

---

## Tổng quan

Mục tiêu chính của các thay đổi:
1. Tích hợp TabCF (VAE-based CF generator) vào GLANCE dưới dạng một `local_cf_generator` thay thế
2. Đảm bảo các feature bất biến (immutable) không bị thay đổi trong kết quả recourse
3. Bổ sung ba metric chất lượng mới để so sánh TabGLANCE vs GLANCE baseline
4. Lưu kết quả thí nghiệm ra CSV để so sánh

---

## Chi tiết thay đổi theo file

### `src/glance/local_cfs/tabcf_method.py`

**Vấn đề:** Sau khi tối ưu hoá trong không gian latent và giải mã ngược về không gian feature, các feature bất biến (ví dụ: `age`, `sex`, `race`) vẫn bị thay đổi bởi decoder của VAE. Tham số `feat_to_vary` được lưu lại nhưng chưa được áp dụng.

**Sửa đổi — áp dụng `feat_to_vary` sau khi decode:**

```python
# Trước (feat_to_vary không được dùng)
cf_row = self._decode_z_to_row(z_cf.detach())
if cf_row is not None:
    all_rows.append(cf_row)

# Sau (snap các feature bất biến về giá trị gốc)
cf_row = self._decode_z_to_row(z_cf.detach())
if cf_row is not None:
    if self.feat_to_vary != "all":
        for col in self.num_cols + self.cat_cols:
            if col not in self.feat_to_vary:
                cf_row[col] = row[col]
    all_rows.append(cf_row)
```

**Lý do:** TabCF tối ưu trong không gian latent liên tục bằng Adam optimizer — không có cơ chế mask feature trực tiếp. Cách duy nhất để đảm bảo immutability là ghi đè sau khi decode.

---

### `src/glance/glance/glance.py`

**Thay đổi 1 — Xoá `latent_heuristic_weight`:**

Tham số `latent_heuristic_weight` được thêm vào trước đó cho một hướng thử nghiệm khác (latent-space merging). Đã xoá khỏi:
- `__init__` signature
- `self.latent_heuristic_weight = ...` assignment
- Lời gọi `_find_candidate_clusters(..., latent_weight=self.latent_heuristic_weight)`

Hàm `_find_candidate_clusters` giữ nguyên tham số `latent_weight` với default = 0.0, không ảnh hưởng đến các chạy thí nghiệm hiện tại.

**Thay đổi 2 — Ngưỡng hiển thị thay đổi feature:**

```python
# Trước (in ra +0.00 với residual từ gradient descent như 3.2e-09)
if value[0] > 0:
    output_string += f"... +{value[0]} ..."
elif value[0] < 0:
    output_string += f"... {value[0]} ..."

# Sau
if value[0] > 1e-6:
    output_string += f"... +{value[0]} ..."
elif value[0] < -1e-6:
    output_string += f"... {value[0]} ..."
```

**Lý do:** TabCF tối ưu bằng gradient descent nên kết quả có thể có residual nhỏ (ví dụ: `3.207e-09`) — những giá trị này in ra dưới dạng `+0.00` gây hiểu nhầm là feature bị thay đổi trong khi thực tế không có.

---

### `src/glance/metrics/recourse_metrics.py` *(file mới)*

Ba metric chất lượng mới được thêm vào:

**`feasibility_score(action, train_df, num_cols, cat_cols)`**
- Đo mức độ thực tế của thay đổi số so với phân phối training
- Công thức: `1 / (1 + |delta| / std_train)` cho mỗi feature số
- Thay đổi categorical luôn được gán điểm 1.0
- Trả về trung bình trên tất cả feature thay đổi; 0.0 nếu không có feature nào thay đổi

**`dominant_feature_concentration(action, train_df, num_cols, cat_cols)`**
- Đo mức độ một feature chiếm hầu hết thay đổi trong action
- Chuẩn hoá delta số theo range training; categorical đóng góp 1.0
- Công thức: `max(contributions) / sum(contributions)`
- Giá trị gần 1.0 cho thấy một feature đang bị khai thác (ví dụ: capital-gain)

**`action_diversity(actions, num_cols, cat_cols)`**
- Đo sự khác biệt giữa K action toàn cục
- Công thức: `1 - mean pairwise Jaccard similarity`
- 0.0 = tất cả action thay đổi cùng feature; 1.0 = tất cả hoàn toàn khác nhau

---

### `src/glance/metrics/__init__.py`

Bổ sung export cho ba metric mới:

```python
from .recourse_metrics import feasibility_score, dominant_feature_concentration, action_diversity
```

---

### `run_tabglance_adult.py`

**Thay đổi 1 — Định nghĩa feature bất biến:**

```python
IMMUTABLE = {'age', 'native.country', 'race', 'sex', 'marital.status'}
feat_to_vary = [c for c in num_cols + cat_cols if c not in IMMUTABLE]
```

`feat_to_vary` được truyền vào cả `tabcf.fit()` và `glance.fit()`.

**Thay đổi 2 — Tính toán và in metric mới:**

Sau `explain_group()`, tính:
- Feasibility Score trung bình trên tất cả action
- Dominant Feature Concentration trung bình
- Action Diversity

**Thay đổi 3 — Lưu kết quả ra CSV:**

Kết quả mỗi lần chạy được append vào `examples/results.csv` với các cột:
`timestamp`, `method`, `generator`, `dataset`, `initial_clusters`, `final_clusters`, `n_local_cfs`, `effectiveness`, `mean_cost`, `feasibility_score`, `dominant_concentration`, `action_diversity`

**Thay đổi 4 — Xoá các tham số CLI không còn dùng:**

Đã xoá `--algo` và `--eff-threshold` khỏi argparser, và các tham số tương ứng khỏi `glance.fit()` để đảm bảo cấu hình nhất quán với `examples/script.py`.

---

### `run_glance_baseline_adult.py`

**Thay đổi — Tính toán metric và lưu CSV:**

Tương tự `run_tabglance_adult.py`, bổ sung:
- Tính ba metric mới sau `explain_group()`
- Lưu kết quả ra `examples/results.csv` với cùng schema
- `method='GLANCE'`, `generator=args.generator` (NearestNeighbors / RandomSampling / Dice)

---

### `examples/script.py`

**Thay đổi 1 — Bổ sung metric mới vào k-fold experiment:**

Trong vòng lặp k-fold của nhánh GLANCE, tính thêm ba metric sau mỗi fold:

```python
train_ref = data.drop(columns=[target_name])
fold_actions = [stats['action'] for stats in clusters_res.values()]
fold_feasibility  = float(np.mean([feasibility_score(...) for a in fold_actions]))
fold_concentration = float(np.mean([dominant_feature_concentration(...) for a in fold_actions]))
fold_diversity    = action_diversity(fold_actions, _numfeats, _catfeats)
```

Kết quả các fold được tổng hợp dưới dạng `mean ± std`.

**Thay đổi 2 — Xoá các cột `IM__*` khỏi CSV:**

Các tham số nội bộ (`IM__cluster_action_choice_algo`, `IM__nns__n_scalars`, v.v.) không còn xuất hiện trong file kết quả CSV. Chỉ giữ lại 8 cột tham số chính:
`dataset`, `model`, `method`, `local_cf_generator`, `clustering_method`, `n_initial_clusters`, `n_final_clusters`, `n_local_counterfactuals`

**Thay đổi 3 — Đường dẫn output mặc định:**

```python
# Trước: không có default, bắt buộc truyền -o
# Sau
default="/Users/ngocle/Projects/Learning/GLANCE/examples/results.csv"
```

**Thay đổi 4 — Cập nhật return và unpack 7 giá trị:**

```python
# Trước: 4 giá trị
return eff, mean_cost, size, total_time

# Sau: 7 giá trị
return eff, mean_cost, size, total_time, feasibility, concentration, diversity
```

---

## Tóm tắt các file bị thay đổi

| File | Loại thay đổi |
|------|--------------|
| `src/glance/local_cfs/tabcf_method.py` | **Sửa lỗi** — áp dụng `feat_to_vary` sau decode |
| `src/glance/glance/glance.py` | **Sửa lỗi** — ngưỡng 1e-6, xoá `latent_heuristic_weight` |
| `src/glance/metrics/recourse_metrics.py` | **Mới** — 3 metric chất lượng |
| `src/glance/metrics/__init__.py` | **Cập nhật** — export metric mới |
| `run_tabglance_adult.py` | **Cập nhật** — IMMUTABLE, metric, CSV |
| `run_glance_baseline_adult.py` | **Cập nhật** — metric, CSV |
| `examples/script.py` | **Cập nhật** — metric k-fold, xoá IM__ columns, default path |

---

## Lưu ý khi chạy thí nghiệm

```bash
# Chạy TabGLANCE
cd /Users/ngocle/Projects/Learning/GLANCE
/opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_tabglance_adult.py

# Chạy GLANCE baseline với từng generator
/opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_glance_baseline_adult.py --generator NearestNeighbors
/opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_glance_baseline_adult.py --generator RandomSampling
/opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_glance_baseline_adult.py --generator Dice

# Kết quả tất cả được lưu vào
examples/results.csv
```
