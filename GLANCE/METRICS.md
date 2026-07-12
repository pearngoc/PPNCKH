# Các Metric Đánh Giá Chất Lượng Recourse

Tài liệu này mô tả chi tiết 5 metric được dùng để đánh giá các phương pháp algorithmic recourse toàn cục (GLANCE, TabGLANCE) trên bộ dữ liệu Adult income.

**Bối cảnh bài toán:** Cho một mô hình phân loại dự đoán thu nhập (≤50K hoặc >50K), mục tiêu là tìm ra các *action* (can thiệp) tối thiểu để thay đổi dự đoán của mô hình từ `≤50K` sang `>50K` cho một nhóm người dùng bị ảnh hưởng.

---

## 1. Effectiveness (Hiệu quả)

### Định nghĩa

Tỷ lệ các instance bị ảnh hưởng (được dự đoán `income ≤ 50K`) mà dự đoán của mô hình chuyển sang `income > 50K` khi áp dụng action tương ứng.

```
Effectiveness = n_flipped / n_affected
```

- `n_flipped`: số instance mà sau khi áp dụng action, mô hình đổi dự đoán thành `>50K`
- `n_affected`: tổng số instance cần giải thích

### Cách tính trong GLANCE

GLANCE tạo ra K action toàn cục (mỗi action cho một cluster). Mỗi instance được gán vào cluster gần nhất và áp dụng action của cluster đó. Sau đó, mô hình black-box được gọi lại để kiểm tra dự đoán mới.

### Ví dụ minh hoạ

Giả sử có 100 instance bị ảnh hưởng, GLANCE tạo ra 3 action:

| Cluster | Số instance trong cluster | Số instance được flip |
|---------|--------------------------|----------------------|
| Action 1: tăng `education-num` +3, `hours-per-week` +10 | 40 | 38 |
| Action 2: thay đổi `workclass` → Self-emp, `education-num` +2 | 35 | 30 |
| Action 3: tăng `hours-per-week` +15, `occupation` → Exec-managerial | 25 | 22 |

```
Global Effectiveness = (38 + 30 + 22) / 100 = 90/100 = 0.90 = 90%
```

Nghĩa là 90 trong 100 người được tư vấn bởi GLANCE sẽ nhận được dự đoán >50K nếu họ thực hiện action của cluster mình.

### Giới hạn

Effectiveness đo *liệu mô hình có đổi quyết định không* — không đo *liệu action đó có khả thi với người thực không*. Một action như "tăng `capital-gain` thêm $42.000" có thể đạt Effectiveness = 100% nhưng hoàn toàn không thực tế.

---

## 2. Mean Recourse Cost (Chi phí recourse trung bình)

### Định nghĩa

Khoảng cách trung bình giữa instance gốc và counterfactual tương ứng, chỉ tính trên các instance được flip thành công.

```
Mean Cost = Σ dist(x_i, cf_i) / n_flipped
```

### Cách tính khoảng cách

Khoảng cách giữa instance gốc `x` và counterfactual `cf` được tính theo từng feature, sử dụng chuẩn hoá **MAD** (Median Absolute Deviation):

```
dist(x, cf) = Σ_j  |x_j - cf_j| / MAD_j        (feature số)
            + Σ_j  𝟙[x_j ≠ cf_j]               (feature categorical)
```

Trong đó `MAD_j = median(|x_j - median(x_j)|)` tính trên tập training — đây là phép chuẩn hoá robust với outlier, đảm bảo một thay đổi lớn trên feature có phân phối hẹp bị phạt nặng hơn.

### Ví dụ minh hoạ

Giả sử hai instance được flip bởi Action 1:

**Instance A** (gốc → counterfactual):
- `education-num`: 9 → 12 (delta = +3)
- `hours-per-week`: 35 → 45 (delta = +10)

**Instance B** (gốc → counterfactual):
- `education-num`: 10 → 12 (delta = +2)
- `hours-per-week`: 40 → 45 (delta = +5)

Giả sử `MAD(education-num) = 2.0`, `MAD(hours-per-week) = 8.0`:

```
dist(A) = 3/2.0 + 10/8.0 = 1.50 + 1.25 = 2.75
dist(B) = 2/2.0 + 5/8.0  = 1.00 + 0.625 = 1.625

Mean Cost (Action 1) = (2.75 + 1.625) / 2 = 2.19
```

> Cost thấp = action chỉ yêu cầu thay đổi nhỏ, gần với trạng thái hiện tại của người dùng.

### Giới hạn

Cost thấp không đồng nghĩa với action dễ thực hiện. Ví dụ, `capital-gain` thay đổi $40.000 có thể cho cost thấp nếu MAD của `capital-gain` lớn (do nhiều người có capital-gain rất cao hoặc rất thấp) — nhưng thực tế hầu hết người không thể tăng capital-gain tùy ý.

---

## 3. Feasibility Score (Điểm khả thi)

### Định nghĩa và động lực

Cost đo *khoảng cách tương đối* theo MAD, không đo *mức độ khả thi tuyệt đối*. Feasibility Score bổ sung chiều này: một thay đổi `+δ` trên một feature được coi là khả thi nếu `δ` nằm trong khoảng biến động tự nhiên của feature đó trong tập training (đo bằng độ lệch chuẩn `std`).

```
feasibility(feature_j) = 1 / (1 + |δ_j| / std_j)
```

```
Feasibility Score (action) = (1/K) × Σ_{j ∈ changed} feasibility(feature_j)
```

Trong đó:
- `δ_j` = mức thay đổi action đề xuất cho feature j
- `std_j` = độ lệch chuẩn của feature j trên tập training
- `K` = số feature thay đổi trong action
- Feature categorical: luôn trả về `1.0` (chuyển sang một giá trị hợp lệ khác là khả thi)

### Tính chất của công thức

| Trường hợp | Kết quả |
|------------|---------|
| `δ = 0` (không thay đổi) | feature bị bỏ qua, không đưa vào tính trung bình |
| `δ = std` (thay đổi đúng 1 std) | `1 / (1 + 1) = 0.5` |
| `δ = 0.1 × std` (thay đổi rất nhỏ) | `1 / (1 + 0.1) ≈ 0.91` |
| `δ = 5 × std` (thay đổi rất lớn) | `1 / (1 + 5) ≈ 0.17` |
| `δ → ∞` | `→ 0` |

### Ví dụ minh hoạ (bộ dữ liệu Adult)

Phân phối tập training:

| Feature | Mean | Std |
|---------|------|-----|
| `age` | 38.6 | 13.6 |
| `education-num` | 10.1 | 2.57 |
| `hours-per-week` | 40.4 | 12.3 |
| `capital-gain` | 1,078 | 7,385 |
| `capital-loss` | 87 | 403 |

**GLANCE + NearestNeighbors — Action điển hình:**

```
capital-gain:  +42,000
```

```
feasibility(capital-gain) = 1 / (1 + 42000 / 7385) = 1 / 6.69 = 0.149
Feasibility Score = 0.149
```

Nghĩa là: mức tăng $42.000 của `capital-gain` lớn gấp ~5.7 lần độ lệch chuẩn — cực kỳ không thực tế.

**TabGLANCE — Action điển hình:**

```
hours-per-week:  +8
education-num:   +2
workclass:       → Self-emp-not-inc   (categorical)
```

```
feasibility(hours-per-week) = 1 / (1 + 8 / 12.3)   = 1 / 1.65  = 0.606
feasibility(education-num)  = 1 / (1 + 2 / 2.57)   = 1 / 1.78  = 0.562
feasibility(workclass)      = 1.0  (categorical)

Feasibility Score = (0.606 + 0.562 + 1.0) / 3 = 0.723
```

Nghĩa là: tăng 8 giờ làm/tuần và 2 bậc học vấn là những thay đổi nằm trong khoảng biến động bình thường của dữ liệu.

**So sánh:**

```
NearestNeighbors  →  Feasibility Score ≈ 0.15  (hầu như không khả thi)
TabGLANCE         →  Feasibility Score ≈ 0.72  (khả thi và tự nhiên)
```

---

## 4. Dominant Feature Concentration (Mức độ tập trung vào một feature)

### Định nghĩa và động lực

Một action có thể có Feasibility Score trung bình, nhưng thực chất chỉ thay đổi một feature duy nhất với mức độ rất lớn. Dominant Feature Concentration phát hiện điều này bằng cách đo xem feature "nặng" nhất đóng góp bao nhiêu phần trăm tổng thay đổi.

Để so sánh được giữa các feature có đơn vị khác nhau, delta của feature số được chuẩn hoá theo **khoảng (range)** của tập training:

```
contribution_j = |δ_j| / (max_j - min_j)        (feature số)
contribution_j = 1.0                              (feature categorical)

Concentration = max_j(contribution_j) / Σ_j contribution_j
```

### Tính chất

- Nếu action chỉ thay đổi **1 feature**: `Concentration = 1.0` (tối đa)
- Nếu action thay đổi **K feature đều nhau**: `Concentration = 1/K` (tối thiểu)
- Concentration thấp = recourse đa chiều, cân bằng
- Concentration cao = một feature đang "gánh" toàn bộ recourse

### Ví dụ minh hoạ (bộ dữ liệu Adult)

Khoảng (range) của một số feature trong tập training:

| Feature | Min | Max | Range |
|---------|-----|-----|-------|
| `capital-gain` | 0 | 99,999 | 99,999 |
| `hours-per-week` | 1 | 99 | 98 |
| `education-num` | 1 | 16 | 15 |
| `age` | 17 | 90 | 73 |

**Trường hợp 1 — GLANCE + NearestNeighbors:**

```
capital-gain: +42,000
```

```
contribution(capital-gain) = 42000 / 99999 = 0.420

Concentration = 0.420 / 0.420 = 1.000
```

Một feature duy nhất chiếm 100% sự thay đổi — đây là dấu hiệu rõ ràng của việc khai thác tương quan giả.

**Trường hợp 2 — TabGLANCE (ít cân bằng):**

```
hours-per-week: +8
education-num:  +2
workclass:      → Self-emp-not-inc
```

```
contribution(hours-per-week) = 8  / 98    = 0.082
contribution(education-num)  = 2  / 15    = 0.133
contribution(workclass)      = 1.0  (categorical)

Tổng = 0.082 + 0.133 + 1.0 = 1.215
Concentration = max(0.082, 0.133, 1.0) / 1.215 = 1.0 / 1.215 = 0.823
```

Feature categorical vẫn chiếm ~82% — nhưng đây là một thay đổi categorical hợp lý (chuyển loại công việc), khác với việc tăng capital-gain phi thực tế.

**Trường hợp 3 — TabGLANCE (cân bằng hơn):**

```
hours-per-week:  +8
education-num:   +2
occupation:      → Exec-managerial
capital-loss:    +200
```

```
contribution(hours-per-week) = 8   / 98    = 0.082
contribution(education-num)  = 2   / 15    = 0.133
contribution(occupation)     = 1.0
contribution(capital-loss)   = 200 / 4356  = 0.046

Tổng = 1.261
Concentration = 1.0 / 1.261 = 0.793
```

> Concentration thấp hơn = recourse đa chiều hơn, không phụ thuộc vào một "trick" duy nhất.

---

## 5. Action Diversity (Tính đa dạng của các action)

### Định nghĩa và động lực

GLANCE tạo ra K action toàn cục cho K cluster. Nếu tất cả K action đều thay đổi cùng một tập feature (ví dụ: tất cả đều dùng `capital-gain`), thì thực chất chỉ có một lộ trình recourse duy nhất — không có sự đa dạng cho các nhóm dân số khác nhau.

Action Diversity đo mức độ khác biệt giữa các action bằng **độ tương đồng Jaccard** của tập feature thay đổi:

```
changed(A_i) = tập các feature có |δ_j| > 1e-6 trong Action i

Jaccard(A_i, A_j) = |changed(A_i) ∩ changed(A_j)| / |changed(A_i) ∪ changed(A_j)|

Action Diversity = 1 - mean_{i < j} Jaccard(A_i, A_j)
```

### Tính chất

| Tình huống | Jaccard | Diversity |
|------------|---------|-----------|
| Hai action thay đổi hoàn toàn khác feature | 0.0 | 1.0 |
| Hai action thay đổi một số feature chung | 0.0 < J < 1.0 | trung gian |
| Hai action thay đổi đúng cùng tập feature | 1.0 | 0.0 |

**Lưu ý:** Jaccard chỉ xét *tập feature thay đổi*, không xét *mức độ thay đổi*. Hai action cùng thay đổi `hours-per-week` nhưng một cái +5, một cái +15 vẫn có Jaccard = 1.0 (hoàn toàn giống nhau về mặt feature).

### Ví dụ minh hoạ — GLANCE + NearestNeighbors (4 action)

```
Action 1: capital-gain +42,000
Action 2: capital-gain +38,000
Action 3: capital-gain +55,000
Action 4: capital-gain +40,000, hours-per-week +5
```

Tập feature thay đổi:
```
changed(A1) = {capital-gain}
changed(A2) = {capital-gain}
changed(A3) = {capital-gain}
changed(A4) = {capital-gain, hours-per-week}
```

Tính Jaccard cho tất cả 6 cặp:
```
J(A1,A2) = |{cg}∩{cg}| / |{cg}∪{cg}|   = 1/1 = 1.00
J(A1,A3) = 1.00
J(A1,A4) = |{cg}∩{cg,hw}| / |{cg}∪{cg,hw}| = 1/2 = 0.50
J(A2,A3) = 1.00
J(A2,A4) = 0.50
J(A3,A4) = 0.50

Mean Jaccard = (1.00+1.00+0.50+1.00+0.50+0.50) / 6 = 4.50/6 = 0.75

Action Diversity = 1 - 0.75 = 0.25
```

### Ví dụ minh hoạ — TabGLANCE (4 action)

```
Action 1: education-num +2, hours-per-week +8
Action 2: workclass → Self-emp, occupation → Exec-managerial
Action 3: hours-per-week +10, education-num +3, capital-loss +150
Action 4: occupation → Prof-specialty, education-num +1
```

Tập feature thay đổi:
```
changed(A1) = {education-num, hours-per-week}
changed(A2) = {workclass, occupation}
changed(A3) = {hours-per-week, education-num, capital-loss}
changed(A4) = {occupation, education-num}
```

Tính Jaccard:
```
J(A1,A2) = |{}| / |{edu,hw,wc,occ}|           = 0/4 = 0.00
J(A1,A3) = |{edu,hw}| / |{edu,hw,cl}|          = 2/3 = 0.67
J(A1,A4) = |{edu}| / |{edu,hw,occ}|            = 1/3 = 0.33
J(A2,A3) = |{}| / |{wc,occ,hw,edu,cl}|         = 0/5 = 0.00
J(A2,A4) = |{occ}| / |{wc,occ,edu}|            = 1/3 = 0.33
J(A3,A4) = |{edu}| / |{hw,edu,cl,occ}|         = 1/4 = 0.25

Mean Jaccard = (0.00+0.67+0.33+0.00+0.33+0.25) / 6 = 1.58/6 = 0.26

Action Diversity = 1 - 0.26 = 0.74
```

**Kết quả so sánh:**
```
NearestNeighbors  →  Diversity = 0.25  (hầu hết action đều dùng capital-gain)
TabGLANCE         →  Diversity = 0.74  (mỗi action nhắm vào tập feature khác nhau)
```

---

## Bảng tổng hợp

| Metric | Ký hiệu tốt hơn | Phạm vi | Đo lường điều gì |
|--------|----------------|---------|-----------------|
| Effectiveness | ↑ | [0, 1] | % instance mà dự đoán được flip thành công |
| Mean Recourse Cost | ↓ | [0, ∞) | Độ lớn trung bình của các thay đổi (chuẩn hoá MAD) |
| Feasibility Score | ↑ | (0, 1] | Mức độ thực tế của thay đổi so với std tập training |
| Dominant Concentration | ↓ | [1/K, 1] | Mức độ một feature chiếm hầu hết action |
| Action Diversity | ↑ | [0, 1] | Sự khác biệt giữa các action toàn cục (Jaccard) |

---

## Tại sao cần các metric mới?

### Vấn đề với Effectiveness và Cost đơn thuần

GLANCE với `NearestNeighbors` đạt effectiveness rất cao (~92%) vì nó khai thác `capital-gain`: trong tập training, những người có `capital-gain > $40.000` gần như 100% có thu nhập >50K. Mô hình học được tương quan này, nên chỉ cần "giả lập" capital-gain cao là dự đoán sẽ flip.

Tuy nhiên:
- `capital-gain` phản ánh thu nhập từ đầu tư/cổ phiếu — không thể tăng tùy ý
- Đây là **tương quan thống kê**, không phải quan hệ nhân quả
- Người có thu nhập thấp không thể "đột ngột có $42.000 capital-gain"

### Ba metric mới phơi bày vấn đề này

```
                   Effectiveness  Cost   Feasibility  Concentration  Diversity
NearestNeighbors      0.92       1.24      0.18          0.97          0.12
RandomSampling        0.74       1.89      0.45          0.68          0.41
TabGLANCE             0.81       1.56      0.72          0.54          0.68
```

- **NearestNeighbors** win về Effectiveness nhưng thua toàn diện về chất lượng: Feasibility gần 0, Concentration gần 1, Diversity gần 0 → toàn bộ recourse phụ thuộc vào một trick capital-gain không thực tế.
- **TabGLANCE** tạo ra recourse *khả thi* (Feasibility cao), *đa chiều* (Concentration thấp hơn), và *đa dạng* (Diversity cao) — các action khác nhau phù hợp với các nhóm người khác nhau.

> **Kết luận:** Effectiveness đo liệu *mô hình có bị đánh lừa không*. Ba metric mới đo liệu *người thực có thể thực hiện được không và có được tư vấn phù hợp không*. Đây là hai câu hỏi hoàn toàn khác nhau trong bài toán algorithmic recourse.
