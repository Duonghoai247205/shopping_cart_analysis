# 📦 Case Study: Phân tích độ nhạy tham số (Parameter Sensitivity) trong Apriori

## 👥 Thông tin Nhóm
- **Nhóm:** 13
- **Thành viên:**
  - Nguyễn Hà Phương  
  - Dương Thị Hoài  

- **Chủ đề:** Chủ đề 4 – Phân tích độ nhạy tham số (Parameter Sensitivity)  
- **Dataset:** Online Retail – UCI Machine Learning Repository  
- **Phạm vi dữ liệu:** United Kingdom  

---

## 🎯 Mục tiêu

Mục tiêu của nghiên cứu này không chỉ dừng lại ở việc tìm ra các luật kết hợp bằng thuật toán Apriori, mà tập trung **phân tích tác động của các tham số đầu vào** đến số lượng, chất lượng và giá trị kinh doanh của các luật kết hợp.

Cụ thể, nhóm hướng tới:
- Đo lường sự biến thiên của luật khi thay đổi **min_support** và **min_confidence**  
- Phân tích sự đánh đổi (trade-off) giữa **độ bao phủ thị trường** và **độ tin cậy của luật**  
- Xác định bộ tham số phù hợp cho các chiến lược kinh doanh khác nhau (bán đại trà vs. bán ngách)

---

## 1. Ý tưởng & Feynman Style

Hãy tưởng tượng việc khai phá dữ liệu khách hàng giống như **đánh cá bằng lưới**:

- **min_support** chính là **kích thước mắt lưới**
- Lưới mắt nhỏ (Support ≈ 1%):  
  → Bắt được cả cá lớn lẫn cá nhỏ (sản phẩm ngách), nhưng cũng kéo theo nhiều “rác” (luật nhiễu)
- Lưới mắt to (Support ≈ 3%):  
  → Chỉ bắt được cá lớn (sản phẩm bán chạy), dữ liệu rất sạch nhưng bỏ lỡ nhiều cơ hội giá trị cao

👉 Vì vậy, **không tồn tại một bộ tham số tối ưu cho mọi mục tiêu**. Phân tích độ nhạy giúp ta chọn đúng “kích thước mắt lưới” cho từng chiến lược kinh doanh.

---

## 2. Quy trình Thực hiện

Dự án tuân theo pipeline Data Mining chuẩn, kết hợp tư duy lập trình hướng đối tượng (OOP):

1. Load & làm sạch dữ liệu bán lẻ  
2. Tạo ma trận giỏ hàng (Basket Preparation – One-hot Encoding)  
3. Thiết lập các kịch bản thử nghiệm tham số  
4. Áp dụng Apriori cho từng kịch bản  
5. Trực quan hóa kết quả  
6. So sánh & rút ra insight kinh doanh  

---

## 3. Tiền xử lý Dữ liệu

Dữ liệu bán lẻ thực tế chứa nhiều nhiễu, nhóm thực hiện các bước làm sạch giống **Chủ đề 2**:

- Loại bỏ hóa đơn hủy (`InvoiceNo` bắt đầu bằng `"C"`)
- Loại bỏ các dòng có `Quantity <= 0` hoặc `UnitPrice <= 0`
- Loại bỏ các dòng thiếu `Description`
- Chỉ giữ dữ liệu tại **United Kingdom** để đảm bảo hành vi mua sắm đồng nhất

### Thống kê dữ liệu sau làm sạch
- **Số giao dịch:** ~400,000 dòng  
- **Số khách hàng:** ~4,000  
- **Số sản phẩm duy nhất:** ~4,000  

---

## 4. Áp dụng Apriori & Thiết lập kịch bản tham số

Nhóm sử dụng thư viện `mlxtend.frequent_patterns` để chạy Apriori với **nhiều cấu hình tham số khác nhau**, trên cùng một tập dữ liệu basket boolean (`basket_bool.parquet`).

### Kịch bản A – Baseline (Mở rộng)
- `min_support = 0.01`
- `min_confidence = 0.3`
- `min_lift = 1.2`

**Mục tiêu:**  
Khám phá tối đa các cơ hội bán chéo, bao gồm cả các sản phẩm ngách có mối liên hệ mạnh.

### Kịch bản B – High Support (Khắt khe)
- `min_support = 0.03`
- `min_confidence = 0.3`
- `min_lift = 1.2`

**Mục tiêu:**  
Chỉ giữ lại các mối quan hệ phổ biến nhất giữa các sản phẩm bán chạy.

```python
from mlxtend.frequent_patterns import apriori, association_rules

# Kịch bản A
frequent_itemsets_1 = apriori(basket_bool, min_support=0.01, use_colnames=True)
rules_1 = association_rules(frequent_itemsets_1, metric="lift", min_threshold=1.2)

# Kịch bản B

frequent_itemsets_2 = apriori(basket_bool, min_support=0.03, use_colnames=True)
rules_2 = association_rules(frequent_itemsets_2, metric="lift", min_threshold=1.2)
```
## 5. Trực quan hóa (Visualization)

![Network Graph](images/network.png)  
*Hình 1: Mạng lưới sản phẩm thể hiện sự hình thành các cụm và Product Hub.*

![Top Lift Rules](images/top_lift.png)  
*Hình 2: Top các luật kết hợp có Lift cao nhất ở từng kịch bản tham số.*

---

## 6. Insight từ Kết quả

### Insight #1 – Quy luật “rơi tự do” của số lượng luật
Khi tăng `min_support` từ **1% lên 3%**, số lượng luật giảm hơn **99%**.

👉 Điều này cho thấy dữ liệu bán lẻ có đặc tính **đuôi dài (long-tail)** rất mạnh, nơi phần lớn các mối quan hệ mua sắm nằm ở vùng tần suất thấp.

---

### Insight #2 – Sự đánh đổi về chất lượng
Support cao giúp lọc nhiễu, nhưng lại loại bỏ các luật có **Lift cao nhất**.

👉 Các cơ hội cross-selling giá trị nhất thường đến từ **sản phẩm ngách**, không phải sản phẩm bán chạy đại trà.

---

### Insight #3 – Độ bền của cụm sản phẩm
Nhóm **HERB MARKER** vẫn tồn tại khi tăng `min_confidence` lên cao (~0.6).

👉 Điều này chứng tỏ đây là một **cụm sản phẩm bền vững**, không phải mối quan hệ ngẫu nhiên.

---

### Insight #4 – Product Hub phụ thuộc mạnh vào tham số
Ở Support thấp, xuất hiện các **Product Hub** như *HERB MARKER THYME* kết nối nhiều sản phẩm khác.  
Ở Support cao, các hub này biến mất.

👉 Chọn sai tham số đồng nghĩa với việc **không nhìn thấy cấu trúc mạng lưới thực sự**.

---

### Insight #5 – Chiến lược tồn kho
Các luật còn lại ở Support cao đại diện cho **nhu cầu nền tảng (Cash Cow)**.

👉 Đây là các sản phẩm **không được phép đứt hàng** và cần ưu tiên tồn kho.

---

## 7. Kết luận & Đề xuất Kinh doanh

### Kết luận
Không tồn tại bộ tham số “tốt nhất”, chỉ tồn tại bộ tham số **phù hợp với từng mục tiêu kinh doanh**.

### Đề xuất chiến lược đa tầng

#### 1️⃣ Mass Marketing (Support cao ~0.03)
- Tập trung các sản phẩm chủ lực
- Ưu tiên tồn kho **Jumbo Bag**, **Teacup**
- Trưng bày tại vị trí trung tâm, kệ chính

#### 2️⃣ Cross-selling & Cá nhân hóa (Support thấp ~0.01)
- Khai thác các luật có **Lift cao**
- Tạo bundle như **Herb Marker Set**
- Gợi ý mua kèm trên website/app

---

## 8. Link Code & Notebook
- **Repository:** (Link GitHub)
- **Notebook chính:** `apriori_modelling.ipynb`
- **Pipeline:** `run_papermill.py`
- **Thư viện:** `apriori_library.py`

---

## 9. Slide trình bày
- **Link Slide:** 
