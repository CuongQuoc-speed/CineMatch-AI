# 🎬 CineMatch AI – Hybrid Movie Recommender System
## 1. Giới thiệu

CineMatch AI là hệ thống khuyến nghị phim lai
(Hybrid Movie Recommender System), được phát triển
nhằm đề xuất các bộ phim phù hợp với sở thích
của người dùng.

Đây là sản phẩm thuộc Đề tài 15, sử dụng Python,
Machine Learning và Streamlit.

## 2. Mục tiêu

- Xây dựng hệ thống gợi ý phim cá nhân hóa.
- Áp dụng Content-Based Filtering.
- Áp dụng Collaborative Filtering bằng SVD.
- Kết hợp hai thuật toán bằng Hybrid Recommendation.
- Đánh giá chất lượng khuyến nghị.
- Xây dựng giao diện website trực quan.

## 3. Công nghệ sử dụng

- Python 3.12
- Pandas, NumPy, SciPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Requests
- TMDb API

## 4. Dữ liệu

Bộ dữ liệu: MovieLens 100K.

Dữ liệu gồm:
- 943 người dùng
- 1.682 bộ phim
- 100.000 lượt đánh giá

Dữ liệu được chia thành Train, Validation và Test
phục vụ huấn luyện, lựa chọn tham số và đánh giá.

## 5. Các thuật toán

### 5.1. Content-Based Filtering

Sử dụng TF-IDF để biểu diễn đặc trưng thể loại phim.

Cosine Similarity được sử dụng để đo mức độ
tương đồng giữa hồ sơ sở thích người dùng
và các bộ phim.

### 5.2. Collaborative Filtering

Sử dụng mô hình phân rã ma trận SVD
với các yếu tố ẩn của người dùng và phim.

Mô hình học từ lịch sử đánh giá
để dự đoán mức độ yêu thích.

### 5.3. Hybrid Recommendation

Kết hợp điểm Content-Based và Collaborative:

Hybrid Score = alpha × Content Score
             + (1 - alpha) × Collaborative Score

Giá trị alpha được lựa chọn trên tập Validation.

Giá trị hiện tại: alpha = 0.30.

## 6. Chức năng website

### Trang chủ
- Tổng quan dữ liệu MovieLens.
- Thống kê phim, người dùng và lượt đánh giá.
- Hiển thị phim nổi bật và poster.
- Biểu đồ phân phối điểm đánh giá.

### Khám phá phim
- Tìm kiếm phim theo tên.
- Lọc thể loại và năm phát hành.
- Tìm các phim tương tự.
- Hiển thị poster từ TMDb.

### Gợi ý phim
- Content-Based Filtering.
- Collaborative Filtering bằng SVD.
- Hybrid Recommendation.
- Gợi ý cho người dùng mới.
- Hiển thị điểm khuyến nghị.

### Dashboard
- Precision@10.
- Recall@10.
- NDCG@10.
- Coverage@10.
- RMSE và MAE.
- Biểu đồ so sánh các thuật toán.

## 7. Cấu trúc thư mục

Hybrid_Movie_Recommender/
    .streamlit/
        secrets.toml
    data/
        processed/
    models/
    notebooks/
    report/
    results/
    streamlit_app/
        app.py
    requirements.txt
    README.md

Lưu ý: Không chia sẻ secrets.toml công khai.

## 8. Hướng dẫn cài đặt

Yêu cầu: Python 3.12.

Mở Terminal tại thư mục gốc dự án.

Tạo môi trường:

    py -3.12 -m venv .venv

Cài đặt thư viện:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

## 9. Cấu hình TMDb API

Để hiển thị poster phim, tạo file:

    .streamlit/secrets.toml

Nội dung:

    TMDB_API_KEY = "YOUR_TMDB_V3_API_KEY"

Thay giá trị bằng TMDb API Key (v3 auth) hợp lệ.

Nếu không cấu hình TMDb, hệ thống vẫn có thể
đề xuất phim nhưng sử dụng poster dự phòng.

Poster cần kết nối Internet để tải từ TMDb.

## 10. Chạy website

Chạy lệnh:

    .\.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py

Truy cập:

    http://localhost:8501

Để dừng website, nhấn Ctrl + C trong Terminal.

## 11. Kiểm thử

Hệ thống cần vượt qua 9 nhóm kiểm thử:

1. Trang chủ
2. Poster phim
3. Tìm kiếm và lọc
4. Phim tương tự
5. Content-Based Filtering
6. Collaborative Filtering
7. Hybrid Recommendation
8. Người dùng mới
9. Dashboard và khởi động lại

## 12. Đánh giá mô hình

Các chỉ số đánh giá được lưu trong thư mục results:

- final_test_comparison.csv
- final_test_rating_metrics.csv
- final_test_coverage.csv

Các chỉ số được tính từ dữ liệu kiểm thử
sau khi lựa chọn tham số bằng tập Validation.

## 13. Hạn chế

- Dữ liệu MovieLens 100K có quy mô hạn chế.
- Content-Based chủ yếu sử dụng thể loại phim.
- Người dùng mới chưa có vector SVD cá nhân.
- Poster phụ thuộc vào dữ liệu và API TMDb.
- Một số tên phim có thể không khớp hoàn toàn
  giữa MovieLens và TMDb.

## 14. Nguồn tham khảo và ghi nhận

- MovieLens: https://grouplens.org/datasets/movielens/
- TMDb: https://www.themoviedb.org/
- Streamlit: https://streamlit.io/
- Scikit-learn: https://scikit-learn.org/

This product uses the TMDB API but is not
endorsed or certified by TMDB.

## 15. Kết luận

CineMatch AI minh họa việc kết hợp phương pháp
Content-Based Filtering và Collaborative Filtering
để xây dựng hệ thống khuyến nghị phim lai.

Website cho phép người dùng khám phá phim,
nhận khuyến nghị cá nhân hóa và xem kết quả
đánh giá chất lượng các thuật toán.
