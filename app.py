import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. CẤU HÌNH WEBSITE
# ==========================================

st.set_page_config(
    page_title="CineMatch AI | Gợi ý phim thông minh",
    page_icon="🎬",
    layout="wide"
)


ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT
DATA_DIR = ROOT
RESULT_DIR = ROOT

# API key được lưu riêng, không ghi trực tiếp vào code.
try:
    TMDB_API_KEY = str(
        st.secrets.get("TMDB_API_KEY", "")
    ).strip()
except Exception:
    TMDB_API_KEY = os.getenv(
        "TMDB_API_KEY", ""
    ).strip()


# ==========================================
# 2. GIAO DIỆN CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            ellipse at 80% 0%,
            #1b2850,
            transparent 45%
        ),
        #090f1d;
}

[data-testid="stSidebar"] {
    background: #101a2d;
    border-right: 1px solid #273450;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.7rem;
}

.hero {
    background: linear-gradient(
        115deg,
        #193c82,
        #5437a5,
        #172956
    );
    border: 1px solid #4d5592;
    border-radius: 22px;
    padding: 40px 34px;
    margin-bottom: 22px;
}

.hero h1 {
    color: white;
    font-size: clamp(2rem, 4vw, 3.2rem);
    margin: 0 0 12px;
}

.hero p {
    color: #e2e8ff;
    font-size: 1.05rem;
    max-width: 760px;
    line-height: 1.65;
}

[data-testid="stMetric"] {
    background: #18243a;
    border: 1px solid #2f415e;
    padding: 18px;
    border-radius: 16px;
}

.poster-fallback {
    height: 260px;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: linear-gradient(
        155deg,
        #243758,
        #101827
    );
    color: #aab9dc;
    margin: 6px 0 12px;
    font-size: 3.5rem;
}

.poster-fallback small {
    font-size: 0.9rem;
    margin-top: 8px;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(
        110deg,
        #4167f2,
        #8452dd
    );
    border: 0;
    border-radius: 10px;
    color: white;
    font-weight: 700;
}

.footer-note {
    text-align: center;
    font-size: 0.82rem;
    opacity: 0.82;
    margin-top: 20px;
}

/* Nội dung và tiêu đề */
.stApp,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp p,
.stApp span,
.stApp label,
.stApp strong,
.stApp small {
    color: #FFFFFF !important;
}

/* Thanh menu bên trái */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Các ô thống kê */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Nút bấm */
.stButton button,
.stButton button * {
    color: #FFFFFF !important;
}

/* Ô nhập liệu */
.stTextInput input,
.stNumberInput input {
    color: #FFFFFF !important;
    background-color: #18243A !important;
}

/* Chữ trong danh sách lựa chọn */
[data-baseweb="select"] {
    color: #FFFFFF !important;
}

/* Chú thích */
[data-testid="stCaptionContainer"] {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. TẢI MÔ HÌNH VÀ DỮ LIỆU
# ==========================================

@st.cache_resource(
    show_spinner="Đang tải mô hình khuyến nghị..."
)
def load_models():

    cb = joblib.load(
        MODEL_DIR / "content_based_bundle.joblib"
    )

    cf = joblib.load(
        MODEL_DIR / "collaborative_svd_bundle.joblib"
    )

    hybrid = joblib.load(
        MODEL_DIR / "hybrid_bundle.joblib"
    )

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    movie_ids = np.asarray(
        hybrid["candidate_movie_ids"],
        dtype=int
    )

    info = cb["movie_info"].copy()
    info["movie_id"] = info["movie_id"].astype(int)

    catalog = (
        info.set_index("movie_id")
        .loc[movie_ids]
        .reset_index()
    )

    cb_idx = catalog[
        "matrix_index"
    ].to_numpy(dtype=int)

    cf_idx = np.asarray(
        [
            cf["movie_to_index"][int(mid)]
            for mid in movie_ids
        ],
        dtype=int
    )

    return (
        cb,
        cf,
        hybrid,
        train,
        movie_ids,
        catalog,
        cb["tfidf_matrix"][cb_idx].tocsr(),
        cf["item_factors"][cf_idx],
        cf["item_bias"][cf_idx]
    )


@st.cache_data(show_spinner=False)
def load_results():

    names = {
        "topk": "final_test_comparison.csv",
        "rating": "final_test_rating_metrics.csv",
        "coverage": "final_test_coverage.csv"
    }

    return {
        key: (
            pd.read_csv(RESULT_DIR / filename)
            if (RESULT_DIR / filename).exists()
            else pd.DataFrame()
        )
        for key, filename in names.items()
    }


try:

    (
        cb,
        cf,
        hybrid,
        train,
        movie_ids,
        catalog,
        cb_matrix,
        cf_factors,
        cf_bias
    ) = load_models()

    results = load_results()

except Exception as exc:

    st.error(
        "Không thể tải mô hình hoặc dữ liệu. "
        "Kiểm tra thư mục models và data/processed."
    )

    st.exception(exc)
    st.stop()


alpha = float(hybrid["alpha"])

if TMDB_API_KEY:
    st.sidebar.success("✅ Đã đọc được TMDb API key")
else:
    st.sidebar.error("❌ Chưa đọc được TMDb API key")

weighted_ratings = (
    pd.to_numeric(
        catalog["weighted_rating"],
        errors="coerce"
    )
    .fillna(0)
    .to_numpy(dtype=float)
)

movie_titles = dict(
    zip(
        catalog["movie_id"].astype(int),
        catalog["title"].astype(str)
    )
)

movie_positions = {
    int(mid): i
    for i, mid in enumerate(movie_ids)
}

cb_positions = {
    int(row.movie_id): int(row.matrix_index)
    for row in cb["movie_info"].itertuples()
}


# ==========================================
# 4. CHUYỂN THỂ LOẠI SANG TIẾNG VIỆT
# ==========================================

GENRE_VI = {
    "unknown": "Chưa xác định",
    "Action": "Hành động",
    "Adventure": "Phiêu lưu",
    "Animation": "Hoạt hình",
    "Children": "Thiếu nhi",
    "Comedy": "Hài",
    "Crime": "Tội phạm",
    "Documentary": "Tài liệu",
    "Drama": "Chính kịch",
    "Fantasy": "Giả tưởng",
    "Film_Noir": "Phim đen",
    "Horror": "Kinh dị",
    "Musical": "Âm nhạc",
    "Mystery": "Bí ẩn",
    "Romance": "Lãng mạn",
    "Sci_Fi": "Khoa học viễn tưởng",
    "Thriller": "Giật gân",
    "War": "Chiến tranh",
    "Western": "Miền Tây"
}


def genres_vi(raw):

    tokens = str(raw).split()

    return ", ".join(
        GENRE_VI.get(token, token)
        for token in tokens
    ) or "Chưa rõ thể loại"


catalog["thể_loại"] = (
    catalog["genre_text"]
    .fillna("")
    .map(genres_vi)
)

catalog["năm"] = pd.to_numeric(
    catalog["title"]
    .astype(str)
    .str.extract(r"\((\d{4})\)\s*$")[0],
    errors="coerce"
)


# ==========================================
# 5. TÌM POSTER PHIM TRÊN TMDB
# ==========================================

def split_title_year(title):

    title = str(title).strip()

    match = re.search(
        r"\((\d{4})\)\s*$",
        title
    )

    year = (
        int(match.group(1))
        if match
        else None
    )

    name = (
        title[:match.start()].strip()
        if match
        else title
    )

    # MovieLens có những tên như "Godfather, The".
    for article in ("The", "A", "An"):

        if name.endswith(", " + article):

            name = (
                article
                + " "
                + name[:-(len(article) + 2)]
            )

            break

    return name, year


@st.cache_data(
    ttl=24 * 60 * 60,
    show_spinner=False
)
def poster_url(title, api_key):

    if not api_key:
        return None

    name, year = split_title_year(title)

    base_params = {
        "api_key": api_key,
        "query": name,
        "include_adult": "false",
        "language": "en-US"
    }

    queries = (
        [dict(base_params, year=year), base_params]
        if year
        else [base_params]
    )

    candidates = []

    try:

        for params in queries:

            response = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params=params,
                timeout=(3, 6)
            )

            response.raise_for_status()

            candidates = [
                item
                for item in response.json().get(
                    "results", []
                )
                if item.get("poster_path")
            ]

            if candidates:
                break

    except requests.RequestException:
        return None

    if not candidates:
        return None

    # Ưu tiên tên phim khớp và đúng năm.
    def match_score(item):

        options = [
            str(item.get("title", "")),
            str(item.get("original_title", ""))
        ]

        similarity = max(
            SequenceMatcher(
                None,
                name.casefold(),
                candidate.casefold()
            ).ratio()
            for candidate in options
        )

        release_year = str(
            item.get("release_date", "")
        )[:4]

        year_bonus = (
            0.2
            if year and release_year == str(year)
            else 0
        )

        return similarity + year_bonus

    best = max(
        candidates,
        key=match_score
    )

    if match_score(best) < 0.48:
        return None

    return (
        "https://image.tmdb.org/t/p/w342"
        + best["poster_path"]
    )


def render_poster(title):

    url = poster_url(
        title,
        TMDB_API_KEY
    )

    if url:

        st.image(
            url,
            use_container_width=True
        )

    else:

        st.markdown(
            """
            <div class="poster-fallback">
                🎬
                <small>Chưa có poster</small>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# 6. CONTENT-BASED FILTERING
# ==========================================

def cb_scores_for_user(user_id):

    profile = cb["user_profiles"].get(
        int(user_id)
    )

    if profile is None:
        return np.zeros(len(movie_ids))

    profile = np.asarray(
        profile,
        dtype=float
    ).reshape(1, -1)

    if np.linalg.norm(profile) < 1e-12:
        return np.zeros(len(movie_ids))

    sim = cosine_similarity(
        profile,
        cb_matrix
    ).ravel()

    return (
        np.clip(sim, -1, 1) + 1
    ) / 2


# ==========================================
# 7. COLLABORATIVE FILTERING SVD
# ==========================================

def cf_scores_for_user(user_id):

    scores = np.asarray(
        cf["global_mean"] + cf_bias,
        dtype=float
    ).copy()

    user_idx = cf["user_to_index"].get(
        int(user_id)
    )

    if user_idx is not None:

        scores += (
            cf["user_bias"][user_idx]
            + cf_factors @ cf["user_factors"][user_idx]
        )

    return (
        np.clip(scores, 1, 5) - 1
    ) / 4


# ==========================================
# 8. XẾP HẠNG PHIM
# ==========================================

def rank_movies(
    scores,
    excluded_ids=None,
    top_n=10
):

    scores = np.nan_to_num(
        np.asarray(scores, dtype=float),
        nan=-1.0
    )

    excluded = (
        set(int(x) for x in excluded_ids)
        if excluded_ids is not None
        else set()
    )

    valid = np.flatnonzero(
        ~np.isin(
            movie_ids,
            list(excluded)
        )
    )

    ordering = np.lexsort((
        movie_ids[valid],
        -weighted_ratings[valid],
        -scores[valid]
    ))

    picked = valid[
        ordering[:top_n]
    ]

    output = (
        catalog.iloc[picked]
        .copy()
        .reset_index(drop=True)
    )

    output.insert(
        0,
        "Hạng",
        np.arange(1, len(output) + 1)
    )

    output["score"] = scores[picked]

    return output, picked


def recommend_existing(
    user_id,
    algorithm,
    top_n
):

    cb_s = cb_scores_for_user(user_id)

    cf_s = cf_scores_for_user(user_id)

    hybrid_s = (
        alpha * cb_s
        + (1 - alpha) * cf_s
    )

    scores = {
        "Content-Based": cb_s,
        "Collaborative Filtering": cf_s,
        "Hybrid": hybrid_s
    }[algorithm]

    seen = cb["seen_by_user"].get(
        int(user_id),
        set()
    )

    recs, positions = rank_movies(
        scores,
        excluded_ids=seen,
        top_n=top_n
    )

    recs["content_score"] = cb_s[positions]

    recs["collaborative_score"] = cf_s[positions]

    recs["hybrid_score"] = hybrid_s[positions]

    return recs


# ==========================================
# 9. GỢI Ý CHO NGƯỜI DÙNG MỚI
# ==========================================

def recommend_new(ratings, top_n):

    indices = []
    weights = []

    for mid, rating in ratings.items():

        if int(mid) in cb_positions:

            indices.append(
                cb_positions[int(mid)]
            )

            weights.append(
                float(rating) - 3
            )

    if indices:

        matrix = cb["tfidf_matrix"][indices]

        profile = np.asarray(
            matrix.multiply(
                np.asarray(weights)[:, None]
            ).sum(axis=0)
        ).ravel()

    else:

        profile = np.zeros(
            cb_matrix.shape[1]
        )

    norm = np.linalg.norm(profile)

    fallback = norm < 1e-12

    if fallback:

        scores = np.clip(
            (weighted_ratings - 1) / 4,
            0,
            1
        )

    else:

        sim = cosine_similarity(
            (profile / norm).reshape(1, -1),
            cb_matrix
        ).ravel()

        scores = (
            np.clip(sim, -1, 1) + 1
        ) / 2

    recs, _ = rank_movies(
        scores,
        excluded_ids=ratings.keys(),
        top_n=top_n
    )

    return recs, fallback


# ==========================================
# 10. HIỂN THỊ THẺ PHIM KÈM POSTER
# ==========================================

def show_movie_cards(
    frame,
    label="Điểm gợi ý",
    ncols=2
):

    if frame.empty:

        st.info(
            "Không tìm thấy phim phù hợp."
        )
        return

    cols = st.columns(ncols)

    for i, (_, movie) in enumerate(
        frame.iterrows()
    ):

        with cols[i % ncols]:

            with st.container(border=True):

                render_poster(
                    movie["title"]
                )

                st.markdown(
                    f"**{movie['title']}**"
                )

                st.caption(
                    movie["thể_loại"]
                )

                score = float(movie["score"])

                st.write(
                    f"**{label}: {score:.3f}**"
                )

                st.progress(
                    float(np.clip(score, 0, 1))
                )

                if "hybrid_score" in frame.columns:

                    with st.expander(
                        "Xem điểm từng thuật toán"
                    ):

                        st.write(
                            f"Content-Based: "
                            f"{movie['content_score']:.4f}"
                        )

                        st.write(
                            f"Collaborative SVD: "
                            f"{movie['collaborative_score']:.4f}"
                        )

                        st.write(
                            f"Hybrid: "
                            f"{movie['hybrid_score']:.4f}"
                        )


# ==========================================
# 11. THANH ĐIỀU HƯỚNG
# ==========================================

st.sidebar.title(
    "🎬 CINEMATCH AI"
)

st.sidebar.caption(
    "Hệ thống khuyến nghị phim thông minh"
)

page = st.sidebar.radio(
    "DANH MỤC",
    [
        "🏠 Trang chủ",
        "🔎 Khám phá phim",
        "✨ Gợi ý phim",
        "📊 Dashboard"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    "**🧠 Content-Based:** TF-IDF"
)

st.sidebar.markdown(
    "**🤝 Collaborative:** SVD"
)

st.sidebar.metric(
    "Trọng số Hybrid (α)",
    f"{alpha:.2f}"
)

st.sidebar.caption(
    "Dữ liệu: MovieLens 100K"
)

if not TMDB_API_KEY:

    st.sidebar.info(
        "Chưa cài TMDb API key. "
        "Website vẫn hoạt động nhưng "
        "sẽ hiện poster dự phòng."
    )


# ==========================================
# 12. TRANG CHỦ
# ==========================================

if page == "🏠 Trang chủ":

    st.markdown(
        """
        <section class="hero">
            <h1>🎬 CineMatch AI</h1>
            <p>
                Khám phá điện ảnh theo sở thích
                với hệ thống khuyến nghị phim lai,
                kết hợp Content-Based Filtering
                và Collaborative Filtering (SVD).
            </p>
        </section>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "🎥 Tổng số phim",
        f"{len(cb['movie_info']):,}"
    )

    m2.metric(
        "👥 Người dùng",
        f"{len(cb['user_profiles']):,}"
    )

    m3.metric(
        "⭐ Lượt đánh giá Train",
        f"{len(train):,}"
    )

    m4.metric(
        "🧠 Hybrid Alpha",
        f"{alpha:.2f}"
    )

    st.subheader(
        "🔥 Những bộ phim nổi bật"
    )

    popular = catalog.sort_values(
        ["weighted_rating", "rating_count"],
        ascending=False
    ).head(8)

    cols = st.columns(4)

    for i, (_, movie) in enumerate(
        popular.iterrows()
    ):

        with cols[i % 4]:

            with st.container(border=True):

                render_poster(
                    movie["title"]
                )

                st.markdown(
                    f"**{movie['title']}**"
                )

                st.caption(
                    movie["thể_loại"]
                )

                st.write(
                    f"⭐ **{float(movie['weighted_rating']):.2f}/5**"
                )

                st.caption(
                    f"{int(movie['rating_count'])} lượt đánh giá"
                )

    st.subheader(
        "📈 Phân bố lượt đánh giá"
    )

    hist = (
        train["rating"]
        .value_counts()
        .sort_index()
        .rename_axis("Số sao")
        .reset_index(name="Số lượt đánh giá")
    )

    fig = px.bar(
        hist,
        x="Số sao",
        y="Số lượt đánh giá",
        template="plotly_dark",
        title="Phân bố điểm đánh giá trong tập Train"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# 13. TRANG KHÁM PHÁ PHIM
# ==========================================

elif page == "🔎 Khám phá phim":

    st.title(
        "🔎 Khám phá phim"
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        keyword = st.text_input(
            "Tìm theo tên phim",
            placeholder="Ví dụ: Toy Story, Star Wars..."
        )

    with col2:

        selected_genre = st.selectbox(
            "Lọc theo thể loại",
            ["Tất cả"] + list(GENRE_VI.values())
        )

    filtered = catalog.copy()

    if keyword.strip():

        filtered = filtered[
            filtered["title"].str.contains(
                keyword.strip(),
                case=False,
                regex=False,
                na=False
            )
        ]

    if selected_genre != "Tất cả":

        genre_key = next(
            key
            for key, value in GENRE_VI.items()
            if value == selected_genre
        )

        filtered = filtered[
            filtered["genre_text"]
            .fillna("")
            .str.split()
            .map(
                lambda values: genre_key in values
            )
        ]

    valid_years = filtered["năm"].dropna()

    if not valid_years.empty:

        low = int(valid_years.min())
        high = int(valid_years.max())

        if low < high:

            yr = st.slider(
                "Khoảng năm phát hành",
                low,
                high,
                (low, high)
            )

            filtered = filtered[
                filtered["năm"].between(*yr)
            ]

    st.write(
        f"**Tìm thấy {len(filtered)} phim**"
    )

    shown = filtered[
        [
            "title",
            "thể_loại",
            "năm",
            "rating_count",
            "weighted_rating"
        ]
    ].rename(columns={
        "title": "Tên phim",
        "thể_loại": "Thể loại",
        "năm": "Năm",
        "rating_count": "Số đánh giá",
        "weighted_rating": "Điểm xếp hạng"
    })

    st.dataframe(
        shown.round(2),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "🎯 Tìm phim tương tự"
    )

    if not filtered.empty:

        selected_id = st.selectbox(
            "Chọn phim",
            filtered["movie_id"]
            .astype(int)
            .tolist(),
            format_func=lambda mid: movie_titles.get(
                mid,
                str(mid)
            )
        )

        if st.button(
            "Tìm 10 phim tương tự",
            type="primary"
        ):

            matrix_i = movie_positions[
                int(selected_id)
            ]

            sims = cosine_similarity(
                cb_matrix[matrix_i],
                cb_matrix
            ).ravel()

            output, _ = rank_movies(
                sims,
                excluded_ids=[selected_id],
                top_n=10
            )

            show_movie_cards(
                output,
                label="Độ tương đồng"
            )

    else:

        st.info(
            "Không có phim phù hợp "
            "với điều kiện tìm kiếm."
        )


# ==========================================
# 14. TRANG GỢI Ý PHIM
# ==========================================

elif page == "✨ Gợi ý phim":

    st.title(
        "✨ Gợi ý phim dành cho bạn"
    )

    mode = st.radio(
        "Chế độ",
        [
            "Người dùng MovieLens",
            "Người dùng mới"
        ],
        horizontal=True
    )

    top_n = st.slider(
        "Số phim đề xuất",
        min_value=5,
        max_value=20,
        value=10
    )

    # --------------------------------------
    # NGƯỜI DÙNG ĐÃ CÓ TRONG MOVIELENS
    # --------------------------------------

    if mode == "Người dùng MovieLens":

        users = sorted(
            set(
                int(x)
                for x in cb["user_profiles"]
            )
            &
            set(
                int(x)
                for x in cf["user_to_index"]
            )
        )

        if not users:

            st.warning(
                "Không tìm thấy người dùng "
                "có sẵn trong cả hai mô hình."
            )

            st.stop()

        user_id = st.selectbox(
            "Mã người dùng",
            users,
            index=(
                users.index(196)
                if 196 in users
                else 0
            )
        )

        methods = {
            "Hybrid": "Hybrid – Kết hợp hai thuật toán",
            "Content-Based": "Content-Based – Theo thể loại",
            "Collaborative Filtering":
                "Collaborative Filtering – SVD"
        }

        method = st.selectbox(
            "Thuật toán",
            list(methods),
            format_func=lambda x: methods[x]
        )

        count = int(
            (
                train["user_id"] == user_id
            ).sum()
        )

        st.info(
            f"Người dùng {user_id} "
            f"đã đánh giá {count} phim "
            "trong tập Train."
        )

        if st.button(
            "✨ Đề xuất phim ngay",
            type="primary"
        ):

            recommendations = recommend_existing(
                user_id,
                method,
                top_n
            )

            st.success(
                f"Đã tìm được "
                f"{len(recommendations)} phim "
                f"cho User {user_id}."
            )

            show_movie_cards(
                recommendations
            )

            with st.expander(
                "Bảng kết quả chi tiết"
            ):

                st.dataframe(
                    recommendations[
                        [
                            "title",
                            "thể_loại",
                            "content_score",
                            "collaborative_score",
                            "hybrid_score"
                        ]
                    ].round(4),
                    hide_index=True,
                    use_container_width=True
                )

    # --------------------------------------
    # NGƯỜI DÙNG MỚI (COLD START)
    # --------------------------------------

    else:

        st.info(
            "Chọn từ 2 đến 5 phim đã xem "
            "và đánh giá từ 1 đến 5 sao "
            "để hệ thống hiểu sở thích."
        )

        chosen = st.multiselect(
            "Chọn các phim đã xem",
            options=movie_ids.tolist(),
            format_func=lambda mid: movie_titles.get(
                mid,
                str(mid)
            ),
            max_selections=5
        )

        ratings = {}

        for mid in chosen:

            ratings[int(mid)] = st.slider(
                f"⭐ {movie_titles[int(mid)]}",
                1,
                5,
                4,
                key=f"rating_{mid}"
            )

        if len(ratings) < 2:

            st.caption(
                "Vui lòng chọn ít nhất 2 phim."
            )

        elif st.button(
            "✨ Tìm phim hợp sở thích",
            type="primary"
        ):

            rec, fallback = recommend_new(
                ratings,
                top_n
            )

            if fallback:

                st.warning(
                    "Sở thích chưa đủ rõ. "
                    "Hệ thống gợi ý theo "
                    "mức độ nổi bật."
                )

            else:

                st.success(
                    "Đã phân tích sở thích "
                    "và tạo danh sách đề xuất!"
                )

            show_movie_cards(rec)

        st.caption(
            "Người dùng mới sử dụng Content-Based. "
            "SVD chỉ cá nhân hóa cho người dùng "
            "đã có dữ liệu huấn luyện."
        )


# ==========================================
# 15. DASHBOARD ĐÁNH GIÁ
# ==========================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Dashboard đánh giá hệ thống"
    )

    st.write(
        "Kết quả thực nghiệm "
        "trên tập Test MovieLens 100K."
    )

    topk = results["topk"]
    rating = results["rating"]
    coverage = results["coverage"]

    # --------------------------------------
    # PRECISION, RECALL, NDCG
    # --------------------------------------

    if topk.empty:

        st.warning(
            "Không tìm thấy file "
            "final_test_comparison.csv."
        )

    else:

        hybrid_rows = topk[
            topk["Algorithm"] == "Hybrid"
        ]

        if not hybrid_rows.empty:

            row = hybrid_rows.iloc[0]

            cols = st.columns(4)

            metrics = [
                "Precision@10",
                "Recall@10",
                "NDCG@10",
                "Coverage@10"
            ]

            for col, metric in zip(
                cols,
                metrics
            ):

                col.metric(
                    metric,
                    f"{float(row[metric]):.4f}"
                )

        st.subheader(
            "So sánh các thuật toán"
        )

        st.dataframe(
            topk.round(4),
            hide_index=True,
            use_container_width=True
        )

        melted = topk.melt(
            id_vars="Algorithm",
            value_vars=[
                "Precision@10",
                "Recall@10",
                "NDCG@10"
            ],
            var_name="Chỉ số",
            value_name="Giá trị"
        )

        fig = px.bar(
            melted,
            x="Algorithm",
            y="Giá trị",
            color="Chỉ số",
            barmode="group",
            template="plotly_dark",
            title="Precision, Recall và NDCG"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------
    # RMSE VÀ MAE
    # --------------------------------------

    st.subheader(
        "📉 RMSE và MAE"
    )

    if not rating.empty:

        st.dataframe(
            rating.round(4),
            hide_index=True,
            use_container_width=True
        )

        subset = rating[
            rating["Subset"] == "All Test"
        ]

        if not subset.empty:

            mel = subset.melt(
                id_vars="Algorithm",
                value_vars=["RMSE", "MAE"],
                var_name="Chỉ số",
                value_name="Sai số"
            )

            fig = px.bar(
                mel,
                x="Algorithm",
                y="Sai số",
                color="Chỉ số",
                barmode="group",
                template="plotly_dark",
                title="So sánh RMSE và MAE"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Chưa có dữ liệu "
            "sai số dự đoán."
        )

    # --------------------------------------
    # COVERAGE
    # --------------------------------------

    st.subheader(
        "🎞️ Độ bao phủ phim"
    )

    if not coverage.empty:

        st.dataframe(
            coverage.round(4),
            hide_index=True,
            use_container_width=True
        )

        fig = px.bar(
            coverage,
            x="Algorithm",
            y="Coverage@10",
            template="plotly_dark",
            title="Coverage@10"
        )

        fig.update_yaxes(
            range=[0, 1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Chưa có dữ liệu Coverage."
        )

    st.info(
        f"Hệ số Hybrid α = {alpha:.2f}, "
        "được chọn từ tập Validation. "
        "Tập Test dùng để đánh giá cuối cùng."
    )


# ==========================================
# 16. CHÂN TRANG VÀ GHI NHẬN NGUỒN
# ==========================================

st.divider()

with st.expander(
    "ℹ️ Giới thiệu và nguồn dữ liệu poster"
):

    st.write(
        "CineMatch AI sử dụng MovieLens 100K "
        "để nghiên cứu hệ thống khuyến nghị phim. "
        "Poster được tìm từ TMDB khi có API key "
        "và kết nối Internet. Một số poster "
        "có thể thiếu hoặc không chính xác "
        "do sự khác biệt tên phim giữa "
        "hai cơ sở dữ liệu."
    )

    st.markdown(
        "Nguồn poster: "
        "[The Movie Database (TMDB)]"
        "(https://www.themoviedb.org/)."
    )

    st.caption(
        "This product uses the TMDB API but is "
        "not endorsed or certified by TMDB."
    )

st.markdown(
    """
    <div class="footer-note">
        🎬 CineMatch AI • Đề tài 15
        • Python · Machine Learning · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
