
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

try:
    from streamlit_searchbox import st_searchbox
except ImportError:
    st_searchbox = None


# ======================================================
# 1. CẤU HÌNH WEBSITE
# ======================================================

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide"
)

HERE = Path(__file__).resolve().parent

ROOT = (
    HERE.parent
    if (HERE.parent / "models").is_dir()
    else HERE
)

MODEL_DIR = (
    ROOT / "models"
    if (ROOT / "models").is_dir()
    else ROOT
)

DATA_DIR = (
    ROOT / "data" / "processed"
    if (ROOT / "data" / "processed").is_dir()
    else ROOT
)

RESULT_DIR = (
    ROOT / "results"
    if (ROOT / "results").is_dir()
    else ROOT
)

try:
    TMDB_API_KEY = str(
        st.secrets.get("TMDB_API_KEY", "")
    ).strip()
except Exception:
    TMDB_API_KEY = os.getenv(
        "TMDB_API_KEY", ""
    ).strip()


# ======================================================
# 2. GIAO DIỆN CINEMATCH AI
# ======================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            ellipse at 80% 0%,
            #1b2c57 0%,
            transparent 47%
        ),
        #0a1120;

    color: white;
}

[data-testid="stSidebar"] {
    background: #111b30;
    border-right: 1px solid #30405f;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.7rem;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp p,
.stApp label,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"] {
    color: #f1f5ff !important;
}

.hero {
    padding: 32px;
    border-radius: 20px;

    background: linear-gradient(
        110deg,
        #204889,
        #5e3ca7
    );

    border: 1px solid #6373b0;
    margin-bottom: 22px;
}

.hero h1 {
    margin: 0 0 10px;
    color: white !important;
}

.hero p {
    color: #e6eaff !important;
    margin: 0;
    font-size: 16px;
}

[data-testid="stMetric"] {
    background: #192744;
    border: 1px solid #394d75;
    border-radius: 16px;
    padding: 16px;
}

.stButton button[kind="primary"] {
    background: linear-gradient(
        110deg,
        #496be6,
        #8251d6
    ) !important;

    border: 0 !important;
    color: white !important;
}

.stTextInput input {
    background: #182743 !important;
    color: #ffffff !important;
}

.poster-empty {
    aspect-ratio: 2 / 3;
    border-radius: 12px;

    background: linear-gradient(
        135deg,
        #293d67,
        #111b31
    );

    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;

    font-size: 42px;
}

.poster-empty small {
    font-size: 13px;
    color: #c9d6f6;
}

.db-heading {
    padding: 23px 27px;
    border-radius: 17px;

    background: linear-gradient(
        110deg,
        #1d4287,
        #5539a0
    );

    margin-bottom: 18px;
}

.db-heading h2 {
    margin: 0 0 6px;
    color: #ffffff !important;
}

.db-heading p {
    margin: 0;
    color: #e5ecff !important;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# 3. TẢI MÔ HÌNH
# ======================================================

@st.cache_resource(
    show_spinner="Đang tải mô hình..."
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

    ids = np.asarray(
        hybrid["candidate_movie_ids"],
        dtype=int
    )

    info = cb["movie_info"].copy()

    info["movie_id"] = (
        info["movie_id"].astype(int)
    )

    catalog = (
        info.set_index("movie_id")
        .loc[ids]
        .reset_index()
    )

    cb_indices = catalog[
        "matrix_index"
    ].to_numpy(dtype=int)

    cf_indices = np.asarray(
        [
            cf["movie_to_index"][int(m)]
            for m in ids
        ],
        dtype=int
    )

    return (
        cb,
        cf,
        hybrid,
        train,
        ids,
        catalog,
        cb["tfidf_matrix"][cb_indices].tocsr(),
        cf["item_factors"][cf_indices],
        cf["item_bias"][cf_indices]
    )


@st.cache_data(show_spinner=False)
def load_results():

    files = {
        "topk": "final_test_comparison.csv",
        "rating": "final_test_rating_metrics.csv",
        "coverage": "final_test_coverage.csv"
    }

    return {
        key: pd.read_csv(RESULT_DIR / filename)
        if (RESULT_DIR / filename).is_file()
        else pd.DataFrame()

        for key, filename in files.items()
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

except Exception as error:

    st.error(
        "Không thể tải mô hình hoặc CSV. "
        "Hãy kiểm tra cấu trúc thư mục dự án."
    )

    st.exception(error)
    st.stop()


ALPHA = float(
    hybrid["alpha"]
)

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


# ======================================================
# 4. THỂ LOẠI PHIM TIẾNG VIỆT
# ======================================================

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


def translate_genre(value):

    return ", ".join(
        GENRE_VI.get(x, x)
        for x in str(value).split()
    ) or "Chưa rõ thể loại"


def title_key(value):

    return " ".join(
        str(value).casefold().split()
    )


catalog["thể_loại"] = (
    catalog["genre_text"]
    .fillna("")
    .map(translate_genre)
)

catalog["năm"] = pd.to_numeric(
    catalog["title"]
    .astype(str)
    .str.extract(r"\((\d{4})\)\s*$")[0],
    errors="coerce"
)


# ======================================================
# 5. POSTER PHIM TMDB
# ======================================================

def parse_movie_name(title):

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
    ttl=86400,
    show_spinner=False
)
def get_poster_url(title, api_key):

    if not api_key:
        return None

    name, year = parse_movie_name(title)

    params = {
        "api_key": api_key,
        "query": name,
        "include_adult": "false",
        "language": "en-US"
    }

    attempts = (
        [dict(params, year=year), params]
        if year
        else [params]
    )

    try:

        movies = []

        for query in attempts:

            response = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params=query,
                timeout=(3, 7)
            )

            response.raise_for_status()

            movies = [
                x
                for x in response.json().get(
                    "results", []
                )
                if x.get("poster_path")
            ]

            if movies:
                break

    except (
        requests.RequestException,
        ValueError
    ):
        return None

    if not movies:
        return None

    def match_score(movie):

        similarity = max(
            SequenceMatcher(
                None,
                name.casefold(),
                str(
                    movie.get(k, "")
                ).casefold()
            ).ratio()

            for k in (
                "title",
                "original_title"
            )
        )

        bonus = (
            0.2
            if year and str(
                movie.get("release_date", "")
            )[:4] == str(year)
            else 0
        )

        return similarity + bonus

    best = max(
        movies,
        key=match_score
    )

    if match_score(best) >= 0.55:

        return (
            "https://image.tmdb.org/t/p/w342"
            + best["poster_path"]
        )

    return None


def show_poster(title):

    url = get_poster_url(
        title,
        TMDB_API_KEY
    )

    if url:

        st.image(
            url,
            width="stretch"
        )

    else:

        st.markdown(
            """
            <div class="poster-empty">
                🎬
                <small>Chưa có poster</small>
            </div>
            """,
            unsafe_allow_html=True
        )


# ======================================================
# 6. CONTENT-BASED FILTERING
# ======================================================

def content_scores(user_id):

    profile = cb["user_profiles"].get(
        int(user_id)
    )

    if profile is None:

        return np.zeros(
            len(movie_ids)
        )

    profile = np.asarray(
        profile,
        dtype=float
    ).reshape(1, -1)

    if np.linalg.norm(profile) < 1e-12:

        return np.zeros(
            len(movie_ids)
        )

    similarity = cosine_similarity(
        profile,
        cb_matrix
    ).ravel()

    return (
        np.clip(similarity, -1, 1) + 1
    ) / 2


# ======================================================
# 7. COLLABORATIVE FILTERING SVD
# ======================================================

def collaborative_scores(user_id):

    prediction = np.asarray(
        cf["global_mean"] + cf_bias,
        dtype=float
    ).copy()

    user_idx = cf[
        "user_to_index"
    ].get(int(user_id))

    if user_idx is not None:

        prediction += (
            cf["user_bias"][user_idx]
            + cf_factors
            @ cf["user_factors"][user_idx]
        )

    return (
        np.clip(prediction, 1, 5) - 1
    ) / 4


# ======================================================
# 8. XẾP HẠNG PHIM – LOẠI PHIM TRÙNG
# ======================================================

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
        {
            int(m)
            for m in excluded_ids
        }
        if excluded_ids is not None
        else set()
    )

    valid = np.flatnonzero(
        ~np.isin(
            movie_ids,
            list(excluded)
        )
    )

    order = np.lexsort((
        movie_ids[valid],
        -weighted_ratings[valid],
        -scores[valid]
    ))

    selected = []
    used_names = set()

    for i in valid[order]:

        name = title_key(
            catalog.iloc[i]["title"]
        )

        if name in used_names:
            continue

        used_names.add(name)
        selected.append(i)

        if len(selected) == top_n:
            break

    chosen = np.asarray(
        selected,
        dtype=int
    )

    output = (
        catalog.iloc[chosen]
        .copy()
        .reset_index(drop=True)
    )

    output.insert(
        0,
        "Hạng",
        np.arange(
            1,
            len(output) + 1
        )
    )

    output["score"] = scores[chosen]

    return output, chosen


# ======================================================
# 9. GỢI Ý CHO NGƯỜI DÙNG MOVIELENS
# ======================================================

def recommend_known_user(
    user_id,
    algorithm,
    top_n
):

    cb_s = content_scores(user_id)
    cf_s = collaborative_scores(user_id)

    hy_s = (
        ALPHA * cb_s
        + (1 - ALPHA) * cf_s
    )

    ranking = {
        "Hybrid": hy_s,
        "Content-Based": cb_s,
        "Collaborative Filtering": cf_s
    }[algorithm]

    seen = cb["seen_by_user"].get(
        int(user_id),
        set()
    )

    output, idx = rank_movies(
        ranking,
        excluded_ids=seen,
        top_n=top_n
    )

    output["content_score"] = cb_s[idx]

    output["collaborative_score"] = cf_s[idx]

    output["hybrid_score"] = hy_s[idx]

    return output


# ======================================================
# 10. GỢI Ý CHO NGƯỜI DÙNG MỚI
# ======================================================

def recommend_new_user(
    ratings,
    top_n
):

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

        matrix = cb[
            "tfidf_matrix"
        ][indices]

        profile = np.asarray(
            matrix.multiply(
                np.asarray(
                    weights,
                    dtype=float
                )[:, None]
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

    output, _ = rank_movies(
        scores,
        excluded_ids=ratings.keys(),
        top_n=top_n
    )

    return output, fallback


# ======================================================
# 11. THẺ PHIM – 5 PHIM MỖI HÀNG
# ======================================================

def movie_cards(
    frame,
    show_score=True,
    score_label="Điểm phù hợp",
    n_cols=5
):

    if frame.empty:

        st.info(
            "Không có phim phù hợp."
        )

        return

    if show_score:

        if score_label == "Độ tương đồng thể loại":

            st.caption(
                "ℹ️ Độ giống nhau của vector TF-IDF "
                "thể loại. Điểm 1.000 không có nghĩa "
                "cốt truyện giống hệt nhau."
            )

        else:

            st.caption(
                "ℹ️ Điểm xếp hạng chuẩn hóa 0–1, "
                "không phải xác suất người dùng "
                "thích phim."
            )

    # Tạo từng hàng riêng biệt
    for start in range(
        0,
        len(frame),
        n_cols
    ):

        columns = st.columns(n_cols)

        batch = frame.iloc[
            start:start + n_cols
        ]

        for col, (_, movie) in zip(
            columns,
            batch.iterrows()
        ):

            with col:

                with st.container(border=True):

                    show_poster(
                        movie["title"]
                    )

                    st.markdown(
                        f"**{movie['title']}**"
                    )

                    st.caption(
                        movie["thể_loại"]
                    )

                    if show_score:

                        score = float(
                            movie["score"]
                        )

                        st.markdown(
                            f"**{score_label}: "
                            f"{score:.3f} / 1.000**"
                        )

                        st.progress(
                            float(
                                np.clip(
                                    score,
                                    0,
                                    1
                                )
                            )
                        )

                    else:

                        st.markdown(
                            f"⭐ **{float(movie['weighted_rating']):.2f}/5**"
                        )

                        st.caption(
                            f"{int(movie['rating_count'])} "
                            "lượt đánh giá"
                        )

                    if "hybrid_score" in frame.columns:

                        with st.expander(
                            "Điểm từng thuật toán"
                        ):

                            st.write(
                                f"Content-Based: "
                                f"{float(movie['content_score']):.4f}"
                            )

                            st.write(
                                f"Collaborative SVD: "
                                f"{float(movie['collaborative_score']):.4f}"
                            )

                            st.write(
                                f"Hybrid: "
                                f"{float(movie['hybrid_score']):.4f}"
                            )


# ======================================================
# 12. THANH ĐIỀU HƯỚNG
# ======================================================

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

st.sidebar.write(
    "🧠 Content-Based: TF-IDF"
)

st.sidebar.write(
    "🤝 Collaborative: SVD"
)

st.sidebar.metric(
    "Trọng số Hybrid (α)",
    f"{ALPHA:.2f}"
)

st.sidebar.caption(
    "Dữ liệu: MovieLens 100K"
)


# ======================================================
# 13. TRANG CHỦ
# ======================================================

if page == "🏠 Trang chủ":

    st.markdown(
        """
        <div class="hero">
            <h1>🎬 CineMatch AI</h1>
            <p>
                Hệ thống khuyến nghị phim
                kết hợp Content-Based Filtering
                và Collaborative Filtering (SVD).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "🎞️ Tổng số phim",
        f"{len(cb['movie_info']):,}"
    )

    b.metric(
        "👥 Người dùng",
        f"{len(cb['user_profiles']):,}"
    )

    c.metric(
        "⭐ Đánh giá Train",
        f"{len(train):,}"
    )

    d.metric(
        "🧠 Hybrid Alpha",
        f"{ALPHA:.2f}"
    )

    st.subheader(
        "🔥 Những bộ phim nổi bật"
    )

    popular = (
        catalog.sort_values(
            ["weighted_rating", "rating_count"],
            ascending=False
        )
        .drop_duplicates(
            subset="title"
        )
        .head(8)
    )

    movie_cards(
        popular,
        show_score=False,
        n_cols=4
    )

    st.subheader(
        "📈 Phân bố điểm đánh giá"
    )

    histogram = (
        train["rating"]
        .value_counts()
        .sort_index()
        .rename_axis("Số sao")
        .reset_index(
            name="Lượt đánh giá"
        )
    )

    fig = px.bar(
        histogram,
        x="Số sao",
        y="Lượt đánh giá",
        color="Lượt đánh giá",
        color_continuous_scale="Purples",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ======================================================
# 14. KHÁM PHÁ PHIM
# ======================================================

elif page == "🔎 Khám phá phim":

    st.title(
        "🔎 Khám phá phim"
    )

    st.caption(
        "Tìm kiếm tên phim, lọc thể loại, "
        "năm phát hành và tìm 10 phim tương tự."
    )

    titles = (
        catalog["title"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    def suggest_movies(searchterm):

        q = str(
            searchterm
        ).strip().casefold()

        if not q:
            return []

        found = [
            t
            for t in titles
            if q in t.casefold()
        ]

        found.sort(
            key=lambda t: (
                not t.casefold().startswith(q),
                t.casefold()
            )
        )

        return found[:15]

    c1, c2 = st.columns([2, 1])

    with c1:

        if st_searchbox:

            keyword = st_searchbox(
                suggest_movies,
                label="🔍 Tìm theo tên phim",
                placeholder="Toy Story, Star Wars...",
                key="movie_search",
                default_use_searchterm=True,
                clear_on_submit=False,
                edit_after_submit="option",
                debounce=250,
                style_overrides={
                    "searchbox": {
                        "control": {
                            "backgroundColor": "#192942",
                            "border": "1px solid #798bf2",
                            "borderRadius": "12px"
                        },
                        "input": {
                            "color": "#ffffff"
                        },
                        "singleValue": {
                            "color": "#ffffff"
                        },
                        "placeholder": {
                            "color": "#c2cde9"
                        },
                        "menuList": {
                            "backgroundColor": "#142039"
                        },
                        "option": {
                            "color": "#ffffff",
                            "backgroundColor": "#1b2a49"
                        }
                    }
                }
            ) or ""

        else:

            keyword = st.text_input(
                "🔍 Tìm theo tên phim",
                placeholder="Nhập tên phim..."
            )

    with c2:

        genre = st.selectbox(
            "🎭 Thể loại",
            ["Tất cả"] + list(
                GENRE_VI.values()
            )
        )

    years = catalog["năm"].dropna()

    year_range = None

    if (
        not years.empty
        and int(years.min()) < int(years.max())
    ):

        year_range = st.slider(
            "🗓️ Năm phát hành",
            int(years.min()),
            int(years.max()),
            (
                int(years.min()),
                int(years.max())
            )
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

    if genre != "Tất cả":

        genre_key = next(
            k
            for k, v in GENRE_VI.items()
            if v == genre
        )

        filtered = filtered[
            filtered["genre_text"]
            .fillna("")
            .str.split()
            .map(
                lambda tokens: genre_key in tokens
            )
        ]

    if year_range is not None:

        filtered = filtered[
            filtered["năm"].between(
                *year_range
            )
        ]

    filtered = (
        filtered.drop_duplicates(
            subset="title"
        ).copy()
    )

    st.markdown(
        f"**Tìm thấy {len(filtered):,} phim**"
    )

    table = filtered[
        [
            "title",
            "thể_loại",
            "năm",
            "rating_count",
            "weighted_rating"
        ]
    ].rename(
        columns={
            "title": "Tên phim",
            "thể_loại": "Thể loại",
            "năm": "Năm",
            "rating_count": "Số đánh giá",
            "weighted_rating": "Điểm xếp hạng"
        }
    )

    st.dataframe(
        table.round(2),
        hide_index=True,
        width="stretch",
        height=min(
            365,
            max(
                75,
                (len(table) + 1) * 35 + 8
            )
        )
    )

    st.divider()

    st.subheader(
        "🎯 Tìm 10 phim tương tự"
    )

    if filtered.empty:

        st.info(
            "Không tìm thấy phim phù hợp. "
            "Hãy thay đổi bộ lọc."
        )

    else:

        chosen_id = st.selectbox(
            "Chọn bộ phim",
            filtered["movie_id"]
            .astype(int)
            .tolist(),
            format_func=lambda mid:
                movie_titles.get(
                    int(mid),
                    str(mid)
                )
        )

        if st.button(
            "✨ Tìm phim tương tự",
            type="primary"
        ):

            position = movie_positions[
                int(chosen_id)
            ]

            similarity = cosine_similarity(
                cb_matrix[position],
                cb_matrix
            ).ravel()

            chosen_title = title_key(
                movie_titles[int(chosen_id)]
            )

            excluded = [
                int(mid)
                for mid, title in movie_titles.items()
                if title_key(title) == chosen_title
            ]

            similar, _ = rank_movies(
                similarity,
                excluded_ids=excluded,
                top_n=10
            )

            movie_cards(
                similar,
                score_label="Độ tương đồng thể loại",
                n_cols=5
            )


# ======================================================
# 15. GỢI Ý PHIM
# ======================================================

elif page == "✨ Gợi ý phim":

    st.title(
        "✨ Gợi ý phim dành cho bạn"
    )

    st.caption(
        "Khuyến nghị cá nhân hóa "
        "theo TF-IDF, SVD và Hybrid."
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
        5,
        20,
        10
    )

    if mode == "Người dùng MovieLens":

        users = sorted(
            set(
                map(
                    int,
                    cb["user_profiles"]
                )
            )
            &
            set(
                map(
                    int,
                    cf["user_to_index"]
                )
            )
        )

        if not users:

            st.error(
                "Không có người dùng chung "
                "giữa hai mô hình."
            )

            st.stop()

        c1, c2 = st.columns(2)

        with c1:

            user_id = st.selectbox(
                "👤 Mã người dùng",
                users,
                index=(
                    users.index(196)
                    if 196 in users
                    else 0
                )
            )

        with c2:

            algorithms = {
                "Hybrid":
                    "Hybrid – Kết hợp CB và SVD",

                "Content-Based":
                    "Content-Based – Theo nội dung",

                "Collaborative Filtering":
                    "Collaborative Filtering – SVD"
            }

            algorithm = st.selectbox(
                "🧠 Thuật toán",
                list(algorithms),
                format_func=lambda x:
                    algorithms[x]
            )

        st.info(
            f"Người dùng {user_id} có "
            f"{int((train['user_id'] == user_id).sum())} "
            "đánh giá trong tập Train."
        )

        if st.button(
            "🎬 Đề xuất phim ngay",
            type="primary"
        ):

            recs = recommend_known_user(
                user_id,
                algorithm,
                top_n
            )

            st.success(
                f"Tìm được {len(recs)} phim gợi ý."
            )

            movie_cards(
                recs,
                n_cols=5
            )

            with st.expander(
                "📋 Điểm từng thuật toán dạng bảng"
            ):

                st.dataframe(
                    recs[
                        [
                            "title",
                            "thể_loại",
                            "content_score",
                            "collaborative_score",
                            "hybrid_score"
                        ]
                    ].round(4),
                    hide_index=True,
                    width="stretch"
                )

    else:

        st.info(
            "Chọn 2–5 phim đã xem, "
            "rồi đánh giá mỗi phim từ 1 đến 5 sao."
        )

        ids = (
            catalog.drop_duplicates(
                subset="title"
            )["movie_id"]
            .astype(int)
            .tolist()
        )

        chosen = st.multiselect(
            "🎞️ Phim đã xem",
            ids,
            format_func=lambda mid:
                movie_titles.get(
                    int(mid),
                    str(mid)
                ),
            max_selections=5
        )

        ratings = {
            int(mid): st.slider(
                f"⭐ {movie_titles[int(mid)]}",
                1,
                5,
                4,
                key=f"new_user_{mid}"
            )
            for mid in chosen
        }

        if len(ratings) < 2:

            st.caption(
                "Hãy chọn ít nhất hai phim."
            )

        elif st.button(
            "✨ Phân tích sở thích và gợi ý",
            type="primary"
        ):

            recs, fallback = recommend_new_user(
                ratings,
                top_n
            )

            if fallback:

                st.warning(
                    "Hồ sơ chưa đủ rõ, "
                    "đang sử dụng danh sách phim nổi bật."
                )

            else:

                st.success(
                    "Đã phân tích sở thích!"
                )

            movie_cards(
                recs,
                n_cols=5
            )

        st.caption(
            "Người dùng mới được gợi ý "
            "theo Content-Based do chưa có "
            "lịch sử SVD."
        )


# ======================================================
# 16. DASHBOARD – PHONG CÁCH C
# ĐẸP + HỌC THUẬT + KHÔNG CHỒNG CHỮ
# ======================================================

elif page == "📊 Dashboard":

    # --------------------------------------------------
    # 16.1 BANNER DASHBOARD
    # --------------------------------------------------

    st.markdown(
        """
        <div class="db-heading">
            <h2>📊 Dashboard – Đánh giá CineMatch AI</h2>
            <p>
                Phân tích kết quả thực nghiệm
                trên MovieLens 100K
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"Hybrid: α = {ALPHA:.2f} "
        f"(Content-Based {ALPHA:.0%}, "
        f"Collaborative {1 - ALPHA:.0%}). "
        "Alpha chọn trên Validation; "
        "Test dùng để đánh giá cuối."
    )

    # --------------------------------------------------
    # 16.2 MÀU BIỂU ĐỒ THỐNG NHẤT
    # --------------------------------------------------

    ALGORITHM_COLORS = {
        "Content-Based": "#60a5fa",
        "Collaborative Filtering": "#2dd4bf",
        "Collaborative SVD": "#2dd4bf",
        "Global Mean Baseline": "#60a5fa",
        "Hybrid": "#aa8cf5"
    }

    # --------------------------------------------------
    # 16.3 HÀM BIỂU ĐỒ KHÔNG CHỒNG CHỮ
    # --------------------------------------------------

    def style_chart(
        fig,
        height=420,
        show_legend=True
    ):

        # Không đặt tiêu đề Plotly bên trong biểu đồ.
        # Tiêu đề đã được hiển thị bằng Streamlit.

        fig.update_layout(
            title=None,

            template="plotly_dark",

            height=height,

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="#f7f8ff",
                size=12
            ),

            # Tăng khoảng trống để tránh dính chữ
            margin=dict(
                l=38,
                r=45,
                t=95 if show_legend else 25,
                b=65
            ),

            # Đưa chú thích màu lên riêng
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=0,
                title_text="",
                font=dict(size=12),
                bgcolor="rgba(0,0,0,0)"
            ),

            showlegend=show_legend
        )

        fig.update_xaxes(
            showgrid=False,
            automargin=True
        )

        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.12)",
            automargin=True,
            zeroline=False
        )

        return fig

    # --------------------------------------------------
    # 16.4 ĐỌC KẾT QUẢ CSV
    # --------------------------------------------------

    topk = results["topk"].copy()

    rating_df = results["rating"].copy()

    coverage_df = results["coverage"].copy()

    TOPK = [
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Coverage@10"
    ]

    if (
        not topk.empty
        and "Algorithm" in topk.columns
    ):

        topk["Algorithm"] = (
            topk["Algorithm"]
            .astype(str)
            .str.strip()
        )

        for name in TOPK:

            if name in topk.columns:

                topk[name] = pd.to_numeric(
                    topk[name],
                    errors="coerce"
                )

    else:

        topk = pd.DataFrame()

    # --------------------------------------------------
    # 16.5 BỐN THẺ KPI HYBRID
    # --------------------------------------------------

    st.subheader(
        "🎯 Kết quả Hybrid trên tập Test"
    )

    hybrid_row = (
        topk.loc[
            topk["Algorithm"] == "Hybrid"
        ]
        if not topk.empty
        else pd.DataFrame()
    )

    METRIC_INFO = {
        "Precision@10":
            "Tỷ lệ phim liên quan trong 10 phim đề xuất.",

        "Recall@10":
            "Tỷ lệ phim liên quan được hệ thống tìm thấy.",

        "NDCG@10":
            "Chất lượng thứ tự xếp hạng Top 10.",

        "Coverage@10":
            "Độ bao phủ danh mục phim."
    }

    kpi_columns = st.columns(4)

    for col, name in zip(
        kpi_columns,
        TOPK
    ):

        with col:

            value = (
                pd.to_numeric(
                    hybrid_row.iloc[0].get(name),
                    errors="coerce"
                )
                if not hybrid_row.empty
                else np.nan
            )

            st.metric(
                name,
                f"{float(value):.4f}"
                if pd.notna(value)
                else "—"
            )

            st.caption(
                METRIC_INFO[name]
            )

    if hybrid_row.empty:

        st.warning(
            "Không tìm thấy hàng Hybrid trong "
            "final_test_comparison.csv."
        )

    # --------------------------------------------------
    # 16.6 BA TAB PHÂN TÍCH
    # --------------------------------------------------

    tab_topk, tab_rating, tab_analysis = st.tabs([
        "📈 So sánh Top-K",
        "📉 RMSE · MAE · Coverage",
        "📝 Phân tích & kết luận"
    ])

    # ==================================================
    # TAB 1 – TOP-K
    # ==================================================

    with tab_topk:

        st.subheader(
            "So sánh Precision, Recall và NDCG"
        )

        metrics = [
            m
            for m in TOPK[:3]
            if m in topk.columns
        ]

        if (
            not topk.empty
            and metrics
        ):

            long_df = topk[
                ["Algorithm"] + metrics
            ].melt(
                id_vars="Algorithm",
                var_name="Chỉ số",
                value_name="Giá trị"
            ).dropna(
                subset=["Giá trị"]
            )

            if not long_df.empty:

                fig = px.bar(
                    long_df,

                    x="Chỉ số",
                    y="Giá trị",

                    color="Algorithm",

                    barmode="group",

                    color_discrete_map=ALGORITHM_COLORS,

                    labels={
                        "Algorithm": "Thuật toán"
                    }
                )

                # Hiển thị giá trị trên cột
                fig.update_traces(
                    texttemplate="%{y:.4f}",
                    textposition="outside",
                    cliponaxis=False
                )

                # Chừa khoảng trên đầu cột
                fig.update_yaxes(
                    range=[
                        0,
                        max(
                            0.02,
                            float(
                                long_df["Giá trị"].max()
                            ) * 1.30
                        )
                    ]
                )

                st.plotly_chart(
                    style_chart(fig),
                    width="stretch"
                )

            st.markdown(
                "**Bảng kết quả kiểm thử**"
            )

            st.dataframe(
                topk.round(4),
                hide_index=True,
                width="stretch"
            )

        else:

            st.warning(
                "Không đủ dữ liệu Top-K trong "
                "final_test_comparison.csv."
            )

        with st.expander(
            "📘 Giải thích các chỉ số Top-K"
        ):

            st.markdown(
                """
                **Precision@10**

                Tỷ lệ phim liên quan
                trong 10 phim đề xuất.

                **Recall@10**

                Tỷ lệ phim liên quan
                được hệ thống tìm ra.

                **NDCG@10**

                Đánh giá chất lượng thứ tự
                các phim được xếp hạng.

                **Coverage@10**

                Đánh giá độ bao phủ
                danh mục phim được đề xuất.

                Các chỉ số được tính trên
                dữ liệu Test offline.
                """
            )

    # ==================================================
    # TAB 2 – RMSE, MAE, COVERAGE
    # ==================================================

    with tab_rating:

        st.subheader(
            "Sai số dự đoán rating"
        )

        required_columns = {
            "Algorithm",
            "RMSE",
            "MAE"
        }

        if (
            not rating_df.empty
            and required_columns.issubset(
                rating_df.columns
            )
        ):

            rating_df["RMSE"] = pd.to_numeric(
                rating_df["RMSE"],
                errors="coerce"
            )

            rating_df["MAE"] = pd.to_numeric(
                rating_df["MAE"],
                errors="coerce"
            )

            chosen_rating = rating_df.copy()

            if "Subset" in rating_df.columns:

                subsets = (
                    rating_df["Subset"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if subsets:

                    selected = st.selectbox(
                        "Nhóm đánh giá rating",

                        subsets,

                        index=(
                            subsets.index("All Test")
                            if "All Test" in subsets
                            else 0
                        )
                    )

                    chosen_rating = rating_df[
                        rating_df["Subset"].astype(str)
                        == selected
                    ]

            rating_long = chosen_rating.melt(
                id_vars="Algorithm",

                value_vars=[
                    "RMSE",
                    "MAE"
                ],

                var_name="Chỉ số",
                value_name="Sai số"
            ).dropna(
                subset=["Sai số"]
            )

            if not rating_long.empty:

                fig = px.bar(
                    rating_long,

                    x="Chỉ số",
                    y="Sai số",

                    color="Algorithm",

                    barmode="group",

                    color_discrete_map=ALGORITHM_COLORS
                )

                fig.update_traces(
                    texttemplate="%{y:.4f}",
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_yaxes(
                    range=[
                        0,
                        max(
                            0.1,
                            float(
                                rating_long["Sai số"].max()
                            ) * 1.30
                        )
                    ]
                )

                st.plotly_chart(
                    style_chart(fig),
                    width="stretch"
                )

            else:

                st.info(
                    "Nhóm dữ liệu này không có "
                    "giá trị RMSE/MAE."
                )

            with st.expander(
                "📋 Bảng RMSE và MAE"
            ):

                st.dataframe(
                    rating_df.round(4),
                    hide_index=True,
                    width="stretch"
                )

        else:

            st.info(
                "Thiếu dữ liệu hợp lệ trong "
                "final_test_rating_metrics.csv."
            )

        # ----------------------------------------------
        # COVERAGE
        # ----------------------------------------------

        st.subheader(
            "Độ bao phủ danh mục phim"
        )

        if (
            coverage_df.empty
            or not {
                "Algorithm",
                "Coverage@10"
            }.issubset(coverage_df.columns)
        ):

            coverage_df = (
                topk[
                    [
                        "Algorithm",
                        "Coverage@10"
                    ]
                ].copy()

                if (
                    not topk.empty
                    and "Coverage@10" in topk.columns
                )

                else pd.DataFrame()
            )

        if not coverage_df.empty:

            coverage_df["Coverage@10"] = (
                pd.to_numeric(
                    coverage_df["Coverage@10"],
                    errors="coerce"
                )
            )

            valid_cov = coverage_df.dropna(
                subset=["Coverage@10"]
            )

            if not valid_cov.empty:

                fig = px.bar(
                    valid_cov,

                    x="Coverage@10",
                    y="Algorithm",

                    orientation="h",

                    color="Algorithm",

                    color_discrete_map=ALGORITHM_COLORS
                )

                fig.update_traces(
                    texttemplate="%{x:.4f}",
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_xaxes(
                    range=[
                        0,
                        max(
                            1.08,
                            float(
                                valid_cov[
                                    "Coverage@10"
                                ].max()
                            ) * 1.2
                        )
                    ]
                )

                st.plotly_chart(
                    style_chart(
                        fig,
                        height=320,
                        show_legend=False
                    ),
                    width="stretch"
                )

                st.dataframe(
                    coverage_df.round(4),
                    hide_index=True,
                    width="stretch"
                )

            else:

                st.info(
                    "Không có số liệu Coverage@10 "
                    "hợp lệ."
                )

        else:

            st.info(
                "Thiếu final_test_coverage.csv "
                "hoặc trường Coverage@10."
            )

        st.caption(
            "RMSE và MAE càng thấp càng tốt. "
            "Coverage@10 càng cao cho thấy phạm vi "
            "phim được khuyến nghị càng rộng."
        )

    # ==================================================
    # TAB 3 – PHÂN TÍCH VÀ KẾT LUẬN
    # ==================================================

    with tab_analysis:

        st.subheader(
            "Nhận xét tự động theo dữ liệu Test"
        )

        def best_algorithms(
            frame,
            metric,
            higher_is_better=True
        ):

            if (
                frame.empty
                or not {
                    "Algorithm",
                    metric
                }.issubset(frame.columns)
            ):

                return None

            check = frame[
                [
                    "Algorithm",
                    metric
                ]
            ].copy()

            check[metric] = pd.to_numeric(
                check[metric],
                errors="coerce"
            )

            check = check.dropna(
                subset=[metric]
            )

            if check.empty:
                return None

            best = (
                check[metric].max()
                if higher_is_better
                else check[metric].min()
            )

            names = check.loc[
                np.isclose(
                    check[metric],
                    best,
                    atol=1e-9,
                    rtol=0
                ),
                "Algorithm"
            ].astype(str).drop_duplicates().tolist()

            return (
                ", ".join(names),
                float(best)
            )

        found = False

        # Các chỉ số Top-K
        for metric in TOPK:

            res = best_algorithms(
                topk,
                metric
            )

            if res:

                st.markdown(
                    f"- **{metric}:** "
                    f"{res[0]} đạt giá trị "
                    f"cao nhất (**{res[1]:.4f}**)."
                )

                found = True

        # RMSE và MAE
        if (
            not rating_df.empty
            and {
                "Algorithm",
                "RMSE",
                "MAE"
            }.issubset(rating_df.columns)
        ):

            if "Subset" in rating_df.columns:

                rate_test = rating_df[
                    rating_df["Subset"].astype(str)
                    == "All Test"
                ]

            else:

                rate_test = rating_df

            for metric in [
                "RMSE",
                "MAE"
            ]:

                res = best_algorithms(
                    rate_test,
                    metric,
                    higher_is_better=False
                )

                if res:

                    st.markdown(
                        f"- **{metric} (All Test):** "
                        f"{res[0]} có sai số thấp nhất "
                        f"(**{res[1]:.4f}**)."
                    )

                    found = True

        if not found:

            st.info(
                "Chưa có đủ dữ liệu CSV "
                "để tổng hợp kết quả."
            )

        # ----------------------------------------------
        # NHẬN XÉT HYBRID
        # ----------------------------------------------

        best_ndcg = best_algorithms(
            topk,
            "NDCG@10"
        )

        if best_ndcg:

            if "Hybrid" in best_ndcg[0].split(", "):

                st.success(
                    "Hybrid có NDCG@10 cao nhất "
                    "hoặc đồng hạng. Vẫn cần đối chiếu "
                    "Precision, Recall, Coverage "
                    "và sai số rating."
                )

            else:

                st.info(
                    "Theo NDCG@10, Hybrid chưa đứng đầu. "
                    "Báo cáo nên trình bày kết quả "
                    "thực nghiệm khách quan."
                )

        # ----------------------------------------------
        # CÔNG THỨC HYBRID
        # ----------------------------------------------

        st.subheader(
            "⚖️ Công thức Hybrid"
        )

        st.latex(
            r"S_{Hybrid}=\alpha S_{CB}+(1-\alpha)S_{CF}"
        )

        st.write(
            f"Với α = **{ALPHA:.2f}**, "
            "điểm Hybrid kết hợp "
            f"**{ALPHA:.0%} Content-Based** "
            f"và **{1 - ALPHA:.0%} "
            "Collaborative SVD** "
            "trên thang điểm đã chuẩn hóa."
        )

        st.caption(
            "Alpha được chọn trên Validation, "
            "không điều chỉnh dựa trên điểm Test."
        )

        # ----------------------------------------------
        # HẠN CHẾ NGHIÊN CỨU
        # ----------------------------------------------

        st.subheader(
            "📌 Hạn chế nghiên cứu"
        )

        st.info(
            "Content-Based dựa chủ yếu trên thể loại; "
            "SVD phụ thuộc lịch sử rating. "
            "Độ tương đồng 1.000 chỉ thể hiện vector "
            "đặc trưng cùng hướng. "
            "Kết quả offline không đảm bảo hiệu quả "
            "giống hệt với người dùng thực tế."
        )


# ======================================================
# 17. CHÂN TRANG
# ======================================================

st.divider()

with st.expander(
    "ℹ️ Giới thiệu và nguồn poster"
):

    st.write(
        "CineMatch AI sử dụng MovieLens 100K "
        "và hình ảnh phim từ TMDb. "
        "Một số poster có thể không trùng khớp "
        "hoàn toàn."
    )

    st.markdown(
        "Nguồn poster: "
        "[TMDb](https://www.themoviedb.org/)."
    )

    st.caption(
        "This product uses the TMDB API but is "
        "not endorsed or certified by TMDB."
    )

st.caption(
    "🎬 CineMatch AI · Đề tài 15 · "
    "TF-IDF · SVD · Hybrid · Streamlit"
)
