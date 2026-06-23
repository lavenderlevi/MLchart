from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
st.set_page_config(page_title="Materials Band Gap Explorer", layout="wide")
st.title("🔬 Materials Band Gap Explorer")

# ── ĐỌC DỮ LIỆU ───────────────────────────────────────────────────────────────
df = pd.read_csv("materials_bandgap.csv")


# ── TẠO NHÃN ──────────────────────────────────────────────────────────────────
def classify_material(gap):
    if gap < 0.1:
        return "Conductor"
    elif gap <= 3.0:
        return "Semiconductor"
    else:
        return "Insulator"


df["label"] = df["band_gap_eV"].apply(classify_material)

COLOR_MAP = {"Conductor": "#e74c3c", "Semiconductor": "#f39c12", "Insulator": "#2ecc71"}

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📊 Scatter Explorer", "🤖 ML Classifier"])


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCATTER EXPLORER
# ╔══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.sidebar.header("📊 Scatter Explorer")

    # Slider lọc theo band_gap_eV
    gap_min, gap_max = float(df["band_gap_eV"].min()), float(df["band_gap_eV"].max())
    gap_range = st.sidebar.slider(
        "Band Gap (eV)", gap_min, gap_max, (gap_min, gap_max), step=0.1
    )

    # Slider lọc theo n_atoms
    atom_min, atom_max = int(df["n_atoms"].min()), int(df["n_atoms"].max())
    atom_range = st.sidebar.slider(
        "Số nguyên tử (n_atoms)", atom_min, atom_max, (atom_min, atom_max)
    )

    # Số điểm sample
    n_sample = st.sidebar.slider(
        "Số điểm hiển thị", 50, min(500, len(df)), 200, step=50
    )

    # ── LỌC & SAMPLE ──────────────────────────────────────────────────────────
    df_filtered = df[
        (df["band_gap_eV"] >= gap_range[0])
        & (df["band_gap_eV"] <= gap_range[1])
        & (df["n_atoms"] >= atom_range[0])
        & (df["n_atoms"] <= atom_range[1])
    ]

    if len(df_filtered) == 0:
        st.warning("Không có dữ liệu khớp với bộ lọc.")
        st.stop()

    df_sample = df_filtered.sample(n=min(n_sample, len(df_filtered)), random_state=42)
    df_sample = df_sample.reset_index(drop=True)
    df_sample["_id"] = df_sample.index.astype(str)

    st.sidebar.markdown(
        f"**Tìm thấy:** {len(df_filtered):,} dòng → hiển thị {len(df_sample)} điểm"
    )

    # ── TÔ MÀU THEO NHÃN ──────────────────────────────────────────────────────
    tooltip_cols = [c for c in df_sample.columns if c not in ("_id",)]

    scatter = (
        alt.Chart(df_sample)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X(
                "band_gap_eV",
                title="Band Gap (eV)",
                axis=alt.Axis(format=".2f", titleFontSize=13),
            ),
            y=alt.Y("n_atoms", title="Số nguyên tử", axis=alt.Axis(titleFontSize=13)),
            size=alt.value(60),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())
                ),
                legend=alt.Legend(title="Loại vật liệu"),
            ),
            tooltip=tooltip_cols,
        )
    )

    trendline = (
        alt.Chart(df_sample)
        .transform_regression("band_gap_eV", "n_atoms")
        .mark_line(color="white", strokeDash=[6, 3], strokeWidth=2)
        .encode(x="band_gap_eV:Q", y="n_atoms:Q", tooltip=alt.value(None))
    )

    chart = (
        (scatter + trendline)
        .properties(
            title=alt.TitleParams(
                "Band Gap vs Số nguyên tử (màu theo loại vật liệu)",
                fontSize=16,
                anchor="middle",
            ),
            width=700,
            height=450,
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    # ── BẢNG DỮ LIỆU ──────────────────────────────────────────────────────────
    with st.expander("📋 Xem bảng dữ liệu", expanded=False):
        st.dataframe(
            df_filtered.drop(columns=["_id"], errors="ignore"),
            use_container_width=True,
            height=300,
        )
        st.caption(f"Tổng: {len(df_filtered):,} dòng × {len(df_filtered.columns)} cột")


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML CLASSIFIER
# ╔══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.sidebar.divider()
    st.sidebar.header("⚙️ ML Classifier")

    test_size = st.sidebar.slider("Test size (%)", 10, 40, 20) / 100
    n_estimators = st.sidebar.slider("Số cây (n_estimators)", 50, 500, 200, step=50)
    max_depth = st.sidebar.slider("Độ sâu cây (max_depth)", 2, 10, 6)
    learning_rate = st.sidebar.select_slider(
        "Learning rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
    )

    # ── PHÂN BỐ NHÃN ──────────────────────────────────────────────────────────
    st.subheader("📊 Phân bố nhãn trong dataset")

    label_counts = df["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]

    col1, col2, col3 = st.columns(3)
    for col, (_, row) in zip([col1, col2, col3], label_counts.iterrows()):
        col.metric(
            row["label"],
            f"{row['count']:,} mẫu",
            f"{row['count'] / len(df) * 100:.1f}%",
        )

    bar_chart = (
        alt.Chart(label_counts)
        .mark_bar()
        .encode(
            x=alt.X("label", title="Loại vật liệu"),
            y=alt.Y("count", title="Số lượng"),
            color=alt.Color(
                "label",
                scale=alt.Scale(
                    domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())
                ),
                legend=None,
            ),
            tooltip=["label", "count"],
        )
        .properties(height=250)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    st.divider()

    # ── TRAIN MODEL ───────────────────────────────────────────────────────────
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "band_gap_eV"
    ]
    X = df[feature_cols].fillna(df[feature_cols].median())
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    with st.spinner("⏳ Đang huấn luyện mô hình XGBoost..."):
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ── KẾT QUẢ ───────────────────────────────────────────────────────────────
    st.subheader("🎯 Kết quả mô hình")
    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Accuracy", f"{acc * 100:.2f}%")
    m2.metric("📦 Train samples", f"{len(X_train):,}")
    m3.metric("🧪 Test samples", f"{len(X_test):,}")

    st.divider()

    # ── CONFUSION MATRIX ──────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrix")

    labels = le.classes_
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        [
            (labels[i], labels[j], int(cm[i][j]))
            for i in range(len(labels))
            for j in range(len(labels))
        ],
        columns=["Thực tế", "Dự đoán", "Số lượng"],
    )

    heatmap = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("Dự đoán:O"),
            y=alt.Y("Thực tế:O"),
            color=alt.Color("Số lượng:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Thực tế", "Dự đoán", "Số lượng"],
        )
        .properties(width=350, height=300)
    )
    text = (
        alt.Chart(cm_df)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x="Dự đoán:O",
            y="Thực tế:O",
            text="Số lượng:Q",
            color=alt.condition(
                alt.datum["Số lượng"] > cm.max() / 2,
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )
    st.altair_chart(heatmap + text, use_container_width=False)

    st.divider()

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────────────
    st.subheader("📈 Feature Importance")

    fi_df = (
        pd.DataFrame(
            {"Feature": feature_cols, "Importance": model.feature_importances_}
        )
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    fi_chart = (
        alt.Chart(fi_df)
        .mark_bar(color="#3498db")
        .encode(
            x=alt.X("Importance:Q"),
            y=alt.Y("Feature:N", sort="-x", title=""),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(height=400)
    )
    st.altair_chart(fi_chart, use_container_width=True)

    st.divider()

    # ── CLASSIFICATION REPORT ─────────────────────────────────────────────────
    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test, y_pred, target_names=labels, output_dict=True
    )
    report_df = pd.DataFrame(report).T.drop(["accuracy"], errors="ignore").round(3)
    st.dataframe(report_df, use_container_width=True)

    st.divider()

    # ── DỰ ĐOÁN THỬ ───────────────────────────────────────────────────────────
    st.subheader("🧪 Dự đoán thử một mẫu")

    sample_input = {}
    cols = st.columns(min(len(feature_cols), 4))
    for i, feat in enumerate(feature_cols):
        with cols[i % 4]:
            sample_input[feat] = st.number_input(
                feat, value=float(X[feat].mean()), format="%.4f"
            )

    if st.button("🔍 Dự đoán", type="primary"):
        sample_df = pd.DataFrame([sample_input])
        pred_class = le.inverse_transform(model.predict(sample_df))[0]
        proba = model.predict_proba(sample_df)[0]
        color = COLOR_MAP[pred_class]

        st.markdown(
            f"<h3 style='color:{color}'>→ Kết quả: {pred_class}</h3>",
            unsafe_allow_html=True,
        )

        proba_df = pd.DataFrame({"Loại": labels, "Xác suất": proba})
        proba_chart = (
            alt.Chart(proba_df)
            .mark_bar()
            .encode(
                x=alt.X("Xác suất:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Loại:N", sort="-x"),
                color=alt.Color(
                    "Loại:N",
                    scale=alt.Scale(
                        domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())
                    ),
                    legend=None,
                ),
                tooltip=["Loại", alt.Tooltip("Xác suất:Q", format=".2%")],
            )
            .properties(height=150)
        )
        st.altair_chart(proba_chart, use_container_width=True)
