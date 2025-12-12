import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Seating Plan App", layout="wide")
st.title("Seating Plan App")
st.caption("Option B: Upload students with Grade. Layout: 4 per table.")

SEATS = ["A", "B", "C", "D"]

def load_students_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    if "name" not in df.columns or "grade" not in df.columns:
        raise ValueError("CSV must have columns: name, grade")
    df = df[["name", "grade"]].dropna()
    df["name"] = df["name"].astype(str).str.strip()
    df["grade"] = df["grade"].astype(str).str.strip()
    df = df[df["name"] != ""]
    return df

def auto_tables(count: int) -> int:
    return max(1, (count + 3) // 4)

def build_plan_balanced(df_present: pd.DataFrame, num_tables: int, seed: int | None = None):
    """
    Balanced fill: round-robin by grade buckets -> spreads grades across tables.
    Returns dict: {table: {seat: (name, grade) or ("","")}}
    """
    if seed is not None:
        random.seed(seed)

    plan = {t: {s: ("", "") for s in SEATS} for t in range(1, num_tables + 1)}
    seat_order = []
    for t in range(1, num_tables + 1):
        for s in SEATS:
            seat_order.append((t, s))

    # group by grade
    buckets = {}
    for _, row in df_present.iterrows():
        g = row["grade"]
        buckets.setdefault(g, []).append(row["name"])

    # shuffle within each grade
    grades = sorted(buckets.keys())
    for g in grades:
        random.shuffle(buckets[g])

    # build round-robin queue: take 1 from each grade repeatedly
    queue = []
    made_progress = True
    while made_progress:
        made_progress = False
        for g in grades:
            if buckets[g]:
                queue.append((buckets[g].pop(0), g))
                made_progress = True

    # fill seats
    i = 0
    for (t, s) in seat_order:
        if i < len(queue):
            plan[t][s] = queue[i]
            i += 1
        else:
            plan[t][s] = ("", "")

    return plan

def plan_to_df(plan):
    rows = []
    for t in sorted(plan.keys()):
        row = {"Table": t}
        for s in SEATS:
            name, grade = plan[t][s]
            row[f"{s}_name"] = name
            row[f"{s}_grade"] = grade
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------- UI ----------------
st.subheader("Upload CSV")
uploaded = st.file_uploader("Upload CSV with columns: name, grade", type=["csv"])

col1, col2, col3 = st.columns([1,1,1])

if uploaded:
    try:
        df = load_students_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.write("Preview:")
    st.dataframe(df, use_container_width=True)

    # absences
    absentees = st.multiselect("Mark absent (excluded)", df["name"].tolist())
    df_present = df[~df["name"].isin(absentees)].copy()

    with col1:
        auto = st.checkbox("Auto-calc tables", value=True)
    with col2:
        seed = st.number_input("Random seed (optional)", min_value=0, max_value=999999, value=0, step=1)
        use_seed = st.checkbox("Use seed", value=False)
    with col3:
        num_tables = st.number_input("Number of tables", min_value=1, max_value=30,
                                     value=auto_tables(len(df_present)), step=1, disabled=auto)

    if auto:
        num_tables = auto_tables(len(df_present))
        st.info(f"Auto tables needed: {num_tables}")

    st.divider()
    generate = st.button("Generate balanced seating", type="primary")

    if generate:
        plan = build_plan_balanced(df_present, int(num_tables), seed=int(seed) if use_seed else None)
        st.session_state.plan = plan
        st.session_state.df_present = df_present

if "plan" in st.session_state and st.session_state.plan:
    plan = st.session_state.plan
    df_out = plan_to_df(plan)

    st.subheader("Seating Plan (balanced by grade)")

    cols = st.columns(3)
    idx = 0
    for t in sorted(plan.keys()):
        with cols[idx % 3]:
            st.markdown(f"### Table {t}")
            for s in SEATS:
                name, grade = plan[t][s]
                if name:
                    st.write(f"{s}: {name} (Grade {grade})")
                else:
                    st.write(f"{s}: —")
        idx += 1

    st.divider()
    st.subheader("Download")
    st.dataframe(df_out, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="seating_plan_balanced_by_grade.csv",
        mime="text/csv",
    )

else:
    st.info("Upload your CSV to begin.")
