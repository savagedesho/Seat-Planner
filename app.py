import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Seat Planner", layout="wide")
st.title("Seat Planner")
st.caption("Day-based seating • 4 per table • Grade balancing + constraints")

SEATS = ["A", "B", "C", "D"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]

def load_file(uploaded):
    if uploaded.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def clean(df):
    df = df.copy()
    for c in ["school", "class", "day", "name", "grade"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["name"] != "") & (df["name"].str.lower() != "nan")]
    df = df[(df["grade"] != "") & (df["grade"].str.lower() != "nan")]
    return df

def tables_needed(n): 
    return max(1, (n + 3) // 4)

def seat_rotation_offset(day: str) -> int:
    # Mon=0, Tue=1, Wed=2, Thu=3, Fri=0 (wrap)
    if day in DAY_ORDER:
        return DAY_ORDER.index(day) % 4
    return 0

def rotate_seats_map(offset: int):
    # returns seats in rotated order for assignment
    offset = offset % 4
    return SEATS[offset:] + SEATS[:offset]

def build_tables_with_constraints(present_rows, num_tables: int, max_same_grade: int, attempts: int = 200, seed=None):
    """
    present_rows: list of dicts: {"name":..., "grade":...}
    Returns tables: list of dicts:
      [{"members": [("Name","Grade"),...], "grade_counts": {grade:count}}, ...]
    """
    if seed is not None:
        random.seed(seed)

    for _ in range(attempts):
        # init
        tables = [{"members": [], "grade_counts": {}} for _ in range(num_tables)]

        pool = present_rows[:]
        random.shuffle(pool)

        ok = True
        for student in pool:
            name = student["name"]
            grade = student["grade"]

            # candidate tables: not full and grade count < max
            candidates = []
            for t in tables:
                if len(t["members"]) >= 4:
                    continue
                if t["grade_counts"].get(grade, 0) >= max_same_grade:
                    continue
                candidates.append(t)

            if not candidates:
                ok = False
                break

            # choose table with smallest size, then smallest count of that grade
            candidates.sort(key=lambda t: (len(t["members"]), t["grade_counts"].get(grade, 0)))
            chosen = candidates[0]

            chosen["members"].append((name, grade))
            chosen["grade_counts"][grade] = chosen["grade_counts"].get(grade, 0) + 1

        if ok:
            return tables

    return None  # failed after attempts

def tables_to_plan(tables, day: str, rotate_seats: bool):
    """
    Convert tables (members only) into a plan with seat letters.
    If rotate_seats: seat letters shift by day.
    """
    offset = seat_rotation_offset(day) if rotate_seats else 0
    seat_order = rotate_seats_map(offset)

    plan = {}
    for idx, t in enumerate(tables, start=1):
        plan[idx] = {s: ("", "") for s in SEATS}

        # assign members to seats (stable order so tables stay consistent)
        members = t["members"][:]
        # keep deterministic order inside table for nicer outputs
        members.sort(key=lambda x: (x[1], x[0]))  # grade then name

        for i, seat in enumerate(seat_order):
            if i < len(members):
                plan[idx][seat] = members[i]

    return plan

def plan_to_df(plan):
    rows=[]
    for t in sorted(plan.keys()):
        r={"Table": t}
        for s in SEATS:
            n,g = plan[t][s]
            r[f"{s}_Name"] = n
            r[f"{s}_Grade"] = g
        rows.append(r)
    return pd.DataFrame(rows)

# -------- Upload --------
up = st.file_uploader("Upload CSV or Excel with: School, Class, Day, Name, Grade", type=["csv","xlsx"])
if not up:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df = load_file(up)
    require_cols(df, ["school","class","day","name","grade"])
    df = clean(df)
except Exception as e:
    st.error(str(e))
    st.stop()

# -------- Selectors --------
c1, c2, c3 = st.columns(3)
with c1:
    school = st.selectbox("School", sorted(df["school"].unique()))
with c2:
    class_name = st.selectbox("Class", sorted(df[df["school"]==school]["class"].unique()))
with c3:
    day = st.selectbox("Day", sorted(df[(df["school"]==school)&(df["class"]==class_name)]["day"].unique()))

df_day = df[(df["school"]==school)&(df["class"]==class_name)&(df["day"]==day)].copy()
df_day = df_day.drop_duplicates(subset=["name"])  # dedupe within selected day only

st.write("Day roster:")
st.dataframe(df_day, use_container_width=True)

# -------- Absences & settings --------
absent = st.multiselect("Absent today", df_day["name"].tolist())
present = df_day[~df_day["name"].isin(absent)].copy()

s1, s2, s3, s4 = st.columns(4)
with s1:
    auto = st.checkbox("Auto tables", value=True)
with s2:
    max_same = st.number_input("Max same grade per table", min_value=1, max_value=4, value=2, step=1)
with s3:
    rotate = st.checkbox("Rotate seats by day", value=True)
with s4:
    use_seed = st.checkbox("Use seed", value=False)

seed = st.number_input("Seed", 0, 999999, 42, disabled=not use_seed)

tables = tables_needed(len(present))
if not auto:
    tables = st.number_input("Tables", 1, 50, tables, step=1)

st.info(f"Tables needed: {tables} (4 per table)")

st.divider()
if st.button("Generate seating", type="primary", use_container_width=True):
    if present.empty:
        st.warning("No present students to seat (everyone is absent).")
        st.stop()

    present_rows = [{"name": r["name"], "grade": r["grade"]} for _, r in present.iterrows()]

    built = build_tables_with_constraints(
        present_rows=present_rows,
        num_tables=int(tables),
        max_same_grade=int(max_same),
        attempts=400,
        seed=int(seed) if use_seed else None
    )

    if built is None:
        st.error(
            "Could not satisfy the grade constraint with the current roster. "
            "Try increasing 'Max same grade per table', increasing tables, or disabling the constraint."
        )
        st.stop()

    st.session_state.plan = tables_to_plan(built, day=day, rotate_seats=rotate)
    st.session_state.meta = {"school": school, "class": class_name, "day": day, "tables": int(tables)}

# -------- Output --------
if "plan" in st.session_state:
    plan = st.session_state.plan
    meta = st.session_state.meta
    out = plan_to_df(plan)

    st.subheader(f"{meta['school']} • {meta['class']} • {meta['day']}")

    cols = st.columns(3)
    i = 0
    for t in sorted(plan.keys()):
        with cols[i%3]:
            st.markdown(f"### Table {t}")
            for s in SEATS:
                n,g = plan[t][s]
                st.write(f"{s}: {n} (G{g})" if n else f"{s}: —")
        i += 1

    st.divider()
    st.dataframe(out, use_container_width=True)
    st.download_button(
        "Download CSV",
        out.to_csv(index=False).encode("utf-8-sig"),
        f"seating_{meta['school']}_{meta['class']}_{meta['day']}.csv".replace(" ","_"),
        "text/csv",
        use_container_width=True
    )
