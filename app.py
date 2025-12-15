import random
import datetime as dt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Seat Planner", layout="wide")
st.title("Seat Planner")
st.caption("4 per table • Day-based rosters • Grade balancing • Table-view export")

# ----------------------------
# Registry (your real schools/classes)
# ----------------------------
SCHOOL_CLASS_REGISTRY = {
    "Keikyu School": ["Emerald", "Maroon"],
    "Yako School": ["Yako Class"],
    "Tsukagoshi School": ["Tsukagoshi Class"],
    "Saiwai School": ["Saiwai Class"],
}

SEATS = ["A", "B", "C", "D"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ----------------------------
# Helpers
# ----------------------------
def monday_of_week(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def stable_seed(*parts) -> int:
    """Deterministic seed across runs (simple stable hash)."""
    s = "||".join(str(p) for p in parts)
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) % 2_000_000_000
    return h


def seat_rotation_offset(day: str) -> int:
    if day in DAY_ORDER:
        return DAY_ORDER.index(day) % 4
    return 0


def rotate_seats_map(offset: int):
    offset = offset % 4
    return SEATS[offset:] + SEATS[:offset]


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


def clean_df(df):
    df = df.copy()
    for c in ["school", "class", "day", "name", "grade"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["name"] != "") & (df["name"].str.lower() != "nan")]
    df = df[(df["grade"] != "") & (df["grade"].str.lower() != "nan")]
    df = df[(df["day"] != "") & (df["day"].str.lower() != "nan")]
    return df


def tables_needed(n_students: int) -> int:
    return max(1, (n_students + 3) // 4)


def build_tables_with_constraints(
    present_rows,
    num_tables: int,
    max_same_grade: int | None,
    attempts: int = 400,
    seed: int | None = None,
):
    """
    present_rows: list of dicts: {"name":..., "grade":...}
    Constraint: no table has > max_same_grade of the same grade (if max_same_grade not None).
    """
    rng = random.Random(seed)

    for _ in range(attempts):
        tables = [{"members": [], "grade_counts": {}} for _ in range(num_tables)]

        pool = present_rows[:]
        rng.shuffle(pool)

        ok = True
        for student in pool:
            name = student["name"]
            grade = student["grade"]

            candidates = []
            for t in tables:
                if len(t["members"]) >= 4:
                    continue
                if max_same_grade is not None and t["grade_counts"].get(grade, 0) >= max_same_grade:
                    continue
                candidates.append(t)

            if not candidates:
                ok = False
                break

            candidates.sort(key=lambda t: (len(t["members"]), t["grade_counts"].get(grade, 0)))
            chosen = candidates[0]

            chosen["members"].append((name, grade))
            chosen["grade_counts"][grade] = chosen["grade_counts"].get(grade, 0) + 1

        if ok:
            return tables

    return None


def tables_to_plan(tables, day: str, rotate_seats: bool):
    offset = seat_rotation_offset(day) if rotate_seats else 0
    seat_order = rotate_seats_map(offset)

    plan = {}
    for idx, t in enumerate(tables, start=1):
        plan[idx] = {s: ("", "") for s in SEATS}

        members = t["members"][:]
        # stable-ish order inside table for nicer output
        members.sort(key=lambda x: (x[1], x[0]))  # grade then name

        for i, seat in enumerate(seat_order):
            if i < len(members):
                plan[idx][seat] = members[i]

    return plan


def plan_to_tableview_df(plan, school, class_name, week_start, day):
    rows = []
    for t in sorted(plan.keys()):
        row = {
            "School": school,
            "Class": class_name,
            "WeekStart": week_start.isoformat(),
            "Day": day,
            "Table": t,
        }
        # Seat1..Seat4 (table-view export)
        # Use A/B/C/D as display order
        seat_vals = [plan[t]["A"], plan[t]["B"], plan[t]["C"], plan[t]["D"]]
        for i, (n, g) in enumerate(seat_vals, start=1):
            row[f"Seat{i}"] = f"{n} (G{g})" if n else ""
        rows.append(row)
    return pd.DataFrame(rows)


def render_chart(plan):
    # Visual seating chart: Table cards in a grid
    cols = st.columns(3)
    i = 0
    for t in sorted(plan.keys()):
        with cols[i % 3]:
            st.markdown(f"### Table {t}")
            # 2x2 seat layout
            grid = st.columns(2)
            for idx, seat in enumerate(["A", "B", "C", "D"]):
                name, grade = plan[t][seat]
                label = f"{name} (G{grade})" if name else "—"
                grid[idx % 2].markdown(
                    f"<div style='border:1px solid #ccc;border-radius:10px;padding:10px;margin:6px;text-align:center;'>"
                    f"<div style='font-size:12px;opacity:0.7'>Seat {seat}</div>"
                    f"<div style='font-size:16px;font-weight:600'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        i += 1


# ----------------------------
# Upload
# ----------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

with st.expander("Required columns"):
    st.code("School, Class, Day, Name, Grade", language="text")

if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df = load_file(uploaded)
    require_cols(df, ["school", "class", "day", "name", "grade"])
    df = clean_df(df)
except Exception as e:
    st.error(str(e))
    st.stop()

# ----------------------------
# Fixed dropdowns from registry
# ----------------------------
school = st.selectbox("School", sorted(SCHOOL_CLASS_REGISTRY.keys()))
class_name = st.selectbox("Class", SCHOOL_CLASS_REGISTRY[school])

# Filter file to selected school/class
df_sc = df[(df["school"] == school) & (df["class"] == class_name)].copy()

if df_sc.empty:
    st.warning("No rows found in the uploaded file for this School/Class. Check spelling in the file.")
    st.stop()

# Day options from the file for this class
day_options = sorted(df_sc["day"].unique().tolist())

# ----------------------------
# Controls
# ----------------------------
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    lock_week = st.checkbox("Generate whole week", value=False)
with c2:
    selected_date = st.date_input("Pick a date (for week)", value=dt.date.today())
with c3:
    rotate = st.checkbox("Rotate seats by day", value=True)
with c4:
    enforce = st.checkbox("Limit same-grade per table", value=True)

week_start = monday_of_week(selected_date)
st.caption(f"Week starts: {week_start.isoformat()}")

if enforce:
    max_same = st.number_input("Max same grade per table", min_value=1, max_value=4, value=2, step=1)
else:
    max_same = None

use_seed = st.checkbox("Use seed (repeatable results)", value=True)
seed_val = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, disabled=not use_seed)

if not lock_week:
    day = st.selectbox("Day", day_options)
else:
    # Use Mon–Fri order where possible, but only include days present in the file
    ordered = [d for d in DAY_ORDER if d in day_options] + [d for d in day_options if d not in DAY_ORDER]
    day = None
    day_options = ordered

st.divider()

# ----------------------------
# Generate
# ----------------------------
if st.button("Generate seating", type="primary", use_container_width=True):
    outputs = []  # list of (day, plan, table_df)

    run_days = day_options if lock_week else [day]
    for d in run_days:
        df_day = df_sc[df_sc["day"] == d].copy()
        df_day = df_day.drop_duplicates(subset=["name"])  # dedupe within day only

        absent = st.session_state.get(f"absent_{d}", [])
        present = df_day[~df_day["name"].isin(absent)].copy()

        if present.empty:
            plan = {1: {s: ("", "") for s in SEATS}}
            tdf = plan_to_tableview_df(plan, school, class_name, week_start, d)
            outputs.append((d, plan, tdf))
            continue

        tables = tables_needed(len(present))

        # Build deterministic seed per day if locked-week
        if use_seed:
            seed = stable_seed(seed_val, school, class_name, week_start.isoformat(), d)
        else:
            seed = None

        present_rows = [{"name": r["name"], "grade": r["grade"]} for _, r in present.iterrows()]
        built = build_tables_with_constraints(
            present_rows=present_rows,
            num_tables=int(tables),
            max_same_grade=max_same,
            attempts=500,
            seed=seed,
        )

        if built is None:
            st.error(
                f"Could not satisfy grade constraint for {d}. "
                "Try increasing 'Max same grade per table' or turning the constraint off."
            )
            st.stop()

        plan = tables_to_plan(built, day=d, rotate_seats=rotate)
        tdf = plan_to_tableview_df(plan, school, class_name, week_start, d)
        outputs.append((d, plan, tdf))

    st.session_state["outputs"] = outputs
    st.success("Seating generated.")


# ----------------------------
# Absences UI (per day)
# ----------------------------
st.subheader("Absences (optional)")
if lock_week:
    tabs = st.tabs(day_options)
    for i, d in enumerate(day_options):
        with tabs[i]:
            roster = df_sc[df_sc["day"] == d].drop_duplicates(subset=["name"])["name"].tolist()
            st.session_state[f"absent_{d}"] = st.multiselect("Absent today", roster, key=f"abs_{d}")
else:
    roster = df_sc[df_sc["day"] == day].drop_duplicates(subset=["name"])["name"].tolist()
    st.session_state[f"absent_{day}"] = st.multiselect("Absent today", roster, key=f"abs_{day}")

st.divider()

# ----------------------------
# Display + Export (table view)
# ----------------------------
if "outputs" in st.session_state:
    outputs = st.session_state["outputs"]

    if lock_week:
        st.subheader(f"{school} • {class_name} • Week of {week_start.isoformat()}")
        all_df = pd.concat([tdf for _, _, tdf in outputs], ignore_index=True)

        for d, plan, _ in outputs:
            st.markdown(f"## {d}")
            render_chart(plan)
            st.divider()

        csv_bytes = all_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download WEEK (table view CSV)",
            data=csv_bytes,
            file_name=f"{school}_{class_name}_week_{week_start.isoformat()}.csv".replace(" ", "_"),
            mime="text/csv",
            use_container_width=True
        )

        st.dataframe(all_df, use_container_width=True)

    else:
        d, plan, tdf = outputs[0]
        st.subheader(f"{school} • {class_name} • {d}")
        render_chart(plan)

        csv_bytes = tdf.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download DAY (table view CSV)",
            data=csv_bytes,
            file_name=f"{school}_{class_name}_{d}_week_{week_start.isoformat()}.csv".replace(" ", "_"),
            mime="text/csv",
            use_container_width=True
        )

        st.dataframe(tdf, use_container_width=True)
