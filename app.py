import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from difflib import get_close_matches


# -------------------------
# Canonical Schools / Classes
# -------------------------
CANONICAL = {
    "Keikyu School": ["Emerald", "Maroon"],
    "Yako School": ["Yako Class"],
    "Tsukagoshi School": ["Tsukagoshi Class"],
    "Saiwai School": ["Saiwai Class"],
}

DEFAULT_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# -------------------------
# Styling
# -------------------------
APP_CSS = """
<style>
:root{
  --card-bg: rgba(255,255,255,0.06);
  --card-border: rgba(255,255,255,0.10);
  --muted: rgba(255,255,255,0.65);
  --accent: #7dd3fc;
  --good: #86efac;
  --warn: #fde68a;
  --bad: #fca5a5;
  --leader: rgba(253,230,138,0.30);
  --leader-strong: rgba(253,230,138,0.55);
  --absent: rgba(252,165,165,0.30);
}

.block-container{ padding-top: 1.6rem; padding-bottom: 2rem; }
h1, h2, h3{ letter-spacing: .2px; }
.small-muted{ color: var(--muted); font-size: 0.95rem; }

.kpi-row{
  display:flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 14px 0;
}
.kpi{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  padding: 10px 12px;
  border-radius: 12px;
  min-width: 160px;
}
.kpi .label{ color: var(--muted); font-size: 0.85rem; }
.kpi .value{ font-size: 1.2rem; margin-top: 2px; }

.table-grid{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin-top: 10px;
}
.table-card{
  border: 1px solid var(--card-border);
  border-radius: 16px;
  background: var(--card-bg);
  overflow:hidden;
}
.table-card .hdr{
  padding: 10px 12px;
  display:flex;
  align-items:center;
  justify-content: space-between;
  border-bottom: 1px solid var(--card-border);
}
.table-pill{
  font-size: 0.85rem;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--card-border);
  background: rgba(255,255,255,0.04);
  color: rgba(255,255,255,0.85);
}
.seats{
  padding: 10px 12px 12px 12px;
  display:grid;
  gap: 8px;
}
.seat{
  border: 1px solid var(--card-border);
  background: rgba(255,255,255,0.04);
  border-radius: 12px;
  padding: 8px 10px;
  display:flex;
  gap: 10px;
  align-items: flex-start;
}
.seat .badge{
  font-size: 0.72rem;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.85);
  white-space: nowrap;
}
.seat .name{
  font-weight: 700;
  line-height: 1.2;
}
.seat .meta{
  color: var(--muted);
  font-size: 0.85rem;
  margin-top: 2px;
  line-height: 1.2;
}

.seat .tag{
  margin-left:auto;
  font-size: 0.72rem;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.20);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.90);
}
.seat .tag.leader{
  border-color: var(--leader-strong);
  background: var(--leader);
  color: rgba(255,255,255,0.95);
  font-weight: 700;
}
.seat .tag.absent{
  border-color: rgba(252,165,165,0.55);
  background: var(--absent);
  font-weight: 700;
}

.seat.leader{
  border-color: var(--leader-strong);
  background: var(--leader);
  box-shadow: 0 0 0 1px rgba(253,230,138,0.25) inset;
}
.seat.absent{
  opacity: 0.55;
  filter: grayscale(25%);
  text-decoration: line-through;
}

.hr-soft{
  border:0; height:1px;
  background: rgba(255,255,255,0.10);
  margin: 14px 0;
}

.note{
  color: var(--muted);
  font-size: 0.90rem;
}
</style>
"""


# -------------------------
# Helpers
# -------------------------
def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_key(s: str) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def closest_match(query: str, choices: List[str], cutoff: float = 0.72) -> Optional[str]:
    if not query or not choices:
        return None
    matches = get_close_matches(query, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # allow flexible input column names and map to required: School, Class, Name, Grade
    colmap = {}
    for c in df.columns:
        ck = norm_key(c)
        if ck in ["school", "schoolname", "campus"]:
            colmap[c] = "School"
        elif ck in ["class", "classname", "group", "room"]:
            colmap[c] = "Class"
        elif ck in ["name", "student", "studentname", "fullname"]:
            colmap[c] = "Name"
        elif ck in ["grade", "year", "level"]:
            colmap[c] = "Grade"
        elif ck in ["day", "weekday"]:
            colmap[c] = "Day"
        elif ck in ["leader", "teamleader", "tl", "role"]:
            colmap[c] = "Role"
        elif ck in ["attendance", "present", "status"]:
            colmap[c] = "Status"

    df = df.rename(columns=colmap)

    # normalize required columns
    for req in ["School", "Class", "Name", "Grade"]:
        if req not in df.columns:
            df[req] = ""

    # normalize optional columns
    if "Day" not in df.columns:
        df["Day"] = ""
    if "Role" not in df.columns:
        df["Role"] = ""
    if "Status" not in df.columns:
        df["Status"] = ""

    # strip text
    for c in ["School", "Class", "Name", "Grade", "Day", "Role", "Status"]:
        df[c] = df[c].astype(str).map(norm_text)

    return df


def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError("Unsupported file. Please upload CSV or Excel (.xlsx).")


def build_school_class_options(df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    # union canonical + discovered file values
    discovered_schools = sorted({s for s in df["School"].dropna().astype(str).map(norm_text) if s and s.lower() != "nan"})
    discovered_classes = sorted({c for c in df["Class"].dropna().astype(str).map(norm_text) if c and c.lower() != "nan"})

    schools = sorted(set(list(CANONICAL.keys()) + discovered_schools))

    classes_by_school: Dict[str, List[str]] = {}
    for sch in schools:
        base = CANONICAL.get(sch, [])
        in_file = sorted({c for c in df.loc[df["School"] == sch, "Class"].astype(str).map(norm_text) if c and c.lower() != "nan"})
        merged = sorted(set(base + in_file))
        if not merged and discovered_classes:
            merged = discovered_classes
        classes_by_school[sch] = merged

    return schools, classes_by_school


def detect_leader(row: pd.Series) -> bool:
    role = norm_key(row.get("Role", ""))
    name = norm_text(row.get("Name", ""))

    if "leader" in role or role in ["tl", "team leader", "teamleader"]:
        return True
    if re.search(r"\b(tl|leader)\b", norm_key(name)):
        return True
    if "★" in name or "☆" in name:
        return True
    return False


def detect_absent(row: pd.Series) -> bool:
    status = norm_key(row.get("Status", ""))
    name = norm_key(row.get("Name", ""))
    if "absent" in status or status in ["a", "x", "no", "0"]:
        return True
    if "(absent)" in name or "[absent]" in name:
        return True
    return False


def make_student_key(school: str, clazz: str, name: str, grade: str) -> str:
    # stable identity for attendance toggles (avoid same-name collisions across classes)
    return f"{norm_text(school)}||{norm_text(clazz)}||{norm_text(name)}||{norm_text(grade)}"


@dataclass
class Seat:
    table_no: int
    seat_no: int
    name: str
    grade: str
    leader: bool
    absent: bool


def grade_key(g: str) -> Tuple[int, str]:
    t = norm_key(g)
    m = re.search(r"(\d+)", t)
    if m:
        return (int(m.group(1)), t)
    return (999, t)


def split_by_day(df: pd.DataFrame, days: List[str]) -> Dict[str, pd.DataFrame]:
    if df["Day"].astype(str).str.strip().replace("nan", "").eq("").all():
        return {d: df.copy() for d in days}

    def normalize_day(x: str) -> str:
        xk = norm_key(x)
        if xk.startswith("mon"):
            return "Mon"
        if xk.startswith("tue"):
            return "Tue"
        if xk.startswith("wed"):
            return "Wed"
        if xk.startswith("thu"):
            return "Thu"
        if xk.startswith("fri"):
            return "Fri"
        if xk.startswith("sat"):
            return "Sat"
        if xk.startswith("sun"):
            return "Sun"
        return norm_text(x)

    temp = df.copy()
    temp["DayNorm"] = temp["Day"].astype(str).map(normalize_day)

    out: Dict[str, pd.DataFrame] = {}
    for d in days:
        out[d] = temp.loc[temp["DayNorm"] == d].drop(columns=["DayNorm"]).copy()

    return out


def generate_plan_for_day(
    df_day: pd.DataFrame,
    seats_per_table: int = 4,
    max_same_grade_per_table: int = 2,
    rotate_offset: int = 0,
) -> List[Seat]:
    roster = df_day.copy()
    roster = roster[roster["Name"].astype(str).str.strip().ne("")]
    if roster.empty:
        return []

    roster = roster.sort_values(
        ["Grade", "Name"],
        key=lambda s: s.map(lambda v: grade_key(v)[0] if s.name == "Grade" else norm_text(v)),
    )
    roster_list = roster.to_dict(orient="records")

    if roster_list:
        rotate_offset = rotate_offset % len(roster_list)
        roster_list = roster_list[rotate_offset:] + roster_list[:rotate_offset]

    n_students = len(roster_list)
    n_tables = max(1, math.ceil(n_students / seats_per_table))

    tables: List[List[dict]] = [[] for _ in range(n_tables)]
    grade_counts: List[Dict[str, int]] = [dict() for _ in range(n_tables)]

    grades = [r.get("Grade", "") for r in roster_list]
    freq: Dict[str, int] = {}
    for g in grades:
        freq[g] = freq.get(g, 0) + 1

    roster_list.sort(key=lambda r: (-freq.get(r.get("Grade", ""), 0), grade_key(r.get("Grade", ""))[0], norm_text(r.get("Name", ""))))

    for r in roster_list:
        g = r.get("Grade", "")
        placed = False

        table_order = sorted(range(n_tables), key=lambda i: (len(tables[i]), grade_counts[i].get(g, 0)))

        for ti in table_order:
            if len(tables[ti]) >= seats_per_table:
                continue
            if grade_counts[ti].get(g, 0) >= max_same_grade_per_table:
                continue
            tables[ti].append(r)
            grade_counts[ti][g] = grade_counts[ti].get(g, 0) + 1
            placed = True
            break

        if not placed:
            for ti in table_order:
                if len(tables[ti]) < seats_per_table:
                    tables[ti].append(r)
                    grade_counts[ti][g] = grade_counts[ti].get(g, 0) + 1
                    break

    out: List[Seat] = []
    for t_idx, members in enumerate(tables, start=1):
        for s_idx in range(seats_per_table):
            if s_idx < len(members):
                m = members[s_idx]
                out.append(
                    Seat(
                        table_no=t_idx,
                        seat_no=s_idx + 1,
                        name=norm_text(m.get("Name", "")),
                        grade=norm_text(m.get("Grade", "")),
                        leader=detect_leader(pd.Series(m)),
                        absent=detect_absent(pd.Series(m)),
                    )
                )
            else:
                out.append(
                    Seat(
                        table_no=t_idx,
                        seat_no=s_idx + 1,
                        name="(empty)",
                        grade="",
                        leader=False,
                        absent=False,
                    )
                )
    return out


def seats_to_dataframe(seats: List[Seat], day: str) -> pd.DataFrame:
    rows = []
    for s in seats:
        rows.append(
            {
                "Day": day,
                "Table": s.table_no,
                "Seat": s.seat_no,
                "Name": s.name,
                "Grade": s.grade,
                "Leader": "TL" if s.leader else "",
                "Absent": "Yes" if s.absent else "",
            }
        )
    return pd.DataFrame(rows)


def style_table(df_out: pd.DataFrame):

    def row_style(r):
        styles = []
        is_leader = str(r.get("Leader", "")).strip() != ""
        is_absent = str(r.get("Absent", "")).strip() != ""
        if is_absent:
            styles.append("opacity:0.55;")
            styles.append("text-decoration: line-through;")
        if is_leader:
            styles.append("background-color: rgba(253,230,138,0.18);")
            styles.append("font-weight: 700;")
        return ["".join(styles)] * len(r)

    return df_out.style.apply(row_style, axis=1)


def render_table_cards(seats: List[Seat], seats_per_table: int = 4) -> None:
    if not seats:
        st.info("No seating generated for this day.")
        return

    tables: Dict[int, List[Seat]] = {}
    for s in seats:
        tables.setdefault(s.table_no, []).append(s)

    html = ['<div class="table-grid">']
    for tno in sorted(tables.keys()):
        members = sorted(tables[tno], key=lambda x: x.seat_no)
        html.append('<div class="table-card">')
        html.append('<div class="hdr">')
        html.append(f'<div class="name">Table {tno}</div>')
        html.append(f'<div class="table-pill">{seats_per_table} seats</div>')
        html.append("</div>")
        html.append('<div class="seats">')

        for s in members:
            is_empty = s.name == "(empty)"
            cls = "seat"
            if s.leader:
                cls += " leader"
            if s.absent:
                cls += " absent"

            badge = f"Seat {s.seat_no}"
            display_name = s.name if not is_empty else "(empty)"
            meta_bits = []
            if s.grade:
                meta_bits.append(f"Grade: {s.grade}")
            meta = " · ".join(meta_bits) if meta_bits else ""

            tag_html = ""
            if s.leader and not is_empty:
                tag_html += '<span class="tag leader">TL</span>'
            if s.absent and not is_empty:
                tag_html += '<span class="tag absent">ABSENT</span>'

            html.append(f'<div class="{cls}">')
            html.append(f'<div class="badge">{badge}</div>')
            html.append('<div style="flex:1;">')
            html.append(f'<div class="name">{display_name}</div>')
            if meta:
                html.append(f'<div class="meta">{meta}</div>')
            html.append("</div>")
            if tag_html:
                html.append(f'<div style="display:flex; gap:6px; align-items:flex-start;">{tag_html}</div>')
            html.append("</div>")

        html.append("</div>")
        html.append("</div>")

    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)


def init_absence_state(days: List[str]) -> None:
    if "absent_by_day" not in st.session_state:
        st.session_state["absent_by_day"] = {d: set() for d in days}


def set_absences_for_day(day: str, new_set: set) -> None:
    st.session_state["absent_by_day"][day] = set(new_set)


def get_absences_for_day(day: str) -> set:
    return set(st.session_state["absent_by_day"].get(day, set()))


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="Seat Planner", page_icon="🪑", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

st.title("Seat Planner")
st.caption("4 per table · Day-based rosters · Grade balancing · Table-view export")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)

    seats_per_table = st.number_input("Seats per table", min_value=2, max_value=10, value=4, step=1)
    max_same_grade = st.number_input("Max same grade per table", min_value=1, max_value=4, value=2, step=1)

    st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)

    auto_correct = st.toggle("Auto-correct School/Class mismatches", value=True)
    strict_match = st.toggle("Strict match only (no correction)", value=False)

    st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)
    export_mode = st.selectbox("Export format", ["CSV", "Excel"], index=0)
    show_print_mode = st.toggle("Print-friendly mode", value=False)

if show_print_mode:
    st.markdown(
        "<style>.block-container{max-width: 1050px;} .table-card{background:white !important; color:black !important;} .seat{background:#f6f6f6 !important; color:black !important;} .seat .meta{color:#333 !important;} .kpi{background:#f6f6f6 !important; color:black !important;}</style>",
        unsafe_allow_html=True,
    )

if not uploaded:
    st.info("Upload a CSV/XLSX to begin. Required columns: School, Class, Name, Grade (Day optional).")
    st.stop()

try:
    df_raw = load_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = ensure_columns(df_raw)

# add stable student key (for session-only attendance)
df["StudentKey"] = df.apply(lambda r: make_student_key(r["School"], r["Class"], r["Name"], r["Grade"]), axis=1)

schools, classes_by_school = build_school_class_options(df)

# selectors
colA, colB = st.columns([1, 1])
with colA:
    selected_school = st.selectbox("School", schools, index=0 if schools else None)
with colB:
    class_opts = classes_by_school.get(selected_school, [])
    selected_class = st.selectbox("Class", class_opts, index=0 if class_opts else None)

# mismatch handling
df_school_vals = sorted({s for s in df["School"].map(norm_text) if s and s.lower() != "nan"})
df_class_vals = sorted({c for c in df["Class"].map(norm_text) if c and c.lower() != "nan"})

school_in_file = selected_school in df_school_vals
class_in_file = selected_class in df_class_vals

suggest_school = closest_match(selected_school, df_school_vals, cutoff=0.65) if not school_in_file else None
suggest_class = closest_match(selected_class, df_class_vals, cutoff=0.65) if not class_in_file else None

effective_school = selected_school
effective_class = selected_class

if strict_match:
    auto_correct = False

if auto_correct:
    if (not school_in_file) and suggest_school:
        effective_school = suggest_school
        st.warning(f"School mismatch: using closest match found in file: {suggest_school}")
    elif not school_in_file and df_school_vals:
        st.warning("School mismatch: no close match found. Check spelling in the file.")

    if (not class_in_file) and suggest_class:
        effective_class = suggest_class
        st.warning(f"Class mismatch: using closest match found in file: {suggest_class}")
    elif not class_in_file and df_class_vals:
        st.warning("Class mismatch: no close match found. Check spelling in the file.")
else:
    if not school_in_file and df_school_vals:
        msg = "School not found in file."
        if suggest_school:
            msg += f" Suggestion: {suggest_school}"
        st.warning(msg)
    if not class_in_file and df_class_vals:
        msg = "Class not found in file."
        if suggest_class:
            msg += f" Suggestion: {suggest_class}"
        st.warning(msg)

# filter
filtered = df[(df["School"] == effective_school) & (df["Class"] == effective_class)].copy()

if filtered.empty:
    st.error("No rows found for the selected School/Class (after any auto-correction).")
    with st.expander("Quick diagnostic"):
        st.write("Schools found in file:", df_school_vals[:50])
        st.write("Classes found in file:", df_class_vals[:50])
        st.write("First 20 rows preview:")
        st.dataframe(df.head(20), use_container_width=True)
    st.stop()

# KPIs
total_students = filtered["Name"].astype(str).str.strip().ne("").sum()
unique_students = filtered["Name"].astype(str).str.strip().nunique()
grades_present = sorted({g for g in filtered["Grade"].astype(str) if g and g.lower() != "nan"})
days_detected = sorted({d for d in filtered["Day"].astype(str) if d and d.lower() != "nan"})

st.markdown(
    f"""
<div class="kpi-row">
  <div class="kpi"><div class="label">School</div><div class="value">{effective_school}</div></div>
  <div class="kpi"><div class="label">Class</div><div class="value">{effective_class}</div></div>
  <div class="kpi"><div class="label">Rows</div><div class="value">{len(filtered)}</div></div>
  <div class="kpi"><div class="label">Students (rows)</div><div class="value">{total_students}</div></div>
  <div class="kpi"><div class="label">Unique names</div><div class="value">{unique_students}</div></div>
</div>
<p class="small-muted">Grades in file: {", ".join(grades_present) if grades_present else "n/a"} · Days in file: {", ".join(days_detected) if days_detected else "n/a"}</p>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)

# day tabs + day map
days = DEFAULT_DAYS
day_map = split_by_day(filtered, days)

# -------------------------
# Session-only Attendance UI (sidebar)
# -------------------------
init_absence_state(days)

with st.sidebar:
    st.subheader("Attendance (session-only)")
    st.caption("Tick absent kids for a day. This only lasts for this browser session.")

    attendance_day = st.selectbox("Mark absences for day", days, index=0)
    absent_search = st.text_input("Search student", value="", placeholder="Type a name…")

    df_att = day_map.get(attendance_day, pd.DataFrame(columns=filtered.columns)).copy()
    if not df_att.empty:
        # unique list for the day
        df_att = df_att[df_att["Name"].astype(str).str.strip().ne("")]
        df_att["IsLeader"] = df_att.apply(lambda r: detect_leader(pd.Series(r)), axis=1)
        df_att["DefaultAbsent"] = df_att.apply(lambda r: detect_absent(pd.Series(r)), axis=1)

        att_unique = (
            df_att[["StudentKey", "Name", "Grade", "IsLeader", "DefaultAbsent"]]
            .drop_duplicates(subset=["StudentKey"])
            .copy()
        )

        # seed with file-based absent markers + existing session state
        current_abs = get_absences_for_day(attendance_day)
        seeded = set(att_unique.loc[att_unique["DefaultAbsent"], "StudentKey"].tolist())
        current_abs = set(current_abs) | seeded
        set_absences_for_day(attendance_day, current_abs)

        # filter by search
        if absent_search.strip():
            q = norm_key(absent_search)
            att_unique = att_unique[att_unique["Name"].astype(str).map(lambda x: q in norm_key(x))]

        # show editor checklist (best "click to mark absent" UX)
        ui_df = att_unique.copy()
        ui_df["Absent"] = ui_df["StudentKey"].isin(get_absences_for_day(attendance_day))
        ui_df["TL"] = ui_df["IsLeader"].map(lambda x: "TL" if x else "")
        ui_df = ui_df[["Absent", "Name", "Grade", "TL", "StudentKey"]]

        edited = st.data_editor(
            ui_df,
            hide_index=True,
            use_container_width=True,
            height=320,
            column_config={
                "Absent": st.column_config.CheckboxColumn("Absent", help="Tick if absent today"),
                "Name": st.column_config.TextColumn("Name"),
                "Grade": st.column_config.TextColumn("Grade"),
                "TL": st.column_config.TextColumn("TL"),
                "StudentKey": st.column_config.TextColumn("StudentKey", disabled=True),
            },
            disabled=["StudentKey"],
            key=f"att_editor_{effective_school}_{effective_class}_{attendance_day}",
        )

        new_abs = set(edited.loc[edited["Absent"] == True, "StudentKey"].tolist())
        set_absences_for_day(attendance_day, new_abs)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear absences (this day)"):
                set_absences_for_day(attendance_day, set())
                st.rerun()
        with c2:
            st.write(f"Absent: {len(get_absences_for_day(attendance_day))}")

    else:
        st.info("No roster rows for this day (based on your file).")

st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)

tabs = st.tabs(days)
all_exports = []

for i, day in enumerate(days):
    with tabs[i]:
        df_day = day_map.get(day, pd.DataFrame(columns=filtered.columns)).copy()

        # apply session-only absences: REMOVE absent kids from generation for this day
        if not df_day.empty:
            abs_set = get_absences_for_day(day)
            if abs_set:
                df_day = df_day[~df_day["StudentKey"].isin(abs_set)].copy()

        rotate_offset = i

        seats = generate_plan_for_day(
            df_day=df_day,
            seats_per_table=int(seats_per_table),
            max_same_grade_per_table=int(max_same_grade),
            rotate_offset=rotate_offset,
        )

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Seating chart")
            render_table_cards(seats, seats_per_table=int(seats_per_table))

        with right:
            st.subheader("Table view")
            df_out = seats_to_dataframe(seats, day=day)

            # stronger table visuals
            try:
                st.dataframe(style_table(df_out), use_container_width=True, height=440)
            except Exception:
                st.dataframe(df_out, use_container_width=True, height=440)

            all_exports.append(df_out)

            st.caption("TL highlighting uses Role column (Leader/TL) or ★ / '(TL)' in the name. Absence for generation is controlled by the sidebar checklist (session-only).")

# export
export_df = pd.concat(all_exports, ignore_index=True) if all_exports else pd.DataFrame()

st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)
st.subheader("Export")

if export_df.empty:
    st.info("Nothing to export yet.")
else:
    if export_mode == "CSV":
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download seating plan (CSV)",
            data=csv_bytes,
            file_name=f"seating_plan_{norm_key(effective_school)}_{norm_key(effective_class)}.csv",
            mime="text/csv",
        )
    else:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="SeatingPlan")
        st.download_button(
            "Download seating plan (Excel)",
            data=out.getvalue(),
            file_name=f"seating_plan_{norm_key(effective_school)}_{norm_key(effective_class)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with st.expander("Preview uploaded data"):
    st.dataframe(filtered.head(50), use_container_width=True)
