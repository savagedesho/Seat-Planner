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
  font-weight: 800;
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
  font-weight: 800;
}
.seat .tag.absent{
  border-color: rgba(252,165,165,0.55);
  background: var(--absent);
  font-weight: 800;
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

.floor-grid{
  display:grid;
  grid-template-columns: repeat(2, minmax(260px, 1fr));
  gap: 12px;
  margin-top: 10px;
}
.floor-box{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 16px;
  padding: 12px;
  background: rgba(255,255,255,0.05);
}
.floor-hdr{
  display:flex;
  justify-content: space-between;
  align-items:center;
  margin-bottom: 8px;
}
.floor-title{
  font-weight: 900;
  font-size: 1.05rem;
}
.floor-lines{
  line-height: 1.65;
}
.floor-line{
  display:flex;
  justify-content: space-between;
  gap: 10px;
}
.floor-line .left{
  overflow:hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.floor-line .right{
  color: rgba(255,255,255,0.75);
  white-space: nowrap;
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
        elif ck in ["gender", "sex"]:
            colmap[c] = "Gender"

    df = df.rename(columns=colmap)

    for req in ["School", "Class", "Name", "Grade"]:
        if req not in df.columns:
            df[req] = ""

    for opt in ["Day", "Role", "Status", "Gender"]:
        if opt not in df.columns:
            df[opt] = ""

    for c in ["School", "Class", "Name", "Grade", "Day", "Role", "Status", "Gender"]:
        df[c] = df[c].astype(str).map(norm_text)

    # normalize gender to G/B or blank
    def norm_gender(x: str) -> str:
        xk = norm_key(x)
        if xk in ["g", "girl", "girls", "female", "f"]:
            return "G"
        if xk in ["b", "boy", "boys", "male", "m"]:
            return "B"
        return "" if xk in ["nan", "none"] else norm_text(x)

    df["Gender"] = df["Gender"].map(norm_gender)

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


def detect_absent(row: pd.Series) -> bool:
    status = norm_key(row.get("Status", ""))
    name = norm_key(row.get("Name", ""))
    if "absent" in status or status in ["a", "x", "no", "0"]:
        return True
    if "(absent)" in name or "[absent]" in name:
        return True
    return False


def make_student_key(school: str, clazz: str, name: str, grade: str) -> str:
    return f"{norm_text(school)}||{norm_text(clazz)}||{norm_text(name)}||{norm_text(grade)}"


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


@dataclass
class Seat:
    table_no: int
    seat_no: int
    name: str
    grade: str
    gender: str
    leader: bool
    absent: bool
    student_key: str


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
                "Gender": s.gender,
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
            styles.append("font-weight: 800;")
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
            if s.gender:
                meta_bits.append(f"Gender: {s.gender}")
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


def render_floorplan(seats: List[Seat], seats_per_table: int = 4, cols_per_row: int = 2) -> None:
    """
    Print-friendly "map" view. Tables are numbered (Table 1, Table 2, ...).
    """
    if not seats:
        st.info("No seating generated for this day.")
        return

    # group seats by table number
    tables: Dict[int, List[Seat]] = {}
    for s in seats:
        tables.setdefault(s.table_no, []).append(s)

    # dynamic columns per row (default 2 like your sketch)
    table_numbers = sorted(tables.keys())
    for i in range(0, len(table_numbers), cols_per_row):
        chunk = table_numbers[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for ci in range(cols_per_row):
            with cols[ci]:
                if ci >= len(chunk):
                    st.empty()
                    continue

                tno = chunk[ci]
                members = sorted(tables[tno], key=lambda x: x.seat_no)

                lines = []
                for m in members:
                    if m.name == "(empty)":
                        left = "(empty)"
                        right = ""
                    else:
                        tl = "TL" if m.leader else ""
                        g = m.gender if m.gender else ""
                        right_bits = [b for b in [g, tl] if b]
                        right = " · ".join(right_bits)
                        left = m.name
                    lines.append((left, right))

                html = []
                html.append('<div class="floor-box">')
                html.append('<div class="floor-hdr">')
                html.append(f'<div class="floor-title">Table {tno}</div>')
                html.append(f'<div class="table-pill">{seats_per_table} seats</div>')
                html.append("</div>")
                html.append('<div class="floor-lines">')
                for (left, right) in lines:
                    html.append('<div class="floor-line">')
                    html.append(f'<div class="left">• {left}</div>')
                    html.append(f'<div class="right">{right}</div>')
                    html.append("</div>")
                html.append("</div></div>")
                st.markdown("\n".join(html), unsafe_allow_html=True)


# -------------------------
# Session state
# -------------------------
def init_absence_state(days: List[str]) -> None:
    if "absent_by_day" not in st.session_state:
        st.session_state["absent_by_day"] = {d: set() for d in days}


def set_absences_for_day(day: str, new_set: set) -> None:
    st.session_state["absent_by_day"][day] = set(new_set)


def get_absences_for_day(day: str) -> set:
    return set(st.session_state["absent_by_day"].get(day, set()))


def init_fixed_tables_state(days: List[str]) -> None:
    if "fixed_tables_by_day" not in st.session_state:
        st.session_state["fixed_tables_by_day"] = {d: {"enabled": False, "n_tables": 0} for d in days}


def set_fixed_tables(day: str, enabled: bool, n_tables: int) -> None:
    st.session_state["fixed_tables_by_day"][day] = {"enabled": bool(enabled), "n_tables": int(n_tables)}


def get_fixed_tables(day: str) -> Tuple[bool, int]:
    v = st.session_state["fixed_tables_by_day"].get(day, {"enabled": False, "n_tables": 0})
    return bool(v.get("enabled", False)), int(v.get("n_tables", 0))


def init_tl_state(days: List[str]) -> None:
    # per day: dict of table_no -> student_key (or "")
    if "tl_by_day" not in st.session_state:
        st.session_state["tl_by_day"] = {d: {} for d in days}


def set_tl_for_day(day: str, table_no: int, student_key: str) -> None:
    st.session_state["tl_by_day"].setdefault(day, {})
    st.session_state["tl_by_day"][day][int(table_no)] = student_key or ""


def get_tl_for_day(day: str, table_no: int) -> str:
    return st.session_state["tl_by_day"].get(day, {}).get(int(table_no), "")


def ensure_one_tl_per_table(seats: List[Seat], day: str) -> List[Seat]:
    """
    Apply TL selections from session state onto generated seats.
    Exactly one TL per table (or none) depending on user selection.
    """
    if not seats:
        return seats

    # clear all leaders first
    seats2 = [Seat(**{**s.__dict__, "leader": False}) for s in seats]

    # apply per-table selection
    for s in seats2:
        chosen = get_tl_for_day(day, s.table_no)
        if chosen and s.student_key == chosen and s.name != "(empty)":
            s.leader = True

    return seats2


def gender_balance_pass(
    roster_list: List[dict],
    n_tables: int,
    seats_per_table: int,
    max_same_grade_per_table: int,
) -> List[List[dict]]:
    """
    Greedy seating that tries to avoid all-G/all-B tables by tracking gender counts per table.
    Targets mixed tables where possible, but never blocks placement completely.
    """
    tables: List[List[dict]] = [[] for _ in range(n_tables)]
    grade_counts: List[Dict[str, int]] = [dict() for _ in range(n_tables)]
    gender_counts: List[Dict[str, int]] = [dict() for _ in range(n_tables)]  # {"G":x,"B":y,"":z}

    def gnorm(x: str) -> str:
        x = norm_key(x)
        return "G" if x.startswith("g") else ("B" if x.startswith("b") else "")

    for r in roster_list:
        g = r.get("Grade", "")
        sex = gnorm(r.get("Gender", ""))

        # score tables: prefer tables with fewer people,
        # prefer tables that would improve gender mix,
        # prefer tables that keep grade constraint.
        scored = []
        for ti in range(n_tables):
            if len(tables[ti]) >= seats_per_table:
                continue

            grade_ok = grade_counts[ti].get(g, 0) < max_same_grade_per_table
            # current counts
            gcount = gender_counts[ti].get("G", 0)
            bcount = gender_counts[ti].get("B", 0)

            # gender score: lower is better
            # if table empty -> neutral
            # if placing would create 3-0 or 4-0 -> worse
            after_g = gcount + (1 if sex == "G" else 0)
            after_b = bcount + (1 if sex == "B" else 0)

            gender_penalty = 0
            if sex in ["G", "B"]:
                # penalize moving away from balance
                gender_penalty += abs(after_g - after_b)
                # extra penalty if would create mono table when already skewed
                if (after_g == 0 and after_b >= 3) or (after_b == 0 and after_g >= 3):
                    gender_penalty += 3

            scored.append(
                (
                    0 if grade_ok else 100,               # hard-ish preference
                    len(tables[ti]),                      # fill evenly
                    gender_penalty,                       # balance
                    grade_counts[ti].get(g, 0),           # prefer less same-grade
                    ti,
                )
            )

        if not scored:
            continue

        scored.sort()
        chosen_ti = scored[0][-1]

        tables[chosen_ti].append(r)
        grade_counts[chosen_ti][g] = grade_counts[chosen_ti].get(g, 0) + 1
        gender_counts[chosen_ti][sex] = gender_counts[chosen_ti].get(sex, 0) + 1

    return tables


def generate_plan_for_day(
    df_day: pd.DataFrame,
    day: str,
    seats_per_table: int = 4,
    max_same_grade_per_table: int = 2,
    rotate_offset: int = 0,
    fixed_tables: Optional[int] = None,
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
    auto_tables = max(1, math.ceil(n_students / seats_per_table))
    n_tables = int(fixed_tables) if fixed_tables and fixed_tables > 0 else auto_tables
    n_tables = max(1, n_tables)

    # grade frequency sort (more constrained first)
    grades = [r.get("Grade", "") for r in roster_list]
    freq: Dict[str, int] = {}
    for g in grades:
        freq[g] = freq.get(g, 0) + 1

    roster_list.sort(
        key=lambda r: (
            -freq.get(r.get("Grade", ""), 0),
            grade_key(r.get("Grade", ""))[0],
            norm_text(r.get("Name", "")),
        )
    )

    # gender + grade aware placement
    tables = gender_balance_pass(
        roster_list=roster_list,
        n_tables=n_tables,
        seats_per_table=seats_per_table,
        max_same_grade_per_table=max_same_grade_per_table,
    )

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
                        gender=norm_text(m.get("Gender", "")),
                        leader=False,  # applied later from TL selector
                        absent=detect_absent(pd.Series(m)),
                        student_key=norm_text(m.get("StudentKey", "")),
                    )
                )
            else:
                out.append(
                    Seat(
                        table_no=t_idx,
                        seat_no=s_idx + 1,
                        name="(empty)",
                        grade="",
                        gender="",
                        leader=False,
                        absent=False,
                        student_key="",
                    )
                )

    # apply TL selection (one TL per table)
    out = ensure_one_tl_per_table(out, day=day)
    return out


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="Seat Planner", page_icon="🪑", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

st.title("Seat Planner")
st.caption("4 per table · Day-based rosters · Grade + Gender balancing · Table layout print view · Export")

# global toggle state for floorplan view
if "show_floorplan" not in st.session_state:
    st.session_state["show_floorplan"] = False

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
        "<style>"
        ".block-container{max-width: 1100px;}"
        ".table-card{background:white !important; color:black !important;}"
        ".seat{background:#f6f6f6 !important; color:black !important;}"
        ".seat .meta{color:#333 !important;}"
        ".kpi{background:#f6f6f6 !important; color:black !important;}"
        ".floor-box{background:white !important; color:black !important; border:1px solid #ddd !important;}"
        ".floor-line .right{color:#333 !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

if not uploaded:
    st.info("Upload a CSV/XLSX to begin. Required columns: School, Class, Name, Grade. Optional: Day, Gender (G/B), Status.")
    st.stop()

try:
    df_raw = load_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = ensure_columns(df_raw)
df["StudentKey"] = df.apply(lambda r: make_student_key(r["School"], r["Class"], r["Name"], r["Grade"]), axis=1)

schools, classes_by_school = build_school_class_options(df)

colA, colB = st.columns([1, 1])
with colA:
    selected_school = st.selectbox("School", schools, index=0 if schools else None)
with colB:
    class_opts = classes_by_school.get(selected_school, [])
    selected_class = st.selectbox("Class", class_opts, index=0 if class_opts else None)

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
gender_present = sorted({g for g in filtered["Gender"].astype(str) if g and g.lower() != "nan"})

st.markdown(
    f"""
<div class="kpi-row">
  <div class="kpi"><div class="label">School</div><div class="value">{effective_school}</div></div>
  <div class="kpi"><div class="label">Class</div><div class="value">{effective_class}</div></div>
  <div class="kpi"><div class="label">Rows</div><div class="value">{len(filtered)}</div></div>
  <div class="kpi"><div class="label">Students (rows)</div><div class="value">{total_students}</div></div>
  <div class="kpi"><div class="label">Unique names</div><div class="value">{unique_students}</div></div>
</div>
<p class="small-muted">
Grades: {", ".join(grades_present) if grades_present else "n/a"}
 · Gender: {", ".join(gender_present) if gender_present else "n/a"}
 · Days in file: {", ".join(days_detected) if days_detected else "n/a"}
</p>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)

days = DEFAULT_DAYS
day_map = split_by_day(filtered, days)

# session-only state init
init_absence_state(days)
init_fixed_tables_state(days)
init_tl_state(days)

# -------------------------
# Sidebar: Attendance (session-only) + fixed tables per day
# -------------------------
with st.sidebar:
    st.subheader("Attendance (session-only)")
    st.caption("Tick absent kids for a day. This only lasts for this browser session.")

    attendance_day = st.selectbox("Mark absences for day", days, index=0, key="att_day")
    absent_search = st.text_input("Search student", value="", placeholder="Type a name…", key="att_search")

    df_att = day_map.get(attendance_day, pd.DataFrame(columns=filtered.columns)).copy()
    if not df_att.empty:
        df_att = df_att[df_att["Name"].astype(str).str.strip().ne("")]
        df_att["DefaultAbsent"] = df_att.apply(lambda r: detect_absent(pd.Series(r)), axis=1)

        att_unique = df_att[["StudentKey", "Name", "Grade", "Gender", "DefaultAbsent"]].drop_duplicates(subset=["StudentKey"]).copy()

        current_abs = get_absences_for_day(attendance_day)
        seeded = set(att_unique.loc[att_unique["DefaultAbsent"], "StudentKey"].tolist())
        current_abs = set(current_abs) | seeded
        set_absences_for_day(attendance_day, current_abs)

        if absent_search.strip():
            q = norm_key(absent_search)
            att_unique = att_unique[att_unique["Name"].astype(str).map(lambda x: q in norm_key(x))]

        ui_df = att_unique.copy()
        ui_df["Absent"] = ui_df["StudentKey"].isin(get_absences_for_day(attendance_day))
        ui_df = ui_df[["Absent", "Name", "Grade", "Gender", "StudentKey"]]

        edited = st.data_editor(
            ui_df,
            hide_index=True,
            use_container_width=True,
            height=320,
            column_config={
                "Absent": st.column_config.CheckboxColumn("Absent", help="Tick if absent today"),
                "Name": st.column_config.TextColumn("Name"),
                "Grade": st.column_config.TextColumn("Grade"),
                "Gender": st.column_config.TextColumn("Gender"),
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

    st.subheader("Tables per day")
    st.caption("Set the number of tables for a specific day, or leave on Auto.")
    table_day = st.selectbox("Choose day for tables", days, index=0, key="tbl_day")

    enabled, n_tables = get_fixed_tables(table_day)
    enabled = st.toggle("Set fixed tables for this day", value=enabled, key=f"fixed_enable_{table_day}")
    n_tables = st.number_input("Number of tables", min_value=1, max_value=50, value=max(1, n_tables) if enabled else 6, step=1, key=f"fixed_num_{table_day}")

    set_fixed_tables(table_day, enabled, n_tables)

st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)

# Top controls: layout toggle (button)
b1, b2, b3 = st.columns([1, 1.2, 5])
with b1:
    if st.button("🗺️ Table layout"):
        st.session_state["show_floorplan"] = True
with b2:
    if st.button("🪑 Card view"):
        st.session_state["show_floorplan"] = False
with b3:
    st.caption("Table layout is a print-friendly numbered grid of tables with names. Card view is the detailed seating chart.")

tabs = st.tabs(days)
all_exports = []

for i, day in enumerate(days):
    with tabs[i]:
        df_day = day_map.get(day, pd.DataFrame(columns=filtered.columns)).copy()

        # apply absences: remove absent from generation
        if not df_day.empty:
            abs_set = get_absences_for_day(day)
            if abs_set:
                df_day = df_day[~df_day["StudentKey"].isin(abs_set)].copy()

        # fixed tables for this day
        fixed_enabled, fixed_n = get_fixed_tables(day)
        fixed_tables = fixed_n if fixed_enabled else None

        # generate seats (gender+grade balancing)
        seats = generate_plan_for_day(
            df_day=df_day,
            day=day,
            seats_per_table=int(seats_per_table),
            max_same_grade_per_table=int(max_same_grade),
            rotate_offset=i,
            fixed_tables=fixed_tables,
        )

        # TL selector UI (per day, one TL per table)
        # Build options from seats in this day (NOT from the file), so it always matches what's seated.
        if seats:
            st.markdown('<div class="note">Pick one TL per table for this day. This updates the seating chart + export instantly.</div>', unsafe_allow_html=True)
            n_tables_in_day = max(s.table_no for s in seats)
            # use 2 columns to reduce vertical space
            tl_cols = st.columns(2)
            for tno in range(1, n_tables_in_day + 1):
                # candidates are the seated students in this table (non-empty)
                candidates = [s for s in seats if s.table_no == tno and s.name != "(empty)"]
                # display options
                opts = ["— none —"] + [f"{c.name} (Seat {c.seat_no})" for c in candidates]
                key_map = {f"{c.name} (Seat {c.seat_no})": c.student_key for c in candidates}

                current_key = get_tl_for_day(day, tno)
                current_label = "— none —"
                for label, sk in key_map.items():
                    if sk == current_key:
                        current_label = label
                        break

                with tl_cols[(tno - 1) % 2]:
                    chosen = st.selectbox(f"Table {tno}", opts, index=opts.index(current_label), key=f"tl_{day}_{tno}")
                    set_tl_for_day(day, tno, "" if chosen == "— none —" else key_map.get(chosen, ""))

            # re-apply TL selection after UI changes
            seats = ensure_one_tl_per_table(seats, day=day)

        left, right = st.columns([1.25, 1])

        with left:
            if st.session_state["show_floorplan"]:
                st.subheader("Table layout (numbered, print-friendly)")
                render_floorplan(seats, seats_per_table=int(seats_per_table), cols_per_row=2)
            else:
                st.subheader("Seating chart")
                render_table_cards(seats, seats_per_table=int(seats_per_table))

        with right:
            st.subheader("Table view")
            df_out = seats_to_dataframe(seats, day=day)

            try:
                st.dataframe(style_table(df_out), use_container_width=True, height=440)
            except Exception:
                st.dataframe(df_out, use_container_width=True, height=440)

            all_exports.append(df_out)

            st.caption(
                "Absence for generation is controlled by the sidebar checklist (session-only). "
                "Gender balancing prefers mixed tables (G/B) when possible. TL is selected per-table per-day."
            )

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
