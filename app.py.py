import streamlit as st
from collections import defaultdict
from typing import List, Dict, Any, Optional

Student = Dict[str, Any]


def make_seating_plan(
    students: List[Student],
    absences: Optional[List[str]] = None,
    rows: int = 4,
    cols: int = 5,
):
    if absences is None:
        absences = []

    present_students = [s for s in students if s["name"] not in absences]

    level_groups: Dict[str, List[Student]] = defaultdict(list)
    for s in present_students:
        level_groups[s.get("level", "unknown")].append(s)

    for lvl in level_groups:
        level_groups[lvl].sort(key=lambda x: x["name"])

    levels = list(level_groups.keys())
    ordered: List[Student] = []
    idx = 0

    while any(level_groups.values()):
        lvl = levels[idx % len(levels)]
        if level_groups[lvl]:
            ordered.append(level_groups[lvl].pop(0))
        idx += 1
        if idx > 10_000:
            break

    front_students = [s for s in ordered if s.get("needs_front")]
    other_students = [s for s in ordered if not s.get("needs_front")]
    final_order = front_students + other_students

    grid = [[None for _ in range(cols)] for _ in range(rows)]
    pos = 0
    for r in range(rows):
        for c in range(cols):
            if pos < len(final_order):
                grid[r][c] = final_order[pos]["name"]
                pos += 1

    return grid


st.title("Classroom Seating Planner ✏️🪑")

st.write("Upload your students and get an automatic seating plan.")

rows = st.number_input("Number of rows", min_value=1, max_value=10, value=4)
cols = st.number_input("Number of columns", min_value=1, max_value=10, value=5)

st.write("Enter students (one per line, format: Name,Level,NeedsFront)")
st.caption("Example: Aiko,low,yes")

default_text = """Aiko,low,yes
Ben,high,no
Chika,mid,yes
Diego,low,no
Emi,high,no
Farid,mid,no
Hana,low,no
"""

students_text = st.text_area("Students", value=default_text, height=200)

absent_text = st.text_input("Absent students today (comma-separated names)", value="Chika")

if st.button("Generate seating plan"):
    # Parse students
    students: List[Student] = []
    for line in students_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            st.warning(f"Skipping line (need 3 items): {line}")
            continue
        name, level, needs_front_raw = parts[:3]
        needs_front = needs_front_raw.lower() in ("yes", "true", "1", "y")
        students.append(
            {"name": name, "level": level, "needs_front": needs_front}
        )

    absences = [n.strip() for n in absent_text.split(",") if n.strip()]

    if not students:
        st.error("No valid students found. Please check your input.")
    else:
        grid = make_seating_plan(students, absences, rows=int(rows), cols=int(cols))

        st.subheader("Seating Plan")
        # Show as a table
        display_grid = []
        for r in range(len(grid)):
            row_display = []
            for c in range(len(grid[0])):
                row_display.append(grid[r][c] or "")
            display_grid.append(row_display)

        st.table(display_grid)

        st.success("Seating plan generated!")
