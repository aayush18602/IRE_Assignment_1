#!/usr/bin/env python3
"""Assert that CONTEXT.md's task ledger tracks every code item A2_PLAN.md defines.

The two documents split by role: A2_PLAN.md *defines* the relay (item ID, description, files,
size, owner, prerequisites), CONTEXT.md *tracks* it (same IDs, plus status and who finished it
when). The item ID is the join key.

It also checks ordering (an item marked done whose prerequisite is still open means someone built
on code that was never pushed) and that no item is marked done while still owned by TBD.

Exit 0 if consistent, 1 otherwise. Prints progress and names the next item either way. Wire it
into .git/hooks/pre-commit to have git enforce it. A2_PLAN.md is git-ignored, so this skips cleanly
when the plan isn't present (e.g. on a teammate's clone) rather than failing their commits.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "A2_PLAN.md"
CONTEXT = REPO / "CONTEXT.md"

STATUSES = {"todo", "wip", "blocked", "done"}
OWNERS = {"Aayush", "Anurag", "Both", "TBD"}

ID = r"Q\d+-[A-Z]\d?"
# | Q2-A1 | item | files | L | **Aayush** | Q1-E | new |
PLAN_ROW = re.compile(
    rf"^\|\s*({ID})\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([LMS])\s*\|"
    r"\s*\*\*(\w+)\*\*\s*\|\s*(.+?)\s*\|\s*(new|modify)\s*\|$")
# | Q2-A1 | item | Aayush | todo | Aayush, 2026-09-11 |
LEDGER_ROW = re.compile(rf"^\|\s*({ID})\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(.*?)\s*\|$")


def order(sid: str) -> tuple:
    """Sort key for IDs like Q2-A1 -- question number, then the letter/number suffix."""
    q, rest = sid.split("-")
    return (int(q[1:]), rest)


def parse_plan(text: str) -> dict[str, dict]:
    out = {}
    for line in text.splitlines():
        if m := PLAN_ROW.match(line.strip()):
            sid, desc, _files, _size, owner, needs, _kind = m.groups()
            deps = [d.strip() for d in needs.split(",") if re.fullmatch(ID, d.strip())]
            out[sid] = {"desc": desc, "owner": owner, "needs": deps}
    return out


def parse_ledger(text: str) -> dict[str, tuple[str, str, str]]:
    out = {}
    for line in text.splitlines():
        if m := LEDGER_ROW.match(line.strip()):
            out[m.group(1)] = (m.group(3), m.group(4).strip("`"), m.group(5))
    return out


def main() -> int:
    if not PLAN.exists():
        print(f"check_sync: {PLAN.name} not present, nothing to compare -- OK")
        return 0
    if not CONTEXT.exists():
        print(f"check_sync: FAIL -- {CONTEXT.name} is missing")
        return 1

    stages = parse_plan(PLAN.read_text())
    ledger = parse_ledger(CONTEXT.read_text())

    if not stages:
        print("check_sync: FAIL -- no item table found in A2_PLAN.md")
        return 1
    if not ledger:
        print("check_sync: FAIL -- no task ledger found in CONTEXT.md")
        return 1

    problems: list[str] = []

    for sid in sorted(set(stages) - set(ledger), key=order):
        problems.append(f"  untracked: plan defines {sid} ({stages[sid]['desc']}) "
                        f"but the ledger has no row")
    for sid in sorted(set(ledger) - set(stages), key=order):
        problems.append(f"  orphan: ledger has {sid} but the plan defines no such item")

    shared = sorted(set(stages) & set(ledger), key=order)
    for sid in shared:
        owner, status, done_by = ledger[sid]
        if owner not in OWNERS:
            problems.append(f"  owner: {sid} has {owner!r}, not one of {sorted(OWNERS)}")
        elif owner != stages[sid]["owner"]:
            problems.append(f"  owner: plan assigns {sid} to {stages[sid]['owner']}, "
                            f"ledger says {owner!r}")
        if owner == "TBD" and status == "done":
            problems.append(f"  unassigned: {sid} is done but still owned by TBD -- "
                            f"assign it in both files first")
        if status not in STATUSES:
            problems.append(f"  status: {sid} has {status!r}, not one of {sorted(STATUSES)}")
        elif status == "done" and done_by in ("", "—", "-"):
            problems.append(f"  attribution: {sid} is done but has no name/date")

    # Relay ordering: you cannot have finished a stage whose prerequisite never landed.
    done = {s for s in shared if ledger[s][1] == "done"}
    for sid in sorted(done, key=order):
        for dep in stages[sid]["needs"]:
            if dep in stages and dep not in done:
                problems.append(f"  order: {sid} is done but needs {dep}, which is "
                                f"{ledger.get(dep, ('', 'missing', ''))[1]}")

    if problems:
        print("check_sync: FAIL -- CONTEXT.md's ledger does not match A2_PLAN.md\n")
        print("\n".join(problems))
        print("\nFix the ledger, then re-run. See the Sync contract section in A2_PLAN.md.")
        return 1

    print(f"check_sync: OK -- {len(stages)} items, ledger consistent with the plan\n")
    for name in ("Aayush", "Anurag", "Both", "TBD"):
        mine = [s for s in shared if stages[s]["owner"] == name]
        if not mine:
            continue
        n_done = sum(1 for s in mine if ledger[s][1] == "done")
        label = "unassigned" if name == "TBD" else name
        print(f"  {label:<11} {n_done}/{len(mine)} done")

    nxt = next((s for s in shared if ledger[s][1] != "done"), None)
    if nxt:
        blockers = [d for d in stages[nxt]["needs"] if d not in done]
        note = f" -- waiting on {', '.join(blockers)}" if blockers else " -- ready to start"
        print(f"\n  next up: {nxt} ({stages[nxt]['owner']}) {stages[nxt]['desc']}{note}")
    else:
        print("\n  all stages done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
