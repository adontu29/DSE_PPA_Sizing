from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs")
SWEEP_CSV = OUTPUT_DIR / "wing_area_sweep.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
PLOT_PATH = OUTPUT_DIR / "stall_cap_sweep.png"


def as_float(value):
    if value in ("", None):
        return None
    return float(value)


def read_summary(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as csv_file:
        return {row["quantity"]: row["value"] for row in csv.DictReader(csv_file)}


def read_sweep(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            wing_area = as_float(row.get("wing_area_m2"))
            stall = as_float(row.get("wing_stall_EAS_m_s"))
            cap = as_float(row.get("max_stall_EAS_m_s"))
            if wing_area is None or stall is None or cap is None:
                continue
            rows.append({
                "wing_area_m2": wing_area,
                "stall_m_s": stall,
                "cap_m_s": cap,
                "margin_m_s": as_float(row.get("stall_margin_m_s")),
                "ratio": cap / stall if stall else None,
                "feasible": row.get("feasible") == "True",
                "failure_reason": row.get("failure_reason", ""),
                "R": as_float(row.get("stall_limit_safety_factor_R")),
                "Lz_m": as_float(row.get("stall_limit_vertical_arm_m")),
            })
    rows.sort(key=lambda item: item["wing_area_m2"])
    return rows


def find_crossing(rows):
    for previous, current in zip(rows, rows[1:]):
        m0 = previous["margin_m_s"]
        m1 = current["margin_m_s"]
        if m0 is None or m1 is None:
            continue
        if m0 == 0.0:
            return previous["wing_area_m2"]
        if m0 * m1 < 0.0:
            s0 = previous["wing_area_m2"]
            s1 = current["wing_area_m2"]
            return s0 + (0.0 - m0) * (s1 - s0) / (m1 - m0)
    return None


def plot(rows, summary):
    if not rows:
        raise RuntimeError(f"No numeric stall-cap rows found in {SWEEP_CSV}")

    wing_area = [row["wing_area_m2"] for row in rows]
    stall = [row["stall_m_s"] for row in rows]
    cap = [row["cap_m_s"] for row in rows]
    margin = [row["margin_m_s"] for row in rows]
    ratio = [row["ratio"] for row in rows]
    stall_failed = [
        row for row in rows
        if row["margin_m_s"] is not None and row["margin_m_s"] < 0.0
    ]

    selected_area = as_float(summary.get("wing_area_m2"))
    selected_stall = as_float(summary.get("wing_stall_EAS_m_s"))
    selected_cap = as_float(summary.get("max_stall_EAS_m_s"))
    crossing = find_crossing(rows)

    R = next((row["R"] for row in rows if row["R"] is not None), None)
    Lz = next((row["Lz_m"] for row in rows if row["Lz_m"] is not None), None)
    subtitle_parts = []
    if R is not None:
        subtitle_parts.append(f"R={R:g}")
    if Lz is not None:
        subtitle_parts.append(f"Lz={Lz:g} m")
    subtitle = " | ".join(subtitle_parts)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    fig.suptitle("Pitch-moment stall cap across wing-area sweep", fontsize=15, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10, color="#46525e")

    ax = axes[0]
    ax.plot(wing_area, stall, color="#005f73", linewidth=2.4, label="actual stall EAS")
    ax.plot(wing_area, cap, color="#ca6702", linewidth=2.4, label="stall-speed cap")
    ax.fill_between(wing_area, stall, cap, where=[c >= s for c, s in zip(cap, stall)],
                    color="#2e8b57", alpha=0.12, interpolate=True, label="cap passes")
    ax.fill_between(wing_area, stall, cap, where=[c < s for c, s in zip(cap, stall)],
                    color="#9b2226", alpha=0.12, interpolate=True, label="cap fails")
    if stall_failed:
        ax.scatter(
            [row["wing_area_m2"] for row in stall_failed],
            [row["cap_m_s"] for row in stall_failed],
            color="#9b2226",
            marker="x",
            s=38,
            linewidths=1.5,
            label="stall-cap rejection",
        )
    if selected_area is not None and selected_stall is not None and selected_cap is not None:
        ax.axvline(selected_area, color="#33415c", linestyle="--", linewidth=1.4, label="selected")
        ax.scatter([selected_area], [selected_stall], color="#005f73", s=55, zorder=5)
        ax.scatter([selected_area], [selected_cap], color="#ca6702", s=55, zorder=5)
    if crossing is not None:
        ax.axvline(crossing, color="#6c757d", linestyle=":", linewidth=1.5, label=f"cap crossing {crossing:.2f} m^2")
    ax.set_ylabel("EAS [m/s]")
    ax.set_title("Stall speed and active cap")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot(wing_area, margin, color="#0a9396", linewidth=2.2, label="cap - stall")
    ax.axhline(0.0, color="#1f2933", linewidth=1.2)
    if selected_area is not None:
        ax.axvline(selected_area, color="#33415c", linestyle="--", linewidth=1.4)
    if crossing is not None:
        ax.axvline(crossing, color="#6c757d", linestyle=":", linewidth=1.5)
    ax.set_xlabel("wing area [m^2]")
    ax.set_ylabel("stall margin [m/s]")
    ax.set_title("Positive margin means the cap is inactive")
    ax.grid(True, alpha=0.25)

    ax_ratio = ax.twinx()
    ax_ratio.plot(wing_area, ratio, color="#7b2cbf", linewidth=1.8, linestyle="--", label="cap / stall")
    ax_ratio.axhline(1.0, color="#7b2cbf", linewidth=1.0, linestyle=":", alpha=0.8)
    ax_ratio.set_ylabel("cap / stall [-]", color="#7b2cbf")
    ax_ratio.tick_params(axis="y", colors="#7b2cbf")

    handles, labels = ax.get_legend_handles_labels()
    ratio_handles, ratio_labels = ax_ratio.get_legend_handles_labels()
    ax.legend(handles + ratio_handles, labels + ratio_labels, frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=180)
    fig.savefig(PLOT_PATH.with_suffix(".pdf"))
    plt.close(fig)


def main():
    rows = read_sweep(SWEEP_CSV)
    summary = read_summary(SUMMARY_CSV)
    plot(rows, summary)
    print(PLOT_PATH)


if __name__ == "__main__":
    main()
