"""Verify the quantitative claims in the research plan, independently."""
import numpy as np

print("=" * 68)
print("1. SPELLER GEOMETRY  (Chen et al. 2015 PNAS: 5x8 stimulation matrix)")
print("=" * 68)

# Stated acquisition parameters -- THESE MUST BE VERIFIED FROM THE PAPERS
diag_in = 23.6
res_w, res_h = 1920, 1080
view_cm = 70.0
matrix_w_px, matrix_h_px = 1510, 1037
stim_px = 140
n_cols, n_rows = 8, 5

diag_cm = diag_in * 2.54
aspect = np.hypot(16, 9)
screen_w_cm = diag_cm * 16 / aspect
screen_h_cm = diag_cm * 9 / aspect
cm_per_px = screen_w_cm / res_w

print(f"screen        : {screen_w_cm:.2f} x {screen_h_cm:.2f} cm")
print(f"pixel pitch   : {cm_per_px:.5f} cm/px")
print(f"check vertical: {screen_h_cm / res_h:.5f} cm/px  (should match)")

# Column centres: evenly spaced across the matrix, inset by half a stimulus
def centres(n, span_px, stim):
    usable = span_px - stim          # centre-to-centre span of outermost
    return (np.arange(n) - (n - 1) / 2) * (usable / (n - 1))

col_px = centres(n_cols, matrix_w_px, stim_px)
row_px = centres(n_rows, matrix_h_px, stim_px)

col_deg = np.degrees(np.arctan(col_px * cm_per_px / view_cm))
row_deg = np.degrees(np.arctan(row_px * cm_per_px / view_cm))

print(f"\ncolumn eccentricities (deg): {np.round(col_deg, 2)}")
print(f"row    eccentricities (deg): {np.round(row_deg, 2)}")
print(f"full span: {np.ptp(col_deg):.1f} deg H x {np.ptp(row_deg):.1f} deg V")
print(f"mean adjacent step: {np.diff(col_deg).mean():.2f} deg H, "
      f"{np.diff(row_deg).mean():.2f} deg V")
print(f"\nplan claimed +/-14.91 H, +/-9.89 V")
print(f"computed       +/-{abs(col_deg[0]):.2f} H, +/-{abs(row_deg[0]):.2f} V"
      f"   -> {'MATCH' if abs(abs(col_deg[0]) - 14.91) < 0.1 else 'MISMATCH'}")

print()
print("=" * 68)
print("2. HIGH-PASS DECAY  (single-pole 0.15 Hz)")
print("=" * 68)
fc = 0.15
tau = 1.0 / (2 * np.pi * fc)
print(f"tau = 1/(2*pi*{fc}) = {tau:.4f} s   (plan claimed 1.06 s)")
print()
print(f"{'t after saccade':>18} | {'residual':>9} | {'plan':>7}")
print("-" * 42)
claimed = {0.14: 87.6, 0.50: 62.4, 0.64: 54.7, 1.14: 34.1, 2.00: 15.2, 5.00: 0.9}
for t, c in claimed.items():
    r = 100 * np.exp(-t / tau)
    flag = "ok" if abs(r - c) < 0.3 else "DIFFERS"
    print(f"{t:>15.2f} s | {r:>8.1f}% | {c:>6.1f}%  {flag}")

print()
print("=" * 68)
print("3. AMPLITUDE SCALE")
print("=" * 68)
for uv_deg, label in [(15.0, "periorbital EOG (literature)"),
                      (7.5, "50% attenuation at Fp"),
                      (3.0, "20% attenuation at Fp")]:
    ext = uv_deg * abs(col_deg[0])
    step = uv_deg * np.diff(col_deg).mean()
    print(f"{label:<32} {uv_deg:>4.1f} uV/deg -> "
          f"extreme {ext:>6.1f} uV, adjacent step {step:>5.1f} uV")
print(f"\noccipital SSVEP fundamental: ~1-5 uV")
print("Even at the most conservative attenuation the ocular step at the")
print("outermost columns exceeds SSVEP amplitude. The ADJACENT-column step")
print("is what actually limits 40-class separability, and it is the number")
print("to measure in real data.")
