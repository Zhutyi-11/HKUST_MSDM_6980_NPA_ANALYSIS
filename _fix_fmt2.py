path = r"c:\Users\marcozhu\Desktop\6980\.workbuddy\skills\npa-repayment-agent\scripts\generate_dashboard.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Fix line 278 (roi_str) - the nested format chain
lines[277] = '    roi_str = "{: .2f}x".format(float(mr.get("expected_roi", 0)))\n'

# Check remaining issues
content = "".join(lines)
rem_comma = content.count("{:,}")
rem_xfmt = content.count("{:.2f}x")
print(f"After fix - {{,:}}: {rem_comma}, {{:.2f}}x: {rem_xfmt}")

# If still issues, show them
if rem_comma > 0 or rem_xfmt > 0:
    for i, l in enumerate(lines):
        if "{:,}" in l:
            print(f"  Still has comma fmt at line {i+1}: {l.strip()[:150]}")
        if "{:.2f}x" in l and ".format(" in l and "'{:.2f}x'" not in l and '"{:.2f}x"' not in l:
            print(f"  Still has xfmt at line {i+1}: {l.strip()[:150]}")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Done")
