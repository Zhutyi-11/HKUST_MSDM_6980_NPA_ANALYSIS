"""Fix format conflicts in generate_dashboard.py"""
import re

path = r"c:\Users\marcozhu\Desktop\6980\.workbuddy\skills\npa-repayment-agent\scripts\generate_dashboard.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

def fix_line(line):
    if ".format(" not in line:
        return line
    if "{:,}" not in line and "{:.2f}x" not in line:
        return line

    # Pattern: prefix + 'template'.format(args) or "template".format(args)
    # Use a more flexible regex
    pat = r"^(.*?)'(.*)'\.format\((.*)\)\s*$"
    m = re.match(pat, line, re.DOTALL)
    if not m:
        m = re.match(r'^(.*)"(.*)"\.format\((.*)\)\s*$', line, re.DOTALL)
    if not m:
        return line

    prefix, template, args_str = m.group(1), m.group(2), m.group(3).strip()
    
    n_comma = template.count("{:,}")
    n_xfmt = template.count("{:.2f}x")
    if n_comma + n_xfmt == 0:
        return line

    # Split args at top-level commas
    args_list = []
    cur = ""; depth = 0
    for ch in args_str:
        if ch == "," and depth == 0:
            args_list.append(cur.strip()); cur = ""
        else:
            cur += ch
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
    if cur.strip():
        args_list.append(cur.strip())

    new_tpl = template
    new_args = []
    for ai in range(len(args_list)):
        if "{:,}" in new_tpl:
            new_tpl = new_tpl.replace("{:,}", "{}", 1)
            new_args.append("format({}, ',')".format(args_list[ai]))
        elif "{:.2f}x" in new_tpl:
            new_tpl = new_tpl.replace("{:.2f}x", "{}", 1)
            new_args.append("'" + "{:.2f}x'" + "'.format(" + args_list[ai] + ")")
        else:
            new_args.append(args_list[ai])

    return prefix + "'" + new_tpl + "'.format(" + ", ".join(new_args) + ")\n"

new_lines = [fix_line(l) for l in content.split("\n")]
fixed = sum(1 for old, new in zip(content.split("\n"), new_lines) if old != new)

with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))

with open(path, "r", encoding="utf-8") as f:
    c2 = f.read()
print(f"Fixed {fixed} lines. Remaining {{,:}}: {c2.count('{:,}')}. Remaining {{:.2f}}x: {c2.count('{:.2f}x')}")
