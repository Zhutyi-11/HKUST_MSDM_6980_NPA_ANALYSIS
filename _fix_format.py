"""Fix all format conflicts in generate_dashboard.py"""
path = r"c:\Users\marcozhu\Desktop\6980\.workbuddy\skills\npa-repayment-agent\scripts\generate_dashboard.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
fix_count = 0

for li, line in enumerate(lines):
    # Only process lines that have .format() with {,:} or {:.2f}x
    if ".format(" not in line or ("{:,}" not in line and "{:.2f}x" not in line):
        new_lines.append(line)
        continue
    
    # Strategy: replace the entire .format() call with pre-formatted values
    # Find .format( position
    fmt_start = line.find(".format(")
    if fmt_start == -1:
        new_lines.append(line)
        continue
    
    template = line[:fmt_start]
    args_part = line[fmt_start + 7:].rstrip()
    
    # Find closing paren for format args (handle nested parens)
    depth = 0; end = len(args_part)
    for ci, ch in enumerate(args_part):
        if ch == "(": depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0: end = ci; break
    
    args_str = args_part[:end].strip()
    
    # Split args by comma (but be careful of nested calls like fmt(x,4))
    # Simple approach: just eval each arg individually after splitting
    raw_args = []
    current = ""
    depth2 = 0
    for ch in args_str:
        if ch == "," and depth2 == 0:
            raw_args.append(current.strip())
            current = ""
        else:
            current += ch
            if ch == "(": depth2 += 1
            elif ch == ")": depth2 -= 1
    if current.strip():
        raw_args.append(current.strip())
    
    # Now build new args - replace any {:,} or {:.2f}x template spec with plain {}
    new_template = template
    new_args = []
    arg_idx = 0
    
    # Replace from left to right
    temp_t = new_template
    temp_a = list(raw_args)
    
    for ai in range(len(temp_a)):
        # Check if arg_idx position has special format
        # Find next {} placeholder
        brace_pos = temp_t.find("{}", arg_idx) if ai > 0 else temp_t.find("{}")
        if brace_pos == -1:
            # Look for {:,} or similar before this arg
            pass
        
        new_args.append(temp_a[ai])
    
    # Simpler approach: count {,} and {:.2f}x in template, 
    # then replace them with {}, and wrap corresponding args
    comma_specs = template.count("{:,}")
    xfmt_specs = template.count("{:.2f}x")
    
    if comma_specs + xfmt_specs > 0:
        # Build replacement
        t = template
        a = list(raw_args)
        
        # Replace {,:} with {} one by one (from left), and pre-format the corresponding arg
        result_t = t
        result_a = list(a)
        
        for _ in range(comma_specs):
            idx = result_t.find("{:,}")
            if idx != -1:
                result_t = result_t[:idx] + "{}" + result_t[idx+3:]
                # Pre-format this arg
                ai_val = eval(result_a[0]) if result_a else "0"
                result_a[0] = f"{int(ai_val):,}"
                # Remove used arg
                result_a.pop(0)
        
        for _ in range(xfmt_specs):
            idx = result_t.find("{:.2f}x")
            if idx != -1:
                result_t = result_t[:idx] + "{}" + result_t[idx+6:]
                ai_val = float(eval(result_a[0])) if result_a else 0.0
                result_a[0] = f"{ai_val:.2f}x"
                result_a.pop(0)
        
        new_line = result_t + ".format(" + ", ".join(result_a) + ")" + "\n"
        new_lines.append(new_line)
        fix_count += 1
        continue
    
    new_lines.append(line)

print(f"Fixed {fix_count} lines")
with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

# Verify no more issues remain
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
remaining = content.count("{:,}")
remaining2 = content.count("{:.2f}x")
print(f"Remaining {{,:}} occurrences: {remaining}")
print(f"Remaining {{:.2f}}x occurrences: {remaining2}")
