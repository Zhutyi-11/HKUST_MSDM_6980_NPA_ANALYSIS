path = r"c:\Users\marcozhu\Desktop\6980\.workbuddy\skills\npa-repayment-agent\scripts\generate_dashboard.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Fix line 414: queue table row - replace {:,} and {:.1f}x etc with pre-formatted versions
lines[413] = """parts.append('<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{};margin-right:4px;"></span>{}</td><td><strong>' + str(int(q.get("accounts",0))) + '</strong></td><td>{:.1f}%</td><td>{:.4f}</td><td>{:.2f}%</td><td>' + str(int(q.get("balance_proxy_total",0))) + '</td><td>{:,.0f}</td><td><strong>{:,.0f}</strong></td><td>{:,.0f}</td><td style="color:{};font-weight:700;">{:.1f}x</td></tr>'.format(
    clr, act[:20] if act else "",
    int(q.get("accounts",0))/max(total_acc_q,1)*100,
    float(q.get("avg_calibrated_prob",0)), apr,
    int(q.get("balance_proxy_total",0)),
    float(q.get("expected_gross_recovery_total",0)),
    float(q.get("expected_net_recovery_total",0)),
    float(q.get("contact_cost_total",0)),
    roi_v))
"""

# Fix line 447: account table row - same approach
# Read the actual line first
print("Line 447 before:", repr(lines[446][:200]))

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("Saved")
