import json

with open("experiments/scale_5arch_15gb/reports/eval_summary.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print(f"{'Model':<14} | {'Method':<6} | {'PPL (0->100)':<16} | {'ARC Acc (0->100)':<18} | {'MMLU (0->100)':<16} | {'SQuAD F1':<10}")
print("-" * 95)
for k, v in sorted(d.items()):
    rec = v.get("eval_records", [])
    if len(rec) >= 3:
        r0, r100 = rec[0], rec[-1]
        m, met = v["model"], v["method"]
        ppl = f"{r0.get('ppl',0)} -> {r100.get('ppl',0)}"
        arc = f"{r0.get('arc_acc',0)} -> {r100.get('arc_acc',0)}"
        mmlu = f"{r0.get('mmlu_acc',0)} -> {r100.get('mmlu_acc',0)}"
        f1 = f"{r100.get('qa_f1',0)}"
        print(f"{m:<14} | {met:<6} | {ppl:<16} | {arc:<18} | {mmlu:<16} | {f1:<10}")
