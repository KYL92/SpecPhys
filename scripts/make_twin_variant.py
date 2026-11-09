"""트윈 요인 분해 config: python make_twin_variant.py NAME WAVE(0/1) SPEC(0/1) → configs/{train20h,train6hi,train6lo}_<name>.json, out *_<NAME>"""
import json,sys
NAME,W,S=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); PF="/SPECPHYS_ROOT/_workspace/data/physformer"
for c in ["train20h","train6hi","train6lo"]:
    d=json.load(open(f"{PF}/configs/{c}_t.json"))
    for sub in d["subjects"]:
        if not W: sub.pop("wave_bank",None); sub.pop("wave_bank_idx",None)
        if not S: sub.pop("spec_exclude",None)
    d["out"]=f"_workspace/data/physformer/{c.upper()}_{NAME}"; json.dump(d,open(f"{PF}/configs/{c}_{NAME.lower()}.json","w"),indent=1)
print("variant",NAME,"wave",W,"spec",S)
