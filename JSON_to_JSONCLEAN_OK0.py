# JSON_to_JSONCLEAN.py
import json
import sys

def fix_num_string(s: str) -> str:
    # trim spazi
    s = s.strip()
    # "" -> "0"
    if s == "":
        return "0"
    # normalizza "-0" -> "0"
    if s == "-0":
        return "0"
    return s

def clean_value(v):
    """Pulisce ricorsivamente il JSON:
       - stringhe vuote "" -> "0"
       - rimuove graffe esterne { ... } se presenti
       - normalizza "-0" -> "0"
    """
    if isinstance(v, str):
        # rimuovi eventuali graffe esterne usate come 'string wrapper'
        if v.startswith("{") and v.endswith("}"):
            v = v[1:-1]
        return fix_num_string(v)
    elif isinstance(v, list):
        return [clean_value(x) for x in v]
    elif isinstance(v, dict):
        return {k: clean_value(val) for k, val in v.items()}
    else:
        return v

def clean_json(infile, outfile):
    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = clean_value(data)

    # Safety: verifica che non rimangano stringhe vuote
    def any_empty(x):
        if isinstance(x, str):
            return x == ""
        if isinstance(x, list):
            return any(any_empty(t) for t in x)
        if isinstance(x, dict):
            return any(any_empty(t) for t in x.values())
        return False
    if any_empty(data):
        raise ValueError("Sono rimaste stringhe vuote \"\" nel JSON pulito.")

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python JSON_to_JSONCLEAN.py input.json output.json")
        sys.exit(1)
    infile, outfile = sys.argv[1], sys.argv[2]
    clean_json(infile, outfile)
    print("[OK] File pulito scritto in:", outfile)
