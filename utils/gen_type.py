import re
import sys

filename = sys.argv[1]
with open(filename, "r", encoding="utf-8") as f:
    cpp = f.read()

pattern = re.compile(r"struct\s+(\w+)\s*{([^}]*)};", re.MULTILINE)
for struct_name, body in pattern.findall(cpp):
    fields = []
    for line in body.split(";"):
        line = line.strip()
        if not line:
            continue
        t, name = line.split()
        fields.append((t, name))

    print(f"class {struct_name}:")
    args = ", ".join(n for _, n in fields)
    if args == "":
      print(f"    def __init__(self):")
      print(f"        pass")
    else:
      print(f"    def __init__(self, {args}):")
      for _, n in fields:
          print(f"        self.{n} = {n}")
