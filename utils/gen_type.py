import re

cpp = """
struct PullModelRequest {
  string model_name;
  int32_t world_size;
};

struct PullModelResponse {
  string task_id;
};

struct CheckModelRequest {
  string model_name;
  string task_id;
};

struct CheckModelResponse {
  bool done;
};

struct LoadTensorRequest {
  string tensor_name;
  uint64_t tensor_size;
  int32_t rank;
};

struct LoadTensorResponse {
  cudaIpcMemHandle_t handler;
  uint64_t offset;
  uint64_t loaded_size;
};

struct RevertHandlerRequest {
  string tensor_name;
  uint64_t tensor_size;
  int32_t rank;
};

struct RevertHandlerResponse {
  bool success;
};
"""

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
    print(f"    def __init__(self, {args}):")
    for _, n in fields:
        print(f"        self.{n} = {n}")
