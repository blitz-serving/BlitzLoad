import blitz_lib
import torch
import cupy

if __name__ == "__main__":
    tensor = torch.zeros((1024), device="cuda:0", dtype=torch.bfloat16)
    new_tensor = tensor[512:0]
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    tensor_ptr = tensor.data_ptr()
    tensor_ptr_new = new_tensor.data_ptr()
    print(
        f"tensor_ptr={tensor_ptr}, tensor_ptr2 = {tensor_ptr_new}"
    )
    # blitz_lib.pull_model("/nvme/ly/tmp_files")
    # st_file = "/nvme/ly/tmp_files/dangertensors.0.meta"
    # cnt = 0
    # with open(st_file, "r", encoding="utf-8") as f:
    #     lines = f.readlines()[1:]
    #     for line in lines:
    #         # if cnt > 10:
    #         #     break
    #         line = line.strip()
    #         name, length = line.rsplit(" ", 1)
    #         print(f"Loading {name}")
    #         param = torch.zeros((int(int(length) / 2)), device="cuda:0", dtype=torch.bfloat16)
    #         blitz_lib.load_weight_from_ipc_handle(param, name)
            
    #         print(f"First value: {param[0].item()}")
    #         cnt += 1
