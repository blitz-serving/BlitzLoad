import blitz_lib
import torch
import cupy

if __name__ == "__main__":
    print(f"Haha pull model", flush=True)
    task_id = blitz_lib.pull_model("/nvme/ly/tmp_files")
    done = False
    print("Checking")
    while not done:
        done = blitz_lib.check_model(task_id)
    print("Haha pull model return", flush=True)
    import time
    time.sleep(10)
    st_file = "/nvme/ly/tmp_files/dangertensors.0.meta"
    cnt = 0
    with open(st_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            # if cnt > 10:
            #     break
            line = line.strip()
            name, length = line.rsplit(" ", 1)
            print(f"Loading {name}", flush=True)
            param = torch.zeros((int(int(length) / 2)), device="cuda:0", dtype=torch.bfloat16)
            blitz_lib.load_weight_from_ipc_handle(param, name)

            # print(param[:15].float().cpu().numpy())
            cnt += 1
