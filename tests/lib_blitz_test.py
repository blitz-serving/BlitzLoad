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


# import torch
# import time

# x = torch.randn(1024 * 1024 * 256, device="cpu")  # 1GB
# torch.cuda.synchronize()
# start = time.time()
# y = x.to("cuda")
# torch.cuda.synchronize()
# end = time.time()
# bw = 1.0 / (end - start)  # GB/s
# print("Host→Device bandwidth:", bw)

# import grpc
# from grpc_health.v1 import health_pb2 ,health_pb2_grpc
# import os

# proxy_vars = [
#     "HTTP_PROXY",
#     "http_proxy",
#     "HTTPS_PROXY",
#     "https_proxy",
#     "ALL_PROXY",
#     "all_proxy",
#     "NO_PROXY",
#     "no_proxy",
# ]

# # 删除它们
# for var in proxy_vars:
#     if var in os.environ:
#         del os.environ[var]

# channel = grpc.insecure_channel("localhost:60060")
# stub = health_pb2_grpc.HealthStub(channel)
# try:
#     response = stub.Check(health_pb2.HealthCheckRequest(service=""))
#     if response.status == health_pb2.HealthCheckResponse.SERVING:
#         print("Channel is valid and server is serving")
#     else:
#         print("Channel connected but service not ready")
# except grpc.RpcError as e:
#     print(f"Channel not valid: {e}")
