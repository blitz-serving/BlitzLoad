import zmq
import json
import hashlib
import torch
import blitz_lib
from mq_types import *


def gen_sha256(model_name):
    return hashlib.sha256(model_name.encode("utf-8")).hexdigest()


def send_recv(socket, object):
    req_json = json.dumps(object.__dict__)
    socket.send_string(req_json)
    reply = socket.recv_string()
    resp_dict = json.loads(reply)
    return resp_dict


def main():
    blitz_lib.register_rank(0)
    model_path = "/nvme/ly/tmp_files"
    blitz_lib.pull_model(model_path, 1)
    # context = zmq.Context()
    # socket = context.socket(zmq.REQ)
    # socket.connect("tcp://localhost:55555")
    # socket2 = context.socket(zmq.REQ)
    # socket2.connect("tcp://localhost:55556")
    # socket3 = context.socket(zmq.REQ)
    # socket3.connect("tcp://localhost:55557")
    # socket3.connect("tcp://localhost:55558")

    # req = PullModelRequest(model_path, 1)
    # resp_dict = send_recv(socket, req)
    # resp = PullModelResponse(resp_dict["task_id"])
    # task_id = resp.task_id

    # print("Received task_id:", resp.task_id)

    import time

    time.sleep(5)
    done = False
    while not done:
        time.sleep(0.5)
        done = blitz_lib.check_model(model_path)

    st_file = f"{model_path}/dangertensors.0.meta"
    with open(st_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            name, length = line.rsplit(" ", 1)
            print(f"Loading {name}", flush=True)
            param = torch.zeros(
                (int(int(length) / 2)), device="cuda:0", dtype=torch.bfloat16
            )
            blitz_lib.load_tensor(param, name)

            print(param[-15:].float().cpu().numpy())
    blitz_lib.reset_status()

if __name__ == "__main__":
    main()
