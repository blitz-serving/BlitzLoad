**Setup an environment for BlitzLoad**

```bash
docker run -it \
    --privileged \
    --gpus all \
    --ipc host \
    --network host \
    --workdir /root \
    --name lib-blitz \
    nvcr.io/nvidia/pytorch:24.06-py3 /bin/bash
```

```bash
echo 'export CMAKE_PREFIX_PATH=/root/.local' >> /root/.bashrc
echo 'export PATH=/root/.local/bin:$PATH' >> /root/.bashrc
source /root/.bashrc

apt update && \
apt install git vim wget curl autoconf pkg-config libssl-dev libzmq3-dev -y && \
apt reinstall libibverbs-dev
```