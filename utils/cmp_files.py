import os


def compare_files(file1, file2, chunk_size=4096):
    """逐字节比较两个文件，返回是否相同"""
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            b1 = f1.read(chunk_size)
            b2 = f2.read(chunk_size)
            if b1 != b2:
                return False
            if not b1:  # 到达文件末尾
                return True


def count_bit_differences(file1, file2, chunk_size=114688):
    """统计两个文件不同的 bit 数"""
    total_diff = 0
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            b1 = f1.read(chunk_size)
            b2 = f2.read(chunk_size)

            if not b1 and not b2:  # 都读到末尾
                break

            # 如果长度不同，用0填充短的
            if len(b1) < len(b2):
                b1 += b"\x00" * (len(b2) - len(b1))
            elif len(b2) < len(b1):
                b2 += b"\x00" * (len(b1) - len(b2))

            for byte1, byte2 in zip(b1, b2):
                xor = byte1 ^ byte2
                # bin(xor).count("1") 统计不同的 bit
                total_diff += bin(xor).count("1")

    return total_diff


def compare_directories(dir1, dir2):
    results = {}
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    common_files = files1 & files2  # 取交集
    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1

    for fname in common_files:
        f1 = os.path.join(dir1, fname)
        f2 = os.path.join(dir2, fname)
        if os.path.isfile(f1) and os.path.isfile(f2):
            same = compare_files(f1, f2)
            results[fname] = "相同" if same else "不同"

    return results, only_in_dir1, only_in_dir2

num = count_bit_differences(
    "/tmp/blitz/model.embed_tokens.weight.bin",
    "/tmp/vllm/model.embed_tokens.weight.bin",
)


if __name__ == "__main__":
    dir1 = "/tmp/blitz"
    dir2 = "/tmp/vllm"

    results, only1, only2 = compare_directories(dir1, dir2)

    for fname, status in results.items():
        print(f"{fname}: {status}")
    if only1:
        print("仅在目录1:", only1)
    if only2:
        print("仅在目录2:", only2)
