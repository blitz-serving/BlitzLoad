#include <buffer.hpp>
#include <condition_variable>
#include <cstddef>
#include <danger_tensor.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace blitz {

class BlitzEngine {
public:
  BlitzEngine(std::vector<int> buf_devices, size_t buf_size);
  ~BlitzEngine();

  int ssd_to_mem(std::vector<std::string> files);
  void mem_to_tensor(cudaIpcMemHandle_t &handle, std::string tensor_name,
                     size_t tensor_size, int tensor_device);

  void mem_to_buffer(int shard_num);
  void buffer_to_tensor(cudaIpcMemHandle_t &handle, int tensor_device,
                        size_t tensor_size, int shard_id); // deprecated
  size_t export_handler(cudaIpcMemHandle_t *handle, size_t tensor_size,
                        int shard_id);
  void free_handler(size_t tensor_size, int shard_id);

private:
  std::vector<std::thread> threads;
  std::vector<int> buf_devices;
  std::vector<std::unique_ptr<std::mutex>> mtxs;
  std::vector<int> is_empty;
  std::vector<std::unique_ptr<std::condition_variable>> cvs;
  std::vector<std::unique_ptr<buffer::BufferGroup>> buf_groups;
  cudaStream_t buf2tensor_stream;

  /// enable m*(m-1)/2 access
  void enable_p2p_access(std::vector<int> devices);
  // in non-rdma case, ssd -> mem
  std::map<int, std::unique_ptr<dangertensor::DangerTensor>> dangertensor_map;
};
} // namespace blitz
