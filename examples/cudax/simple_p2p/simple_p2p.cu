/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

// includes, system
#include <cuda/memory_resource>

#include <cuda/experimental/buffer.cuh>
#include <cuda/experimental/device.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <stdio.h>
#include <stdlib.h>

namespace cudax = cuda::experimental;

__global__ void SimpleKernel(float* src, float* dst)
{
  // Just a dummy kernel, doing enough for us to verify that everything
  // worked
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx]      = src[idx] * 2.0f;
}

int main(int argc, char** argv)
{
  printf("[%s] - Starting...\n", argv[0]);

  // Number of GPUs
  printf("Checking for multiple GPUs...\n");
  printf("CUDA-capable device count: %lu\n", cudax::devices.size());

  if (cudax::devices.size() < 2)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for "
           "%s.\n",
           argv[0]);
    printf("Waiving test.\n");
    exit(2);
  }

  // Check possibility for peer access
  printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  std::vector<cudax::device_ref> peers;
  for (auto& dev_i : cudax::devices)
  {
    for (auto& dev_j : cudax::devices)
    {
      if (dev_i != dev_j)
      {
        bool can_access_peer = dev_i.is_peer_accessible_from(dev_j);
        // Save all peers of a first device found with a peer
        if (can_access_peer && peers.size() == 0)
        {
          peers = dev_i.get_peers();
          peers.push_back(dev_i);
        }
        printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
               "", // dev_i.get_name(),
               dev_i.get(),
               "", // dev_j.get_name(),
               dev_j.get(),
               can_access_peer ? "Yes" : "No");
      }
    }
  }

  if (peers.size() == 0)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for "
           "%s.\n",
           argv[0]);
    printf("Peer to Peer access is not available amongst GPUs in the system, "
           "waiving test.\n");

    exit(2);
  }

  printf("Found %lu peers, using devices %d and %d\n", peers.size(), peers[0].get(), peers[1].get());

  cudax::stream dev0_stream(peers[0]);
  cudax::stream dev1_stream(peers[1]);

  // Allocate buffers
  constexpr size_t buf_cnt = 1024 * 1024 * 16;
  printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n",
         int(buf_cnt / 1024 / 1024 * sizeof(float)),
         peers[0].get(),
         peers[1].get());

  cudax::mr::async_memory_resource dev0_resource(peers[0]);
  dev0_resource.enable_peer_access(peers[1]);
  cudax::mr::async_memory_resource dev1_resource(peers[1]);
  dev1_resource.enable_peer_access(peers[0]);

  cudax::uninitialized_buffer<float, cuda::mr::device_accessible> dev0_buffer(dev0_resource, buf_cnt);
  cudax::uninitialized_buffer<float, cuda::mr::device_accessible> dev1_buffer(dev1_resource, buf_cnt);
  cudax::uninitialized_buffer<float, cuda::mr::host_accessible> host_buffer(cuda::mr::pinned_memory_resource(), buf_cnt);

#define checkCudaErrors
  // Create CUDA event handles
  printf("Creating event handles...\n");

  auto start_event = dev0_stream.record_timed_event();
  for (int i = 0; i < 100; i++)
  {
    // With UVA we don't need to specify source and target devices, the
    // runtime figures this out by itself from the pointers
    // Ping-pong copy between GPUs
    if (i % 2 == 0)
    {
      checkCudaErrors(cudaMemcpyAsync(
        dev1_buffer.data(), dev0_buffer.data(), dev0_buffer.size_bytes(), cudaMemcpyDefault, dev0_stream.get()));
    }
    else
    {
      checkCudaErrors(cudaMemcpyAsync(
        dev0_buffer.data(), dev1_buffer.data(), dev0_buffer.size_bytes(), cudaMemcpyDefault, dev0_stream.get()));
    }
  }

  auto end_event = dev0_stream.record_timed_event();
  dev0_stream.wait();
  cuda::std::chrono::duration<double> duration(end_event - start_event);
  printf("cudaMemcpyPeer / cudaMemcpy between GPU%d and GPU%d: %.2fGB/s\n",
         peers[0].get(),
         peers[1].get(),
         (dev0_buffer.size_bytes() / (1024 * 1024 * 1024) / duration.count()));

  // Prepare host buffer and copy to GPU 0
  printf("Preparing host buffer and memcpy to GPU%d...\n", peers[0].get());

  for (int i = 0; i < dev0_buffer.size_bytes() / sizeof(float); i++)
  {
    host_buffer.data()[i] = float(i % 4096);
  }

  checkCudaErrors(cudaMemcpyAsync(
    dev0_buffer.data(), host_buffer.data(), dev0_buffer.size_bytes(), cudaMemcpyDefault, dev0_stream.get()));

  // Kernel launch configuration
  const dim3 threads(512, 1);
  const dim3 blocks((dev0_buffer.size_bytes() / sizeof(float)) / threads.x, 1);

  // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
  // output to the GPU 1 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         peers[1].get(),
         peers[0].get(),
         peers[1].get());
  auto dims = cudax::distribute<512>(dev0_buffer.size());
  cudax::launch(dev1_stream, dims, SimpleKernel, dev0_buffer.data(), dev1_buffer.data());

  dev1_stream.wait();

  // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
  // output to the GPU 0 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         peers[0].get(),
         peers[1].get(),
         peers[0].get());
  cudax::launch(dev0_stream, dims, SimpleKernel, dev1_buffer.data(), dev0_buffer.data());

  dev1_stream.wait();

  // Copy data back to host and verify
  printf("Copy data back to host from GPU%d and verify results...\n", peers[0].get());
  checkCudaErrors(cudaMemcpyAsync(
    host_buffer.data(), dev0_buffer.data(), dev0_buffer.size_bytes(), cudaMemcpyDefault, dev0_stream.get()));

  int error_count = 0;

  for (int i = 0; i < dev0_buffer.size_bytes() / sizeof(float); i++)
  {
    // Re-generate input data and apply 2x '* 2.0f' computation of both
    // kernel runs
    if (host_buffer.data()[i] != float(i % 4096) * 2.0f * 2.0f)
    {
      printf("Verification error @ element %i: val = %f, ref = %f\n",
             i,
             host_buffer.data()[i],
             (float(i % 4096) * 2.0f * 2.0f));

      if (error_count++ > 10)
      {
        break;
      }
    }
  }

  // Disable peer access (also unregisters memory for non-UVA cases)
  printf("Disabling peer access...\n");
  dev0_resource.disable_peer_access(peers[1]);
  dev1_resource.disable_peer_access(peers[0]);

  // No cleanup needed
}
