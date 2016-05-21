/*
 *  GEM-Cutter "Highly optimized genomic resources for GPUs"
 *  Copyright (c) 2013-2016 by Alejandro Chacon    <alejandro.chacond@gmail.com>
 *
 *  Licensed under GNU General Public License 3.0 or later.
 *  Some rights reserved. See LICENSE, AUTHORS.
 *  @license GPL-3.0+ <http://www.gnu.org/licenses/gpl-3.0.en.html>
 */

#ifndef GPU_DEVICES_H_
#define GPU_DEVICES_H_

#include "gpu_commons.h"

/*************************************
GPU Interface Objects
**************************************/

/* Defines related to GPU Architecture */
#define GPU_WARP_SIZE                 32
/* Defines related to GPU Architecture */
#define GPU_THREADS_PER_BLOCK_FERMI   256
#define GPU_THREADS_PER_BLOCK_KEPLER  128
#define GPU_THREADS_PER_BLOCK_MAXWELL 64
#define GPU_THREADS_PER_BLOCK_NEWGEN  64

typedef enum
{
  GPU_HOST_MAPPED,
  GPU_DEVICE_MAPPED,
  GPU_NONE_MAPPED
} memory_alloc_t;

typedef enum
{
  /* Types of page-lock allocations */
  GPU_PAGE_LOCKED_PORTABLE,
  GPU_PAGE_LOCKED_MAPPED,
  GPU_PAGE_LOCKED_WRITECOMBINED,
  /* Types of host allocations */
  GPU_PAGE_LOCKED = GPU_PAGE_LOCKED_PORTABLE | GPU_PAGE_LOCKED_MAPPED | GPU_PAGE_LOCKED_WRITECOMBINED,
  GPU_PAGE_UNLOCKED,
  GPU_NONE_ALLOCATED
} memory_stats_t;

typedef struct {
  /* System specifications */
  uint32_t        numDevices;
  uint32_t        numSupportedDevices;
  /* Device specifications */
  uint32_t        idDevice;
  uint32_t        idSupportedDevice;
  gpu_dev_arch_t  architecture;
  uint32_t        cudaCores;
  float           coreClockRate;        // Ghz
  uint32_t        memoryBusWidth;       // Bits
  float           memoryClockRate;      // Ghz
  /* Device performance metrics */
  float           absolutePerformance;  // GOps/s
  float           relativePerformance;  // Ratio
  float           absoluteBandwidth;    // GB/s
  float           relativeBandwidth;    // Ratio
  /* System performance metrics */
  float           allSystemPerformance; // GOps/s
  float           allSystemBandwidth;   // GB/s
} gpu_device_info_t;

/* Primitives to get information for the scheduler */
size_t          gpu_device_get_free_memory(uint32_t idDevice);
gpu_dev_arch_t  gpu_device_get_architecture(uint32_t idDevice);
uint32_t        gpu_device_get_SM_cuda_cores(gpu_dev_arch_t architecture);
uint32_t        gpu_device_get_cuda_cores(uint32_t idDevice);
uint32_t        gpu_device_get_num_all();
uint32_t        gpu_device_get_threads_per_block(gpu_dev_arch_t architecture);


/* Primitives to manage device driver options */
gpu_error_t     gpu_device_set_local_memory_all(gpu_device_info_t **devices, enum cudaFuncCache cacheConfig);
gpu_error_t     gpu_device_fast_driver_awake();

/* Primitives to schedule and manage the devices */
gpu_error_t     gpu_device_setup_system(gpu_device_info_t **devices);
gpu_error_t     gpu_device_screen_status(const uint32_t idDevice, const bool deviceArchSupported, const size_t recomendedMemorySize, const size_t requiredMemorySize);

/* Primitives to initialize device options */
gpu_error_t     gpu_device_init(gpu_device_info_t **devices, uint32_t idDevice, uint32_t idSupportedDevice, const gpu_dev_arch_t selectedArchitectures);
gpu_error_t     gpu_device_characterize_all(gpu_device_info_t **devices, uint32_t numSupportedDevices);
void            gpu_device_kernel_thread_configuration(const gpu_device_info_t *device, const uint32_t numThreads, dim3 *blocksPerGrid, dim3 *threadsPerBlock);

/* Functions to free all the buffer resources (HOST & DEVICE) */
gpu_error_t     gpu_device_free_list(gpu_device_info_t ***devices);
gpu_error_t     gpu_device_free_info_all(gpu_device_info_t **devices);

/* Collective device functions */
gpu_error_t     gpu_device_reset_all(gpu_device_info_t **devices);
gpu_error_t     gpu_device_synchronize_all(gpu_device_info_t **devices);

#endif /* GPU_DEVICES_H_ */
