#pragma once

#include <libumpalumpa/data/payload.hpp>

// save diagnostic state
#pragma GCC diagnostic push
// turn off warnings
#pragma GCC system_header
#include <starpu.h>
#include <vector>

/**
 * Create StarPU mask.
 * Each item of col, if item == true, the respective worker is enabled.
 * Stores number of enabled workers in count
 **/
template<typename T> uint32_t *CreateWorkerMask(unsigned &count, const T &col)
{
  // how many 32b numbers do we need to mask all workers
  const size_t len = (starpu_worker_get_count() / 32) + 1;
  // create and zero-out the mask
  auto *mask = reinterpret_cast<uint32_t *>(calloc(len, sizeof(uint32_t)));
  count = 0;
  for (size_t i = 0; i < col.size(); ++i) {
    if (col[i]) {
      // set the corresponding bit to 1
      mask[i / 32] |= (1 << (i % 32));
      ++count;
    }
  }
  return mask;
}

/**
 * Returns StarPU memory node describing current location of the data referenced by PD
 **/
unsigned GetMemoryNode(const umpalumpa::data::PhysicalDescriptor &pd);

/**
 * Return a vector with IDs of each Nth CPU worker.
 * For example, if 5 CPU workers exists and N=2, it returns 0, 2, 4
 **/
std::vector<unsigned> GetCPUWorkerIDs(unsigned n);

template<typename T>
static void payload_register_data_handle(starpu_data_handle_t handle,
  unsigned home_node,
  void *data_interface)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface);
  for (unsigned node = 0; node < STARPU_MAXNODES; node++) {
    auto *local_interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(
      starpu_data_get_interface_on_node(handle, node));
    *local_interface = *interface;
    if (node != home_node) {
      auto pd = umpalumpa::data::PhysicalDescriptor(nullptr,
        interface->GetRequiredBytes(),
        interface->dataInfo.GetType(),
        umpalumpa::data::ManagedBy::StarPU,
        node);
      local_interface->Set(pd);
    }
  }
}

template<typename T>
static starpu_ssize_t payload_allocate_data_on_node(void *data_interface, unsigned node)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface);

  starpu_ssize_t requested_memory = interface->IsEmpty() ? 0 : interface->GetRequiredBytes();
  void *data = nullptr;
  if (0 != requested_memory) {
    data = reinterpret_cast<void *>(starpu_malloc_on_node(node, requested_memory));
    if (nullptr == data) return -ENOMEM;
  }
  // update the payload
  auto pd = umpalumpa::data::PhysicalDescriptor(data,
    requested_memory,
    interface->dataInfo.GetType(),
    umpalumpa::data::ManagedBy::StarPU,
    node);
  interface->Set(pd);
  return requested_memory;
}

template<typename T> static void payload_free_data_on_node(void *data_interface, unsigned node)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface);
  if (!interface->IsEmpty()) {
    starpu_free_on_node(
      node, reinterpret_cast<uintptr_t>(interface->GetPtr()), interface->dataInfo.GetBytes());
  }
  auto pd = umpalumpa::data::PhysicalDescriptor(
    nullptr, 0, interface->dataInfo.GetType(), umpalumpa::data::ManagedBy::StarPU, node);
  interface->Set(pd);
}

template<typename T>
static int copy_any_to_any(void *src_interface,
  unsigned src_node,
  void *dst_interface,
  unsigned dst_node,
  void *async_data)
{
  auto *src = reinterpret_cast<umpalumpa::data::Payload<T> *>(src_interface);
  auto *dst = reinterpret_cast<umpalumpa::data::Payload<T> *>(dst_interface);

  if (src->IsEmpty()) return 0;// nothing to do
  return starpu_interface_copy(reinterpret_cast<uintptr_t>(src->GetPtr()),
    0,
    src_node,
    reinterpret_cast<uintptr_t>(dst->GetPtr()),
    0,
    dst_node,
    src->GetRequiredBytes(),// copy only what's necessary
    async_data);
}

template<typename T>
static const struct starpu_data_copy_methods payload_copy_methods = { .any_to_any =
                                                                        copy_any_to_any<T> };

template<typename T> static size_t payload_get_size(starpu_data_handle_t handle)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(
    starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM));
  return interface->GetRequiredBytes();
}

template<typename T> static uint32_t payload_footprint(starpu_data_handle_t handle)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(
    starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM));

  return starpu_hash_crc32c_be(interface->info.Elems(), 0);
}

template<typename T> static int payload_compare(void *data_interface_a, void *data_interface_b)
{
  auto *payload_a = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface_a);
  auto *payload_b = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface_b);

  return (payload_a->info.Elems() == payload_b->info.Elems());
}

template<typename T>
static struct starpu_data_interface_ops payload_ops = {
  .register_data_handle = payload_register_data_handle<T>,
  .allocate_data_on_node = payload_allocate_data_on_node<T>,
  .free_data_on_node = payload_free_data_on_node<T>,
  .copy_methods = &payload_copy_methods<T>,
  .get_size = payload_get_size<T>,
  .footprint = payload_footprint<T>,
  .compare = payload_compare<T>,
  .interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
  .interface_size = sizeof(umpalumpa::data::Payload<T>),
};


template<typename T>
void starpu_payload_register(starpu_data_handle_t *handle,
  unsigned home_node,
  const umpalumpa::data::Payload<T> &payload)
{
  if (payload_ops<T>.interfaceid == STARPU_UNKNOWN_INTERFACE_ID) {
    payload_ops<T>.interfaceid =
      static_cast<starpu_data_interface_id>(starpu_data_interface_get_next_id());
  }

  starpu_data_register(handle,
    home_node,
    static_cast<void *>(const_cast<umpalumpa::data::Payload<T> *>(&payload)),
    &payload_ops<T>);
}


// turn the warnings back on
#pragma GCC diagnostic pop
