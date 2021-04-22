#pragma once

#include <libumpalumpa/data/payload.hpp>

// save diagnostic state
#pragma GCC diagnostic push
// turn off warnings
#pragma GCC system_header
#include <starpu.h>

template<typename T>
static void payload_register_data_handle(starpu_data_handle_t handle,
  unsigned home_node,
  void *data_interface)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface);
  for (unsigned node = 0; node < STARPU_MAXNODES; node++) {
    auto *local_interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(
      starpu_data_get_interface_on_node(handle, node));
    if (node == home_node) {
      *local_interface = *interface;
    }
  }
}

template<typename T>
static starpu_ssize_t payload_allocate_data_on_node(void *data_interface, unsigned node)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface);

  starpu_ssize_t requested_memory = interface->dataInfo.bytes;
  void *data = reinterpret_cast<void *>(starpu_malloc_on_node(node, requested_memory));
  if (nullptr == data) return -ENOMEM;
  /* update the data properly in consequence */
  interface->data = data;
  return requested_memory;
}

template<typename T> static void payload_free_data_on_node(void *data_interface, unsigned node)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(data_interface);
  starpu_free_on_node(
    node, reinterpret_cast<uintptr_t>(interface->data), interface->dataInfo.bytes);
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
  *dst = *src; // FIXME this will invalidate data pointer on dst?

  return starpu_interface_copy(reinterpret_cast<uintptr_t>(src->data),
    0,
    src_node,
    reinterpret_cast<uintptr_t>(dst->data),
    0,
    dst_node,
    src->dataInfo.bytes,
    async_data);
}

template<typename T>
static const struct starpu_data_copy_methods payload_copy_methods = { .any_to_any =
                                                                        copy_any_to_any<T> };

template<typename T> static size_t payload_get_size(starpu_data_handle_t handle)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(
    starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM));
  return interface->dataInfo.bytes;
}

template<typename T> static uint32_t payload_footprint(starpu_data_handle_t handle)
{
  auto *interface = reinterpret_cast<umpalumpa::data::Payload<T> *>(
    starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM));

  return starpu_hash_crc32c_be(interface->info.Elems(), 0);
}

template<typename T>
static struct starpu_data_interface_ops payload_ops = {
  .register_data_handle = payload_register_data_handle<T>,
  .allocate_data_on_node = payload_allocate_data_on_node<T>,
  .free_data_on_node = payload_free_data_on_node<T>,
  .copy_methods = &payload_copy_methods<T>,
  .get_size = payload_get_size<T>,
  .footprint = payload_footprint<T>,
  .interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
  .interface_size = sizeof(umpalumpa::data::Payload<T>),
};


template<typename T>
void starpu_payload_register(starpu_data_handle_t *handle,
  unsigned home_node,
  umpalumpa::data::Payload<T> &payload)
{
  if (payload_ops<T>.interfaceid == STARPU_UNKNOWN_INTERFACE_ID) {
    payload_ops<T>.interfaceid =
      static_cast<starpu_data_interface_id>(starpu_data_interface_get_next_id());
  }

  starpu_data_register(handle, home_node, &payload, &payload_ops<T>);
}


// turn the warnings back on
#pragma GCC diagnostic pop
