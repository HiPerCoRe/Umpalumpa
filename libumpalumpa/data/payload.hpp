#ifndef LIBUMPALUMPA_DATA_PAYLOAD
#define LIBUMPALUMPA_DATA_PAYLOAD
#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/physical_desriptor.hpp>

namespace umpalumpa {
namespace data {
template <typename T>
class Payload {
   public:
    Payload(T *d, const LogicalDescriptor &ld, const PhysicalDescriptor &pd) : data(d), info(ld), dataInfo(pd) {}
    T *const data;
    const LogicalDescriptor info;
    const PhysicalDescriptor dataInfo;
};
}  // namespace data
}  // namespace umpalumpa
#endif /* LIBUMPALUMPA_DATA_PAYLOAD */
