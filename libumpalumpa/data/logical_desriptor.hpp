#ifndef LIBUMPALUMPA_DATA_LOGICAL_DESRIPTOR
#define LIBUMPALUMPA_DATA_LOGICAL_DESRIPTOR
#include <libumpalumpa/data/size.hpp>

namespace umpalumpa {
namespace data {
class LogicalDescriptor {
   public:
    LogicalDescriptor(const Size &size, const Size &paddedSize) : size(size), paddedSize(paddedSize) {}
    bool IsValid() const {
        return true;  // FIXME implement: paddedSize >= size;
    }

    const Size size;
    const Size paddedSize;
};
}  // namespace data
}  // namespace umpalumpa
#endif /* LIBUMPALUMPA_DATA_LOGICAL_DESRIPTOR */
