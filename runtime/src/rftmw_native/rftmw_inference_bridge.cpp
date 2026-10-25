#include "rftmw_inference.hpp"

namespace rftmw {
namespace inference {

void dense_sgemv_f32_bridge(
    const float* weights,
    const float* x,
    float* y,
    int rows,
    int cols
) {
    dense_sgemv_f32(weights, x, y, rows, cols);
}

} // namespace inference
} // namespace rftmw
