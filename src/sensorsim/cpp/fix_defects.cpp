// Defect interpolation: mean of valid 8-neighbours, filled in passes.
// Port of sensorsim.correction.defect_fix.fix_defects (NumPy).
//
// Behaviour kept identical to the NumPy version:
//  - a pixel filled in a pass only uses pixels valid at the START of that pass
//  - neighbours are summed in the same order (dy outer, dx inner)
//  - NaN / inf are treated as invalid
//  - same four ValueError cases
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::array_t<double> fix_defects(
    py::array_t<double, py::array::c_style | py::array::forcecast> image,
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask,
    int max_iter = 50) {
  if (image.ndim() != 2) {
    throw std::invalid_argument("image must be 2-D");
  }
  if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) ||
      mask.shape(1) != image.shape(1)) {
    throw std::invalid_argument("shape mismatch between image and mask");
  }

  const py::ssize_t h = image.shape(0);
  const py::ssize_t w = image.shape(1);
  const py::ssize_t n = h * w;
  const double* img = image.data();
  const bool* msk = mask.data();

  std::vector<double> out(static_cast<size_t>(n));
  std::vector<unsigned char> valid(static_cast<size_t>(n));
  std::vector<py::ssize_t> pending;  // pixels still to fill

  for (py::ssize_t i = 0; i < n; ++i) {
    const bool ok = !msk[i] && std::isfinite(img[i]);
    valid[i] = ok ? 1 : 0;
    out[i] = ok ? img[i] : 0.0;
    if (!ok) pending.push_back(i);
  }
  if (static_cast<py::ssize_t>(pending.size()) == n) {
    throw std::invalid_argument("no valid pixels to interpolate from");
  }

  std::vector<py::ssize_t> filled, still;
  for (int it = 0; it < max_iter && !pending.empty(); ++it) {
    filled.clear();
    still.clear();
    for (const py::ssize_t idx : pending) {
      const py::ssize_t y = idx / w;
      const py::ssize_t x = idx % w;
      double s = 0.0;
      int c = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dy == 0 && dx == 0) continue;
          const py::ssize_t yy = y + dy;
          const py::ssize_t xx = x + dx;
          if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
          const py::ssize_t nidx = yy * w + xx;
          if (valid[nidx]) {
            s += out[nidx];
            ++c;
          }
        }
      }
      if (c > 0) {
        out[idx] = s / static_cast<double>(c);
        filled.push_back(idx);
      } else {
        still.push_back(idx);
      }
    }
    // mark valid only after the whole pass (matches "valid |= fill" in NumPy)
    for (const py::ssize_t idx : filled) valid[idx] = 1;
    pending.swap(still);
  }

  if (!pending.empty()) {
    throw std::invalid_argument("could not fill all pixels within max_iter passes");
  }

  py::array_t<double> result(std::vector<py::ssize_t>{h, w});
  std::memcpy(result.mutable_data(), out.data(),
              static_cast<size_t>(n) * sizeof(double));
  return result;
}

PYBIND11_MODULE(_fixcore, m) {
  m.doc() = "C++ defect interpolation (port of the NumPy fix_defects)";
  m.def("fix_defects", &fix_defects, py::arg("image"), py::arg("mask"),
        py::arg("max_iter") = 50,
        "Replace masked / non-finite pixels with the mean of valid 8-neighbours.");
}