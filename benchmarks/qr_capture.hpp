#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace form_benchmark {
// Wire format is little endian; doubles are IEEE754 in column-major order.
inline void checkNativeFormat() {
  static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
  static_assert(!Eigen::MatrixXd::IsRowMajor);
  const uint16_t one=1;
  if (*reinterpret_cast<const unsigned char*>(&one)!=1)
    throw std::runtime_error("FORMQR01 requires a little-endian host");
}
inline void writeBatch(const std::filesystem::path& path,
                       const std::vector<Eigen::MatrixXd>& inputs) {
  checkNativeFormat();
  if (inputs.empty() || inputs.size()>100000)
    throw std::runtime_error("Invalid QR matrix count");
  if (std::filesystem::exists(path))
    throw std::runtime_error("Refusing to overwrite QR batch "+path.string());
  std::ofstream out(path,std::ios::binary);
  out.exceptions(std::ios::badbit|std::ios::failbit);
  out.write("FORMQR01",8);
  const uint32_t count=static_cast<uint32_t>(inputs.size());
  out.write(reinterpret_cast<const char*>(&count),4);
  for (const auto& matrix:inputs) {
    if ((matrix.cols()!=7 && matrix.cols()!=13) || matrix.rows()>std::numeric_limits<int>::max())
      throw std::runtime_error("Unsupported QR matrix shape");
    const uint64_t rows=matrix.rows(); const uint32_t cols=matrix.cols();
    out.write(reinterpret_cast<const char*>(&rows),8);
    out.write(reinterpret_cast<const char*>(&cols),4);
    if (matrix.size()) out.write(reinterpret_cast<const char*>(matrix.data()),matrix.size()*8);
  }
}
inline std::vector<Eigen::MatrixXd> readBatch(const std::filesystem::path& path) {
  checkNativeFormat();
  uint64_t remaining=std::filesystem::file_size(path);
  std::ifstream in(path,std::ios::binary);
  auto read=[&](void* data,uint64_t bytes) {
    if (bytes>remaining || !in.read(static_cast<char*>(data),static_cast<std::streamsize>(bytes)))
      throw std::runtime_error("Truncated QR batch "+path.string());
    remaining-=bytes;
  };
  char magic[8];uint32_t count;
  read(magic,8);read(&count,4);
  if (std::memcmp(magic,"FORMQR01",8) || !count || count>100000 || count>remaining/12)
    throw std::runtime_error("Invalid QR batch header "+path.string());
  std::vector<Eigen::MatrixXd> inputs;inputs.reserve(count);
  for(uint32_t i=0;i<count;++i) {
    uint64_t rows;uint32_t cols;read(&rows,8);read(&cols,4);
    if ((cols!=7 && cols!=13) || rows>std::numeric_limits<int>::max() || rows>remaining/(8*cols))
      throw std::runtime_error("Invalid QR matrix dimensions "+path.string());
    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(rows),cols);
    if(rows) read(matrix.data(),rows*cols*8);
    inputs.emplace_back(std::move(matrix));
  }
  if(remaining) throw std::runtime_error("Trailing QR batch data "+path.string());
  return inputs;
}
} // namespace form_benchmark
