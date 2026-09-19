#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include "form/optimization/cuda_matcher.hpp"

namespace form::detail {
// Opt-in, deliberately synchronous diagnostic. Never enable in timing runs.
class MatchAudit {
  std::ofstream output_;
  std::ofstream stats_;
  std::mutex mutex_;
  template<class T> void write(const T& value) {
    output_.write(reinterpret_cast<const char*>(&value),sizeof(value));
  }
public:
  MatchAudit() {
    const char* stats_path=std::getenv("FORM_MATCH_STATS_PATH");
    if(stats_path && *stats_path) {
      stats_.exceptions(std::ios::failbit|std::ios::badbit);
      stats_.open(stats_path);
      stats_ << "scan,kind,total,certified,searched,cell_fallback,gap_fallback,unchanged,mismatches,oracle_searched\n";
    }
    const char* path=std::getenv("FORM_MATCH_AUDIT_PATH");
    if(!path || !*path) return;
    const uint16_t endian=1;
    if(*reinterpret_cast<const unsigned char*>(&endian)!=1)
      throw std::runtime_error("Match audit requires little-endian host");
    output_.exceptions(std::ios::failbit|std::ios::badbit);
    output_.open(path,std::ios::binary|std::ios::trunc);
    output_.write("FORMMAT1",8);
  }
  bool enabled() const { return output_.is_open(); }
  bool statsEnabled() const { return stats_.is_open(); }
  void appendStats(size_t scan,int kind,const CudaMatcher::ReuseStats& s) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ << scan << ',' << kind << ',' << s.total << ',' << s.certified << ',' << s.searched
           << ',' << s.cell_fallback << ',' << s.gap_fallback << ',' << s.unchanged << ',' << s.mismatches
           << ',' << s.oracle_searched << '\n';
    stats_.flush();
  }
  void append(size_t scan,int kind,const std::vector<size_t>& groups,
              const std::vector<int>& target_groups,double threshold,
              const std::vector<CudaMatcher::Result>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    for(uint64_t n:{uint64_t(scan),uint64_t(kind),uint64_t(results.size()),uint64_t(groups.size())}) write(n);
    for(size_t group:groups) write(uint64_t(group));
    for(const auto& result:results) {
      const int32_t index=result.index;
      const int32_t group=index>=0 && result.distance<threshold ? target_groups.at(index) : -1;
      write(index); write(group); write(result.distance);
    }
    output_.flush();
  }
};
inline MatchAudit& matchAudit() { static MatchAudit audit; return audit; }
}
