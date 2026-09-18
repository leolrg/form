#pragma once
// Optional inclusive host wall timers. No synchronization is added to CUDA work.
// GPU durations/copies are obtained separately by correlating CUDA activities
// to NVTX ranges. Parent and child timers must never be added together.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <vector>
#ifdef FORM_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif
namespace form::diagnostics {
inline bool enabled=false, trace_enabled=false;
using Clock=std::chrono::steady_clock;
enum class Stage { optimizer_summary, optimizer_graph, optimizer_values, cpu_reset, gpu_reset, reset_classify, batch_reset, batch_topology, batch_roots, cuda_batch_reset, cpu_frozen_setup, cuda_configure, cuda_configure_host, cuda_configure_upload, cpu_linearize, gpu_linearize, pose_pack, batch_linearize, cpu_summary_edges, cpu_summary_assemble, auxiliary_linearize, cpu_frozen_shift, cuda_linearize, cpu_error, gpu_error, batch_error, auxiliary_error, cuda_error, cpu_solve, gpu_solve, cpu_damping, cpu_cholesky, cpu_backsolve, cpu_model_error, cuda_solve, retraction, lm_accepted, lm_rejected, lm_solve_failed, match_snapshot, match_snapshot_pack, matcher_reset, match_group_setup, match_search_grouped, match_search_launch, match_classify_wait, match_group_sort, match_pack, qr_total, qr_plan, qr_execute, qr_output, match_download, match_ensure, match_reconstruct, match_host_group, match_rows, match_output, extraction_prepare, extraction_search, cpu_frozen_reuse, cuda_topology_reuse, cuda_configure_reuse, count };
inline constexpr const char* names[]={"optimizer_summary","optimizer_graph","optimizer_values","cpu_reset","gpu_reset","reset_classify","batch_reset","batch_topology","batch_roots","cuda_batch_reset","cpu_frozen_setup","cuda_configure","cuda_configure_host","cuda_configure_upload","cpu_linearize","gpu_linearize","pose_pack","batch_linearize","cpu_summary_edges","cpu_summary_assemble","auxiliary_linearize","cpu_frozen_shift","cuda_linearize","cpu_error","gpu_error","batch_error","auxiliary_error","cuda_error","cpu_solve","gpu_solve","cpu_damping","cpu_cholesky","cpu_backsolve","cpu_model_error","cuda_solve","retraction","lm_accepted","lm_rejected","lm_solve_failed","match_snapshot","match_snapshot_pack","matcher_reset","match_group_setup","match_search_grouped","match_search_launch","match_classify_wait","match_group_sort","match_pack","qr_total","qr_plan","qr_execute","qr_output","match_download","match_ensure","match_reconstruct","match_host_group","match_rows","match_output","extraction_prepare","extraction_search","cpu_frozen_reuse","cuda_topology_reuse","cuda_configure_reuse"};
inline constexpr size_t count=static_cast<size_t>(Stage::count);
inline std::array<std::atomic<uint64_t>,count> nanoseconds{}, calls{};
struct Range {
  bool active;
  explicit Range(const char* name):active(trace_enabled) {
#ifdef FORM_ENABLE_CUDA
    if(active) nvtxRangePushA(name);
#endif
  }
  ~Range() {
#ifdef FORM_ENABLE_CUDA
    if(active) nvtxRangePop();
#endif
  }
  Range(const Range&)=delete; Range& operator=(const Range&)=delete;
};
struct Scope {
  size_t index;
  bool active;
  Clock::time_point start;
  Range range;
  explicit Scope(Stage stage):index(static_cast<size_t>(stage)),active(enabled),range(names[index]) {
    if(active) start=Clock::now();
  }
  ~Scope() {
    if(active) {
      nanoseconds[index].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count(),std::memory_order_relaxed);
      calls[index].fetch_add(1,std::memory_order_relaxed);
    }
  }
  Scope(const Scope&)=delete; Scope& operator=(const Scope&)=delete;
};
inline void tick(Stage stage) { if(enabled) calls[static_cast<size_t>(stage)].fetch_add(1,std::memory_order_relaxed); }
struct OptimizerRecord {
  bool gpu,fast,resident=true; size_t dimension; double wall_ms;
  std::array<uint64_t,count> ns{},n{};
};
// Only the estimator caller records optimizations; reset/write outside its timer.
inline std::vector<OptimizerRecord> optimizers;
struct OptimizerCall {
  bool active; OptimizerRecord record{}; Clock::time_point start;
  OptimizerCall(bool gpu,bool fast,size_t dimension,bool resident=true):active(enabled) {
    if(!active) return;
    record.gpu=gpu;record.fast=fast;record.dimension=dimension;record.resident=resident;
    for(size_t i=0;i<count;++i) {record.ns[i]=nanoseconds[i].load();record.n[i]=calls[i].load();}
    start=Clock::now();
  }
  ~OptimizerCall() {
    if(!active) return;
    record.wall_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    for(size_t i=0;i<count;++i) {record.ns[i]=nanoseconds[i].load()-record.ns[i];record.n[i]=calls[i].load()-record.n[i];}
    optimizers.push_back(record);
  }
};
inline void reset() {for(size_t i=0;i<count;++i){nanoseconds[i]=0;calls[i]=0;}optimizers.clear();}
inline void writeHeader(std::ostream& out) {for(auto name:names) out<<",diag_"<<name<<"_ms,diag_"<<name<<"_calls";}
inline void writeRow(std::ostream& out) {for(size_t i=0;i<count;++i)out<<','<<nanoseconds[i].load()*1e-6<<','<<calls[i].load();}
inline void writeOptimizers(std::ostream& out,size_t scan) {
  for(size_t k=0;k<optimizers.size();++k) {
    const auto& r=optimizers[k];out<<scan<<','<<k<<','<<(r.fast?"semi":"full")<<','<<(r.gpu?(r.resident?"gpu":"cpu+gpu"):"cpu")<<','<<r.dimension<<','<<r.wall_ms<<','<<(r.resident?"resident":"dense");
    for(size_t i=0;i<count;++i)out<<','<<r.ns[i]*1e-6<<','<<r.n[i];out<<'\n';
  }
}
} // namespace form::diagnostics
