#include "form/form.hpp"
#include "form/optimization/profile.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <set>
#include <limits>
#ifdef FORM_ENABLE_CUDA
#include "form/feature/cuda_qr.hpp"
#include "qr_capture.hpp"
#include <mutex>
#endif

namespace {
template<class T> bool read(std::istream& in, T& value) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}
#ifdef FORM_ENABLE_CUDA
class QrCapture {
public:
  QrCapture(const std::string& directory,size_t scan,int64_t stamp) : directory_(directory) {
    if(std::filesystem::exists(directory_) && !std::filesystem::is_empty(directory_))
      throw std::runtime_error("QR capture directory must be new or empty");
    std::filesystem::create_directories(directory_);
    manifest_.open(directory_/"manifest.json.partial");
    manifest_.exceptions(std::ios::badbit|std::ios::failbit);
    manifest_ << "{\n  \"format\": \"FORMQR01\",\n  \"scan_index\": " << scan
              << ",\n  \"stamp_ns\": " << stamp
              << ",\n  \"timing_note\": \"Diagnostic capture: host serialization perturbs scan timing\",\n  \"batches\": [";
    active_=this;
    form::BatchedCudaQr::setInputObserver(&observe);
  }
  ~QrCapture() { form::BatchedCudaQr::setInputObserver(nullptr);active_=nullptr; }
  void finish() {
    form::BatchedCudaQr::setInputObserver(nullptr);active_=nullptr;
    manifest_ << "\n  ]\n}\n";manifest_.close();
    std::filesystem::rename(directory_/"manifest.json.partial",directory_/"manifest.json");
    if(!batches_) throw std::runtime_error("Requested scan contained no CUDA QR batches");
    std::cerr << "Captured " << batches_ << " real CUDA QR batches in " << directory_ << '\n';
  }
private:
  static void observe(const std::vector<Eigen::MatrixXd>& inputs) {
    if(active_) active_->write(inputs);
  }
  void write(const std::vector<Eigen::MatrixXd>& inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream name;name << "batch-" << std::setfill('0') << std::setw(6) << batches_ << ".formqr";
    form_benchmark::writeBatch(directory_/name.str(),inputs);
    if(batches_) manifest_ << ',';
    manifest_ << "\n    {\"file\": \"" << name.str() << "\", \"shapes\": [";
    for(size_t i=0;i<inputs.size();++i) {
      if(i) manifest_ << ',';
      manifest_ << '[' << inputs[i].rows() << ',' << inputs[i].cols() << ']';
    }
    manifest_ << "]}";manifest_.flush();++batches_;
  }
  inline static QrCapture* active_=nullptr;
  std::filesystem::path directory_;
  std::ofstream manifest_;
  std::mutex mutex_;
  size_t batches_=0;
};
#endif

class SolveCapture {
public:
  SolveCapture(const std::filesystem::path& directory,size_t scan,int64_t stamp)
      : directory_(directory) {
    if(std::filesystem::exists(directory_) && !std::filesystem::is_empty(directory_))
      throw std::runtime_error("System capture directory must be new or empty");
    std::filesystem::create_directories(directory_);
    manifest_.open(directory_/"manifest.json.partial");
    manifest_.exceptions(std::ios::badbit|std::ios::failbit);
    manifest_ << "{\n  \"format\": \"FORMSYS1\",\n  \"scan_index\": " << scan
              << ",\n  \"stamp_ns\": " << stamp
              << ",\n  \"timing_note\": \"Diagnostic capture; scan timing includes serialization\",\n  \"systems\": [";
    active_=this;form::profile::solve_observer=&observe;
  }
  ~SolveCapture() { form::profile::solve_observer=nullptr;active_=nullptr; }
  void finish() {
    form::profile::solve_observer=nullptr;active_=nullptr;
    if(!systems_) throw std::runtime_error("Requested scan contained no dense solves");
    manifest_ << "\n  ]\n}\n";manifest_.close();
    std::filesystem::rename(directory_/"manifest.json.partial",directory_/"manifest.json");
    std::cerr << "Captured " << systems_ << " damped systems in " << directory_ << '\n';
  }
private:
  static void observe(const Eigen::MatrixXd& augmented) { active_->write(augmented); }
  void write(const Eigen::MatrixXd& augmented) {
    const auto size=augmented.rows()-1;
    if(size<=0 || size>std::numeric_limits<uint32_t>::max() || augmented.cols()!=size+1)
      throw std::runtime_error("Invalid augmented system dimension");
    const uint32_t n=static_cast<uint32_t>(size);
    const uint32_t endian_check=1;
    if(*reinterpret_cast<const unsigned char*>(&endian_check)!=1)
      throw std::runtime_error("System capture requires a little-endian host");
    std::ostringstream name;name << "system-" << std::setfill('0') << std::setw(6) << systems_ << ".formsys";
    const auto path=directory_/name.str();
    if(std::filesystem::exists(path)) throw std::runtime_error("System capture already exists");
    std::ofstream out(path,std::ios::binary);
    out.exceptions(std::ios::badbit|std::ios::failbit);
    out.write("FORMSYS1",8);out.write(reinterpret_cast<const char*>(&n),sizeof(n));
    for(uint32_t col=0;col<n;++col)
      out.write(reinterpret_cast<const char*>(augmented.col(col).data()),n*sizeof(double));
    out.write(reinterpret_cast<const char*>(augmented.col(n).data()),n*sizeof(double));
    out.close();
    if(systems_) manifest_ << ',';
    manifest_ << "\n    {\"file\": \"" << name.str() << "\", \"dimension\": " << n << '}';
    manifest_.flush();++systems_;
  }
  inline static SolveCapture* active_=nullptr;
  std::filesystem::path directory_;
  std::ofstream manifest_;
  size_t systems_=0;
};
}
int main(int argc, char** argv) {
  try {
    std::string input, output, capture_directory, system_capture_directory;
    std::set<size_t> system_capture_scans, captured_system_scans;
    size_t limit = 0, capture_scan = 200;
    bool did_capture=false;
    form::Estimator::Params params;
    params.num_threads = 8;
    params.extraction.min_norm_squared = 0.1 * 0.1;
    params.extraction.max_norm_squared = 50.0 * 50.0;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--batch-summaries") { params.constraints.use_batch_summaries = true; continue; }
      if (arg == "--profile") { form::profile::enabled = true; continue; }
      if (i + 1 == argc) throw std::runtime_error("Missing value for " + arg);
      const std::string value = argv[++i];
      if (arg == "--input") input = value;
      else if (arg == "--output") output = value;
      else if (arg == "--capture-qr") capture_directory = value;
      else if (arg == "--capture-systems") system_capture_directory=value;
      else if (arg == "--capture-system-scans") {
        std::stringstream values(value);std::string item;
        while(std::getline(values,item,',')) system_capture_scans.insert(std::stoull(item));
      }
      else if (arg == "--capture-scan") capture_scan = std::stoull(value);
      else if (arg == "--limit") limit = std::stoull(value);
      else if (arg == "--threads") params.num_threads = std::stoull(value);
      else if (arg == "--points") params.extraction.point_feats_per_sector = std::stoull(value);
      else if (arg == "--planes") params.extraction.planar_feats_per_sector = std::stoull(value);
      else if (arg == "--feature-spacing") params.extraction.feature_spacing = std::stoull(value);
      else if (arg == "--recent") params.scans.max_num_recent_scans = std::stoull(value);
      else if (arg == "--cuda-solve-min-dimension") {
        params.constraints.cuda_solve_min_dimension = std::stoi(value);
        if (params.constraints.cuda_solve_min_dimension < 1)
          throw std::runtime_error("CUDA solve minimum dimension must be positive");
        params.constraints.use_cuda_dense_solver = true;
      }
      else if (arg == "--backend") {
        if (value != "reference" && value != "summary" && value != "cuda" && value != "summary-batch" && value != "cuda-batch" && value != "summary-resident" && value != "cuda-resident" && value != "cuda-resident-hybrid" && value != "cuda-matching") throw std::runtime_error("Unknown backend " + value);
        params.constraints.use_summary = value != "reference";
        params.constraints.use_cuda_summaries = value == "cuda" || value == "cuda-batch" || value == "cuda-resident" || value == "cuda-resident-hybrid" || value == "cuda-matching";
        params.constraints.use_resident_optimizer = value == "summary-resident" || value == "cuda-resident" || value == "cuda-resident-hybrid" || value == "cuda-matching";
        params.matcher.use_cuda = value == "cuda-matching";
        if(value == "cuda-resident-hybrid" || value == "cuda-matching") params.constraints.use_cuda_dense_solver = true;
        if(value == "summary-batch" || value == "cuda-batch" || params.constraints.use_resident_optimizer) params.constraints.use_batch_summaries = true;
      }
      else throw std::runtime_error("Unknown argument " + arg);
    }
    if (input.empty() || output.empty())
      throw std::runtime_error("Usage: form-replay --input sequence.formpc --output prefix [--limit N] [--threads N] [--points N] [--recent N] [--profile]");
    if(!system_capture_directory.empty()) {
      if(!form::profile::enabled) throw std::runtime_error("System capture requires --profile");
      if(system_capture_scans.empty()) system_capture_scans.insert(capture_scan);
      if(limit && *system_capture_scans.rbegin()>=limit)
        throw std::runtime_error("System capture scans must be below --limit");
      if(std::filesystem::exists(system_capture_directory) && !std::filesystem::is_empty(system_capture_directory))
        throw std::runtime_error("System capture root must be new or empty");
      std::cerr << "Diagnostic system capture enabled; selected scan timings include serialization\n";
    }
    if(!capture_directory.empty()) {
#ifndef FORM_ENABLE_CUDA
      throw std::runtime_error("QR capture requires FORM_ENABLE_CUDA");
#else
      if(!params.constraints.use_cuda_summaries)
        throw std::runtime_error("QR capture requires --backend cuda");
      if(limit && capture_scan>=limit)
        throw std::runtime_error("Capture scan is zero-based and must be below --limit");
      std::cerr << "Diagnostic QR capture enabled: captured scan timing includes serialization overhead\n";
#endif
    }
    std::ifstream in(input, std::ios::binary);
    char magic[8]; uint32_t rows = 0, cols = 0;
    if (!in.read(magic, 8) || std::memcmp(magic, "FORMPC01", 8) ||
        !read(in, rows) || !read(in, cols) || !rows || !cols ||
        static_cast<uint64_t>(rows) * cols > 2000000)
      throw std::runtime_error("Invalid FORMPC01 input header");
    params.extraction.num_rows = rows; params.extraction.num_columns = cols;
    std::ofstream timing(output + ".csv"), poses(output + ".tum");
    if (!timing || !poses) throw std::runtime_error("Cannot open output prefix");
    timing << "scan,stamp_ns,total_ms,extract_ms,map_ms,match_ms,semi_ms,full_ms,marginalize_ms,maintenance_ms,poses,planar_features,point_features,rematches,lm_iterations,factor_linearize_cpu_ms,factor_eval_cpu_ms,factor_error_cpu_ms,linearize_wall_ms,assemble_ms,solve_ms,factors,planar_correspondences,point_correspondences,summary_build_cpu_ms,summary_prepare_wall_ms,full_initial_error,full_final_error,cuda_solve_calls,cuda_solve_fallbacks\n";
    timing << std::setprecision(12); poses << std::setprecision(17);
    form::Estimator estimator(params);
    std::vector<form::PointXYZf> scan;
    size_t index = 0;
    static_assert(sizeof(form::PointXYZf) == 4 * sizeof(float));
    while (!limit || index < limit) {
      int64_t stamp; uint32_t count;
      if (!read(in, stamp)) {
        if (in.eof() && in.gcount() == 0) break;
        throw std::runtime_error("Truncated scan timestamp");
      }
      if (!read(in, count) || count != rows * cols)
        throw std::runtime_error("Invalid scan point count");
      scan.resize(count, form::PointXYZf(0, 0, 0));
      if (!in.read(reinterpret_cast<char*>(scan.data()), count * sizeof(scan[0])))
        throw std::runtime_error("Truncated scan data");
#ifdef FORM_ENABLE_CUDA
      std::unique_ptr<QrCapture> capture;
      if(!capture_directory.empty() && index==capture_scan)
        capture=std::make_unique<QrCapture>(capture_directory,index,stamp);
#endif
      std::unique_ptr<SolveCapture> solve_capture;
      if(!system_capture_directory.empty() && system_capture_scans.count(index))
        solve_capture=std::make_unique<SolveCapture>(std::filesystem::path(system_capture_directory)/("scan-"+std::to_string(index)),index,stamp);
      form::profile::reset();
      const auto start = form::profile::Clock::now();
      estimator.register_scan(scan);
      const double total = form::profile::milliseconds(start);
#ifdef FORM_ENABLE_CUDA
      if(capture) { capture->finish();capture.reset();did_capture=true; }
#endif
      if(solve_capture) {solve_capture->finish();solve_capture.reset();captured_system_scans.insert(index);}
      const auto& t = estimator.last_timing;
      timing << index << ',' << stamp << ',' << total << ',' << t.extract_ms << ','
             << t.map_ms << ',' << t.match_ms << ',' << t.semi_ms << ',' << t.full_ms << ','
             << t.marginalize_ms << ',' << t.maintenance_ms << ',' << t.poses << ','
             << t.planar_features << ',' << t.point_features << ',' << t.rematches << ','
             << form::profile::iterations << ',' << form::profile::ms(form::profile::factor_linearize) << ','
             << form::profile::ms(form::profile::factor_eval) << ',' << form::profile::ms(form::profile::factor_error) << ','
             << form::profile::ms(form::profile::linearize_wall) << ',' << form::profile::ms(form::profile::assemble) << ','
             << form::profile::ms(form::profile::solve) << ',' << t.factors << ','
             << t.planar_correspondences << ',' << t.point_correspondences << ','
             << form::profile::ms(form::profile::summary_build_cpu) << ','
             << form::profile::ms(form::profile::summary_prepare_wall) << ','
             << form::profile::last_initial_error.load() << ','
             << form::profile::last_final_error.load() << ','
             << form::profile::cuda_solve_calls.load() << ','
             << form::profile::cuda_solve_fallbacks.load() << '\n';
      const auto pose = estimator.current_lidar_estimate();
      const auto q = pose.rotation().toQuaternion();
      poses << stamp / 1000000000 << '.' << std::setfill('0') << std::setw(9)
            << stamp % 1000000000 << std::setfill(' ') << ' '
            << pose.x() << ' ' << pose.y() << ' ' << pose.z() << ' '
            << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
      if (++index % 100 == 0) {
        std::cerr << "scan " << index << ": " << total << " ms, " << t.poses << " poses\n";
        timing.flush(); poses.flush();
      }
    }
    if (!index) throw std::runtime_error("Input contains no scans");
    if(!capture_directory.empty() && !did_capture)
      throw std::runtime_error("Requested capture scan was not reached");
    if(!system_capture_directory.empty() && captured_system_scans!=system_capture_scans)
      throw std::runtime_error("Requested system capture scans were not all reached");
    std::cerr << "Completed " << index << " scans\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
