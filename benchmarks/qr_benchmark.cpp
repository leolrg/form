#include "form/feature/cuda_qr.hpp"
#include "qr_capture.hpp"
#include <Eigen/QR>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

using Clock=std::chrono::steady_clock;
using Matrices=std::vector<Eigen::MatrixXd>;
namespace {
Matrices cpuQr(const Matrices& input) {
  Matrices roots;roots.reserve(input.size());
  for(const auto& matrix:input) {
    Eigen::MatrixXd root=Eigen::MatrixXd::Zero(matrix.cols(),matrix.cols());
    if(matrix.rows()) {
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrix);
      const auto rows=std::min(matrix.rows(),matrix.cols());
      root.topRows(rows)=qr.matrixQR().topRows(rows).template triangularView<Eigen::Upper>();
    }
    roots.emplace_back(std::move(root));
  }
  return roots;
}
double gramError(const Matrices& input,const Matrices& roots) {
  if(input.size()!=roots.size()) throw std::runtime_error("QR output matrix count differs");
  double worst=0;
  for(size_t i=0;i<input.size();++i) {
    if(!input[i].allFinite() || !roots[i].allFinite())
      throw std::runtime_error("Nonfinite QR input/output");
    const Eigen::MatrixXd expected=input[i].transpose()*input[i];
    const Eigen::MatrixXd actual=roots[i].transpose()*roots[i];
    worst=std::max(worst,(actual-expected).norm()/std::max(1.,expected.norm()));
  }
  return worst;
}
double percentile(std::vector<double> values,double p) {
  std::sort(values.begin(),values.end());
  const double index=p*(values.size()-1);
  const size_t low=static_cast<size_t>(index),high=std::min(low+1,values.size()-1);
  return values[low]+(index-low)*(values[high]-values[low]);
}
struct Measurement {
  size_t matrices=0,rows=0,values=0,cols7=0,cols13=0;
  std::vector<double> gpu,cpu;
  double gpu_error=0,cpu_error=0,checksum=0;
};
void printRow(std::ostream& out,const char* scope,const std::string& name,
              bool pageable,const Measurement& m) {
  const double gpu_mean=std::accumulate(m.gpu.begin(),m.gpu.end(),0.)/m.gpu.size();
  const double cpu_mean=std::accumulate(m.cpu.begin(),m.cpu.end(),0.)/m.cpu.size();
  out << scope << ',' << name << ',' << (pageable?"pageable":"pinned") << ','
      << m.matrices << ',' << m.rows << ',' << m.values << ',' << m.cols7 << ',' << m.cols13 << ','
      << m.gpu.size() << ',' << gpu_mean << ',' << percentile(m.gpu,.5) << ','
      << percentile(m.gpu,.95) << ',' << percentile(m.gpu,.99) << ','
      << cpu_mean << ',' << percentile(m.cpu,.5) << ',' << percentile(m.cpu,.95) << ','
      << percentile(m.cpu,.99) << ',' << cpu_mean/gpu_mean << ',' << m.gpu_error << ','
      << m.cpu_error << ',' << m.checksum << '\n';
}
void synthetic(form::BatchedCudaQr& gpu,int repeat) {
  // Preserve the original synthetic workload and five-column output.
  for(int pairs:{1,10,30}) for(int count:{100,1000,10000}) {
    Matrices inputs;
    for(int i=0;i<pairs;++i) inputs.emplace_back(Eigen::MatrixXd::Random(count,13));
    gpu.compute(inputs);
    auto start=Clock::now();double checksum=0;
    for(int k=0;k<repeat;++k) {auto out=gpu.compute(inputs);checksum+=out[0](0,0);}
    const double gpu_us=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/repeat;
    start=Clock::now();
    for(int k=0;k<repeat;++k) for(const auto& a:inputs) {
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(a);checksum+=qr.matrixQR()(0,0);
    }
    const double cpu_us=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/repeat;
    std::cout << pairs << ',' << count << ',' << gpu_us << ',' << cpu_us << ',' << checksum << std::endl;
  }
}
} // namespace
int main(int argc,char** argv) {
  try {
    std::filesystem::path directory;
    int repeat=30,first_rows=64,reduction_rows=64,block_threads=32;bool pageable=false;
    for(int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if(arg=="--pageable") {pageable=true;continue;}
      if(i+1==argc) throw std::runtime_error("Missing value for "+arg);
      const std::string value=argv[++i];
      if(arg=="--input-dir") directory=value;
      else if(arg=="--repeat") repeat=std::stoi(value);
      else if(arg=="--first-rows") first_rows=std::stoi(value);
      else if(arg=="--reduction-rows") reduction_rows=std::stoi(value);
      else if(arg=="--block-threads") block_threads=std::stoi(value);
      else throw std::runtime_error("Unknown argument "+arg);
    }
    if(repeat<=0) throw std::runtime_error("--repeat must be positive");
    form::BatchedCudaQr gpu(!pageable,first_rows,reduction_rows,block_threads);
    if(directory.empty()) {synthetic(gpu,repeat);return 0;}
    if(std::filesystem::exists(directory/"manifest.json.partial"))
      throw std::runtime_error("Capture is incomplete: manifest.json.partial exists");
    std::vector<std::filesystem::path> files;
    for(const auto& entry:std::filesystem::directory_iterator(directory))
      if(entry.is_regular_file() && entry.path().extension()==".formqr") files.push_back(entry.path());
    std::sort(files.begin(),files.end());
    if(files.empty()) throw std::runtime_error("No .formqr batches in input directory");
    Measurement aggregate;aggregate.gpu.resize(repeat);aggregate.cpu.resize(repeat);
    std::cout << std::setprecision(12)
      << "scope,batch,host_memory,matrices,total_rows,total_values,cols7,cols13,repeats,"
      << "gpu_mean_us,gpu_p50_us,gpu_p95_us,gpu_p99_us,cpu_mean_us,cpu_p50_us,cpu_p95_us,cpu_p99_us,"
      << "cpu_over_gpu,gpu_gram_relative_error,cpu_gram_relative_error,checksum\n";
    for(const auto& file:files) {
      const auto inputs=form_benchmark::readBatch(file);
      Measurement m;m.gpu.resize(repeat);m.cpu.resize(repeat);m.matrices=inputs.size();
      for(const auto& matrix:inputs) {
        m.rows+=matrix.rows();m.values+=matrix.size();
        if(matrix.cols()==7) ++m.cols7;else ++m.cols13;
      }
      // Each batch warms the same GPU object, retaining capacity across all batches.
      // Both complete output paths are checked outside the measured interval.
      m.gpu_error=gramError(inputs,gpu.compute(inputs));
      m.cpu_error=gramError(inputs,cpuQr(inputs));
      if(m.gpu_error>1e-9 || m.cpu_error>1e-9)
        throw std::runtime_error("QR Gram error exceeds 1e-9 for "+file.string());
      auto timeGpu=[&](int k) {
        const auto start=Clock::now();auto roots=gpu.compute(inputs);
        m.gpu[k]=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        m.checksum+=roots.front()(0,0);
      };
      auto timeCpu=[&](int k) {
        const auto start=Clock::now();auto roots=cpuQr(inputs);
        m.cpu[k]=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        m.checksum+=roots.front()(0,0);
      };
      for(int k=0;k<repeat;++k) {
        if(k%2) {timeCpu(k);timeGpu(k);} else {timeGpu(k);timeCpu(k);}
        aggregate.gpu[k]+=m.gpu[k];aggregate.cpu[k]+=m.cpu[k];
      }
      printRow(std::cout,"batch",file.filename().string(),pageable,m);
      aggregate.matrices+=m.matrices;aggregate.rows+=m.rows;aggregate.values+=m.values;
      aggregate.cols7+=m.cols7;aggregate.cols13+=m.cols13;aggregate.checksum+=m.checksum;
      aggregate.gpu_error=std::max(aggregate.gpu_error,m.gpu_error);
      aggregate.cpu_error=std::max(aggregate.cpu_error,m.cpu_error);
    }
    printRow(std::cout,"aggregate","ALL",pageable,aggregate);
    std::cerr << "Replayed " << files.size() << " captured batches, " << repeat
              << " repeats; GPU includes packing, transfers, kernels, synchronization and output construction.\n";
  } catch(const std::exception& error) {
    std::cerr << error.what() << '\n';return 1;
  }
}
