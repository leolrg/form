#include "form/feature/cuda_qr.hpp"
#include "qr_capture.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

// Same-process, interleaved completed-computation comparison. Each configuration
// owns reusable buffers, all warmed before measurement. Input I/O and Gram
// checks are outside timing. No CPU speedup claim is made by this experiment.
int main(int argc,char** argv) {
  try {
    if(argc<2 || argc>3) throw std::invalid_argument("Usage: form-qr-tiles CAPTURE_DIR [REPEATS]");
    const int repeats=argc==3?std::stoi(argv[2]):30;
    if(repeats<1) throw std::invalid_argument("Positive repeat count required");
    const std::filesystem::path directory=argv[1];
    if(std::filesystem::exists(directory/"manifest.json.partial"))
      throw std::runtime_error("Incomplete capture");
    const std::vector<std::array<int,3>> configs{{32,32,128},{32,32,64},{32,64,128},{32,64,64},
                                               {64,64,32},{64,64,64},{64,64,128},{64,64,256}};
    std::vector<std::unique_ptr<form::BatchedCudaQr>> qr;
    for(auto c:configs) qr.emplace_back(std::make_unique<form::BatchedCudaQr>(true,c[0],c[1],c[2]));
    std::vector<std::filesystem::path> files;
    for(const auto& entry:std::filesystem::directory_iterator(directory))
      if(entry.is_regular_file() && entry.path().extension()==".formqr") files.push_back(entry.path());
    std::sort(files.begin(),files.end());
    if(files.empty()) throw std::runtime_error("No captured batches");
    std::mt19937 rng(84);
    std::vector<size_t> order(configs.size()); std::iota(order.begin(),order.end(),0);
    std::vector<std::vector<double>> aggregate(configs.size(),std::vector<double>(repeats));
    double worst_error=0.,checksum=0.;
    std::cout<<std::setprecision(12)<<"batch,first_rows,reduction_rows,block_threads,repeats,mean_us,p50_us,p95_us\n";
    auto emit=[&](const std::string& batch,size_t c,std::vector<double> v) {
      const double mean=std::accumulate(v.begin(),v.end(),0.)/v.size();
      std::sort(v.begin(),v.end());
      std::cout<<batch<<','<<configs[c][0]<<','<<configs[c][1]<<','<<configs[c][2]<<','<<repeats
               <<','<<mean<<','<<v[v.size()/2]<<','<<v[std::min(v.size()-1,size_t(.95*v.size()))]<<'\n';
    };
    for(const auto& file:files) {
      const auto input=form_benchmark::readBatch(file);
      for(auto& gpu:qr) {
        const auto roots=gpu->compute(input);
        for(size_t i=0;i<input.size();++i) {
          const Eigen::MatrixXd gram=input[i].transpose()*input[i];
          const double error=(roots[i].transpose()*roots[i]-gram).norm()/std::max(1.,gram.norm());
          if(!roots[i].allFinite() || !(error<=1e-9)) throw std::runtime_error("QR correctness failure");
          worst_error=std::max(worst_error,error);
        }
      }
      std::vector<std::vector<double>> times(configs.size(),std::vector<double>(repeats));
      for(int r=0;r<repeats;++r) {
        std::shuffle(order.begin(),order.end(),rng);
        for(size_t c:order) {
          const auto start=std::chrono::steady_clock::now();
          const auto roots=qr[c]->compute(input);
          times[c][r]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
          aggregate[c][r]+=times[c][r]; checksum+=roots.front()(0,0);
        }
      }
      for(size_t c=0;c<configs.size();++c) emit(file.filename().string(),c,times[c]);
    }
    for(size_t c=0;c<configs.size();++c) emit("ALL",c,aggregate[c]);
    std::cerr<<"Worst Gram error="<<worst_error<<" checksum="<<checksum<<'\n';
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
