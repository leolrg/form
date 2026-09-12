// Standalone experiment, no production integration. Build instructions in
// cuda_solve_experiment.md. All matrices and arithmetic are IEEE754 FP64.
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <Eigen/Dense>
#ifdef EIGEN_GPUCC
#error Compile this host-only cuSolver benchmark with ordinary C++, not NVCC
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
namespace {
double us(Clock::time_point t){return std::chrono::duration<double,std::micro>(Clock::now()-t).count();}
void cudaCheck(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
void solverCheck(cusolverStatus_t status){if(status!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("cuSolver setup status "+std::to_string(status));}
struct System {
  std::string name;Eigen::MatrixXd A;Eigen::VectorXd b;
  double normA=0,solution_tolerance=1e-6;
  bool expect_failure=false;
  void prepare(){normA=A.cwiseAbs().rowwise().sum().maxCoeff();}
};
System readSystem(const std::filesystem::path& path) {
  static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
  const uint16_t one=1;if(*reinterpret_cast<const uint8_t*>(&one)!=1)throw std::runtime_error("Little endian host required");
  std::ifstream in(path,std::ios::binary);char magic[8];uint32_t n=0;
  if(!in.read(magic,8)||std::memcmp(magic,"FORMSYS1",8)||!in.read(reinterpret_cast<char*>(&n),4)||!n||n>8192)
    throw std::runtime_error("Invalid FORMSYS1 header: "+path.string());
  if(std::filesystem::file_size(path)!=12+8*(uint64_t(n)*n+n))
    throw std::runtime_error("Truncated/trailing FORMSYS1 data: "+path.string());
  System s;s.name=path.filename().string();s.A.resize(n,n);s.b.resize(n);
  if(!in.read(reinterpret_cast<char*>(s.A.data()),uint64_t(n)*n*8)||!in.read(reinterpret_cast<char*>(s.b.data()),n*8))
    throw std::runtime_error("Cannot read system data");
  if(!s.A.allFinite()||!s.b.allFinite())throw std::runtime_error("Nonfinite system");
  s.prepare();
  if((s.A-s.A.transpose()).cwiseAbs().maxCoeff()>1e-12*std::max(1.,s.normA))
    throw std::runtime_error("Captured A is not symmetric");
  return s;
}
struct Result {
  Eigen::VectorXd x;double total=0,pack=0,h2d=0,factor_solve=0;
  int potrf_status=-1,potrs_status=-1,potrf_info=-999,potrs_info=-999,cpu_info=-1;
  bool success=false;
};
Result cpuSolve(const System& s) {
  Result r;auto start=Clock::now();
  Eigen::LLT<Eigen::MatrixXd,Eigen::Upper> factor(s.A);r.cpu_info=static_cast<int>(factor.info());
  if(factor.info()==Eigen::Success){r.x=factor.solve(s.b);r.success=r.x.allFinite();}
  r.total=us(start);return r;
}
class GpuWorkspace {
  int n_,lwork_=0;cudaStream_t stream_=nullptr;cusolverDnHandle_t handle_=nullptr;
  double *A_=nullptr,*b_=nullptr,*work_=nullptr,*hostA_=nullptr,*hostb_=nullptr;
  int *info_=nullptr,*hostInfo_=nullptr;
  cudaEvent_t begin_=nullptr,uploaded_=nullptr,solved_=nullptr;
public:
  explicit GpuWorkspace(int n):n_(n) {
    cudaCheck(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
    solverCheck(cusolverDnCreate(&handle_));solverCheck(cusolverDnSetStream(handle_,stream_));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&A_),size_t(n)*n*8));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&b_),n*8));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&info_),2*sizeof(int)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostA_),size_t(n)*n*8));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostb_),n*8));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostInfo_),2*sizeof(int)));
    solverCheck(cusolverDnDpotrf_bufferSize(handle_,CUBLAS_FILL_MODE_LOWER,n,A_,n,&lwork_));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&work_),std::max(1,lwork_)*sizeof(double)));
    cudaCheck(cudaEventCreate(&begin_));cudaCheck(cudaEventCreate(&uploaded_));cudaCheck(cudaEventCreate(&solved_));
  }
  ~GpuWorkspace(){
    if(stream_)cudaStreamSynchronize(stream_);
    if(begin_)cudaEventDestroy(begin_);if(uploaded_)cudaEventDestroy(uploaded_);if(solved_)cudaEventDestroy(solved_);
    if(handle_)cusolverDnDestroy(handle_);
    cudaFree(A_);cudaFree(b_);cudaFree(work_);cudaFree(info_);
    cudaFreeHost(hostA_);cudaFreeHost(hostb_);cudaFreeHost(hostInfo_);
    if(stream_)cudaStreamDestroy(stream_);
  }
  Result solve(const System& s) {
    Result r;auto start=Clock::now();
    std::memcpy(hostA_,s.A.data(),size_t(n_)*n_*8);std::memcpy(hostb_,s.b.data(),n_*8);
    r.pack=us(start);
    cudaCheck(cudaEventRecord(begin_,stream_));
    cudaCheck(cudaMemcpyAsync(A_,hostA_,size_t(n_)*n_*8,cudaMemcpyHostToDevice,stream_));
    cudaCheck(cudaMemcpyAsync(b_,hostb_,n_*8,cudaMemcpyHostToDevice,stream_));
    cudaCheck(cudaEventRecord(uploaded_,stream_));
    r.potrf_status=cusolverDnDpotrf(handle_,CUBLAS_FILL_MODE_LOWER,n_,A_,n_,work_,lwork_,info_);
    if(r.potrf_status==CUSOLVER_STATUS_SUCCESS)
      r.potrs_status=cusolverDnDpotrs(handle_,CUBLAS_FILL_MODE_LOWER,n_,1,A_,n_,b_,n_,info_+1);
    cudaCheck(cudaEventRecord(solved_,stream_));
    if(r.potrf_status==CUSOLVER_STATUS_SUCCESS)
      cudaCheck(cudaMemcpyAsync(hostInfo_,info_,sizeof(int),cudaMemcpyDeviceToHost,stream_));
    if(r.potrs_status==CUSOLVER_STATUS_SUCCESS) {
      cudaCheck(cudaMemcpyAsync(hostInfo_+1,info_+1,sizeof(int),cudaMemcpyDeviceToHost,stream_));
      cudaCheck(cudaMemcpyAsync(hostb_,b_,n_*8,cudaMemcpyDeviceToHost,stream_));
    }
    cudaCheck(cudaStreamSynchronize(stream_));
    if(r.potrf_status==CUSOLVER_STATUS_SUCCESS)r.potrf_info=hostInfo_[0];
    if(r.potrs_status==CUSOLVER_STATUS_SUCCESS)r.potrs_info=hostInfo_[1];
    r.success=r.potrf_status==0&&r.potrs_status==0&&r.potrf_info==0&&r.potrs_info==0;
    if(r.success){r.x=Eigen::Map<const Eigen::VectorXd>(hostb_,n_);r.success=r.x.allFinite();}
    float duration=0;cudaCheck(cudaEventElapsedTime(&duration,begin_,uploaded_));r.h2d=1000.*duration;
    cudaCheck(cudaEventElapsedTime(&duration,uploaded_,solved_));r.factor_solve=1000.*duration;
    r.total=us(start);return r;
  }
};
double backward(const System& s,const Eigen::VectorXd& x){
  const double denominator=s.normA*x.lpNorm<Eigen::Infinity>()+s.b.lpNorm<Eigen::Infinity>();
  return (s.A*x-s.b).lpNorm<Eigen::Infinity>()/std::max(denominator,std::numeric_limits<double>::min());
}
struct Validation {double difference=0,cpu_backward=0,gpu_backward=0;bool accepted=false;};
Validation validate(const System& s,const Result& cpu,const Result& gpu) {
  Validation v;if(!cpu.success||!gpu.success)return v;
  v.difference=(gpu.x-cpu.x).norm()/std::max(cpu.x.norm(),std::numeric_limits<double>::min());
  v.cpu_backward=backward(s,cpu.x);v.gpu_backward=backward(s,gpu.x);
  v.accepted=std::isfinite(v.difference)&&std::isfinite(v.cpu_backward)&&std::isfinite(v.gpu_backward)
    &&v.difference<=s.solution_tolerance&&v.cpu_backward<=1e-12&&v.gpu_backward<=1e-12;
  return v;
}
System synthetic(int n,double condition) {
  System s;s.name="synthetic-n"+std::to_string(n)+"-condition"+std::to_string(condition);
  Eigen::MatrixXd random=Eigen::MatrixXd::Random(n,n);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(random);Eigen::MatrixXd Q=qr.householderQ();
  Eigen::VectorXd spectrum(n),truth=Eigen::VectorXd::LinSpaced(n,-1.,1.);
  for(int i=0;i<n;++i)spectrum[i]=std::pow(condition,-double(i)/std::max(1,n-1));
  s.A=Q*spectrum.asDiagonal()*Q.transpose();s.A=(.5*(s.A+s.A.transpose())).eval();s.b=s.A*truth;
  // Ill-conditioned forward agreement is looser; backward error stays <=1e-12.
  if(condition>=1e10)s.solution_tolerance=.01;
  s.prepare();return s;
}
double mean(const std::vector<double>& a){return std::accumulate(a.begin(),a.end(),0.)/a.size();}
double percentile(std::vector<double> a,double q){std::sort(a.begin(),a.end());double p=(a.size()-1)*q;
  size_t i=static_cast<size_t>(p);return a[i]+(p-i)*(a[std::min(i+1,a.size()-1)]-a[i]);}
void printResult(const System& s,int repeat,const char* order,const Result& cpu,const Result& gpu,const Validation& v) {
  std::cout<<"sample,"<<s.name<<','<<s.A.rows()<<','<<repeat<<','<<order<<','<<cpu.total<<','<<gpu.total<<','<<gpu.pack<<','<<gpu.h2d<<','<<gpu.factor_solve
    <<','<<cpu.cpu_info<<','<<gpu.potrf_status<<','<<gpu.potrs_status<<','<<gpu.potrf_info<<','<<gpu.potrs_info<<',';
  if(cpu.success&&gpu.success)std::cout<<v.difference<<','<<v.cpu_backward<<','<<v.gpu_backward;
  else std::cout<<",,";
  std::cout<<','<<s.solution_tolerance<<','<<(v.accepted?"pass":"fail")<<",,,,\n";
}
void readerSelfTest(){
  const auto path=std::filesystem::temp_directory_path()/"form-solve-reader-test.formsys";
  auto write=[&](uint32_t n,size_t doubles,const char* magic="FORMSYS1"){
    std::ofstream out(path,std::ios::binary);out.write(magic,8);out.write(reinterpret_cast<const char*>(&n),4);
    for(size_t i=0;i<doubles;++i){double value=1.;out.write(reinterpret_cast<const char*>(&value),8);}
  };
  write(1,2);if(readSystem(path).A(0,0)!=1.)throw std::runtime_error("Reader roundtrip failed");
  for(int mode=0;mode<5;++mode){
    if(mode==0)write(0,0);if(mode==1)write(9000,0);if(mode==2)write(2,1);
    if(mode==3)write(1,3);if(mode==4)write(1,2,"BADMAGIC");
    bool rejected=false;try{readSystem(path);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("Reader accepted malformed input");
  }
  std::filesystem::remove(path);std::cerr<<"FORMSYS1 reader self-tests passed\n";
}
}
int main(int argc,char** argv){
  try {
    std::filesystem::path directory;int repeats=30;bool self_test_only=false,reader_only=false;
    for(int i=1;i<argc;++i){std::string arg=argv[i];
      if(arg=="--self-test-only"){self_test_only=true;continue;}if(arg=="--reader-only"){reader_only=true;continue;}
      if(i+1==argc)throw std::runtime_error("Missing argument");
      if(arg=="--input-dir")directory=argv[++i];else if(arg=="--repeat")repeats=std::stoi(argv[++i]);else throw std::runtime_error("Unknown argument");}
    if(repeats<1)throw std::runtime_error("--repeat must be positive");
    Eigen::setNbThreads(1);std::srand(42);std::cerr<<"CPU triangle=Upper; GPU triangle=Lower; CPU compiler="<<__VERSION__<<", Eigen SIMD="<<Eigen::SimdInstructionSetsInUse()<<", Eigen threads="<<Eigen::nbThreads()<<"\n";readerSelfTest();if(reader_only)return 0;
    std::vector<System> systems{synthetic(8,100),synthetic(33,1e10),synthetic(128,1e12)};
    System bad;bad.name="non-SPD-rejection";bad.A=Eigen::MatrixXd::Identity(8,8);bad.A(0,0)=-1.;bad.b=Eigen::VectorXd::Ones(8);bad.expect_failure=true;bad.prepare();systems.push_back(bad);
    if(!self_test_only){
      if(directory.empty())throw std::runtime_error("Use --input-dir DIR or --self-test-only");
      std::vector<std::filesystem::path> files;for(const auto& e:std::filesystem::recursive_directory_iterator(directory))if(e.path().extension()==".formsys")files.push_back(e.path());
      std::sort(files.begin(),files.end());if(files.empty())throw std::runtime_error("No captured systems found");
      for(const auto& file:files){auto system=readSystem(file);system.name=file.lexically_relative(directory).generic_string();systems.push_back(std::move(system));}
    }
    std::cout<<std::setprecision(12)<<"scope,case,N,repeat,order,cpu_total_us,gpu_total_us,host_pack_us,h2d_us,factor_solve_device_us,cpu_info,potrf_status,potrs_status,potrf_info,potrs_info,relative_solution_difference,cpu_backward_error,gpu_backward_error,solution_tolerance,status,cpu_p50_us,cpu_p95_us,gpu_p50_us,gpu_p95_us\n";
    std::map<int,std::unique_ptr<GpuWorkspace>> workspaces;std::mt19937 generator(17);bool all_valid=true;
    for(const auto& s:systems){
      const int n=s.A.rows();auto& workspace=workspaces[n];if(!workspace)workspace=std::make_unique<GpuWorkspace>(n);
      auto warm_cpu=cpuSolve(s),warm_gpu=workspace->solve(s);auto warm_validation=validate(s,warm_cpu,warm_gpu);
      if(s.expect_failure){
        const bool rejected=!warm_cpu.success&&!warm_gpu.success&&warm_gpu.potrf_info>0;
        std::cerr<<s.name<<": CPU info="<<warm_cpu.cpu_info<<", cuSolver status="<<warm_gpu.potrf_status<<", potrf info="<<warm_gpu.potrf_info<<", rejection "<<(rejected?"PASS":"FAIL")<<'\n';
        if(!rejected)all_valid=false;continue;
      }
      if(!warm_validation.accepted){printResult(s,-1,"warmup",warm_cpu,warm_gpu,warm_validation);all_valid=false;continue;}
      std::vector<double> cpu_times,gpu_times,pack_times,h2d_times,device_times;int failures=0;bool first=generator()&1;
      Validation worst;
      for(int r=0;r<repeats;++r){
        Result cpu,gpu;const bool gpu_first=first^(r%2);
        if(gpu_first){gpu=workspace->solve(s);cpu=cpuSolve(s);}else{cpu=cpuSolve(s);gpu=workspace->solve(s);}
        const auto v=validate(s,cpu,gpu);printResult(s,r,gpu_first?"GPU-first":"CPU-first",cpu,gpu,v);
        if(!v.accepted){++failures;all_valid=false;continue;}
        cpu_times.push_back(cpu.total);gpu_times.push_back(gpu.total);pack_times.push_back(gpu.pack);h2d_times.push_back(gpu.h2d);device_times.push_back(gpu.factor_solve);
        worst.difference=std::max(worst.difference,v.difference);worst.cpu_backward=std::max(worst.cpu_backward,v.cpu_backward);worst.gpu_backward=std::max(worst.gpu_backward,v.gpu_backward);
      }
      if(failures){std::cerr<<s.name<<": "<<failures<<" invalid samples; no timing summary emitted\n";continue;}
      std::cout<<"summary,"<<s.name<<','<<n<<','<<repeats<<",alternating,"<<mean(cpu_times)<<','<<mean(gpu_times)<<','<<mean(pack_times)<<','<<mean(h2d_times)<<','<<mean(device_times)
        <<",0,0,0,0,0,"<<worst.difference<<','<<worst.cpu_backward<<','<<worst.gpu_backward<<','<<s.solution_tolerance<<",pass,"<<percentile(cpu_times,.5)<<','<<percentile(cpu_times,.95)<<','<<percentile(gpu_times,.5)<<','<<percentile(gpu_times,.95)<<'\n';
      std::cerr<<s.name<<": N="<<n<<", CPU="<<mean(cpu_times)<<" us GPU="<<mean(gpu_times)<<" us, relative difference="<<worst.difference<<" backward="<<worst.gpu_backward<<'\n';
    }
    if(!all_valid)return 2;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
