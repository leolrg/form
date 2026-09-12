// Standalone experiment; not used by the production estimator.
#include "direct_hessian.hpp"
#include "qr_capture.hpp"
#include "form/feature/cuda_qr.hpp"
#include "form/feature/summary.hpp"
#include <Eigen/QR>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
using Clock=std::chrono::steady_clock;
using Hessian=form::FeatureSummary::AugmentedHessian;
using Matrices=std::vector<Eigen::MatrixXd>;
namespace {
Eigen::Matrix3d skew(const Eigen::Vector3d& p) {Eigen::Matrix3d s;s<<0,-p.z(),p.y(),p.z(),0,-p.x(),-p.y(),p.x(),0;return s;}
Eigen::Vector3d crossParts(const Eigen::Matrix3d& m) {return {m(1,2)-m(2,1),m(2,0)-m(0,2),m(0,1)-m(1,0)};}
struct Pair {gtsam::Pose3 i,j;};
Pair pairAt(int index,int evaluation,bool near=false) {
  const double x=.003*index;
  const auto i=gtsam::Pose3(gtsam::Rot3::RzRyRx(.31+x,-.23,.41),gtsam::Point3(8.+x,-3.,2.));
  const double d=near?1e-8:1.;
  const auto j=gtsam::Pose3(gtsam::Rot3::RzRyRx(.31+x+d*(.17+.01*evaluation),-.23+d*.08,.41-d*.12),
                           gtsam::Point3(8.+x+d*(.4+.03*evaluation),-3.+d*.2,2.-d*.1));
  return {i,j};
}
std::vector<direct_experiment::PoseData> packPoses(const std::vector<Pair>& pairs) {
  std::vector<direct_experiment::PoseData> out(pairs.size());
  for(size_t k=0;k<pairs.size();++k) {
    const Eigen::Matrix3d Ri=pairs[k].i.rotation().matrix(),Rj=pairs[k].j.rotation().matrix();
    const Eigen::Matrix3d R=Ri.transpose()*Rj,D=Ri.transpose()*(Rj-Ri);
    const Eigen::Vector3d dt=pairs[k].j.translation()-pairs[k].i.translation(),t=Ri.transpose()*dt;
    for(int a=0;a<3;++a) {out[k].dt[a]=dt[a];out[k].t[a]=t[a];
      for(int b=0;b<3;++b) {const int n=3*a+b;out[k].Ri[n]=Ri(a,b);out[k].Rj[n]=Rj(a,b);out[k].R[n]=R(a,b);out[k].D[n]=D(a,b);}}
  }
  return out;
}
Hessian rawHessian(const Eigen::MatrixXd& features,const Pair& poses) {
  const Eigen::Matrix3d Ri=poses.i.rotation().matrix(),Rj=poses.j.rotation().matrix();
  const Eigen::Matrix3d R=Ri.transpose()*Rj,D=Ri.transpose()*(Rj-Ri);
  const Eigen::Vector3d dt=poses.j.translation()-poses.i.translation(),t=Ri.transpose()*dt;
  Hessian result=Hessian::Zero();
  for(Eigen::Index k=0;k<features.rows();++k) {
    if(features.cols()==7) {
      const double w=features(k,0);
      const Eigen::Vector3d p=features.block<1,3>(k,1).transpose(),d=features.block<1,3>(k,4).transpose();
      Eigen::Matrix<double,3,13> a;
      a.block<3,3>(0,0)=Ri*skew(p);a.block<3,3>(0,3)=-w*Ri;
      a.block<3,3>(0,6)=-Rj*skew(p+d);a.block<3,3>(0,9)=w*Rj;
      a.col(12)=-((Rj-Ri)*p+Rj*d+w*dt);
      result.noalias()+=a.transpose()*a;
    } else {
      Eigen::Matrix3d tensor;
      for(int a=0;a<3;++a) for(int b=0;b<3;++b) tensor(a,b)=features(k,3*a+b);
      const Eigen::Vector3d normal=features.block<1,3>(k,9).transpose();
      Eigen::Matrix<double,1,13> a;
      a.block<1,3>(0,0)=(crossParts(tensor*R.transpose())+normal.cross(t)).transpose();
      a.block<1,3>(0,3)=-normal.transpose();a.block<1,3>(0,6)=crossParts(tensor.transpose()*R).transpose();
      a.block<1,3>(0,9)=normal.transpose()*R;
      a(12)=-(tensor.cwiseProduct(D).sum()+normal.dot(t)+features(k,12));
      result.noalias()+=a.transpose()*a;
    }
  }
  return result;
}
std::vector<form::FeatureSummary> summaries(const Matrices& roots) {
  std::vector<form::FeatureSummary> out;out.reserve(roots.size());
  for(const auto& root:roots) {
    Eigen::Matrix<double,7,7> points=Eigen::Matrix<double,7,7>::Zero();
    Eigen::Matrix<double,13,13> planes=Eigen::Matrix<double,13,13>::Zero();
    if(root.cols()==7) points=root;else planes=root;
    out.emplace_back(planes,points);
  }
  return out;
}
std::vector<form::FeatureSummary> cpuSummaries(const Matrices& inputs) {
  Matrices roots;
  for(const auto& matrix:inputs) {
    Eigen::MatrixXd root=Eigen::MatrixXd::Zero(matrix.cols(),matrix.cols());
    if(matrix.rows()) {Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrix);
      root.topRows(std::min(matrix.rows(),matrix.cols()))=qr.matrixQR().topRows(std::min(matrix.rows(),matrix.cols())).triangularView<Eigen::Upper>();}
    roots.push_back(root);
  }
  return summaries(roots);
}
std::vector<direct_experiment::MatrixInput> views(const Matrices& input) {
  std::vector<direct_experiment::MatrixInput> result;
  for(const auto& a:input) result.push_back({a.data(),static_cast<int>(a.rows()),static_cast<int>(a.cols())});return result;
}
std::vector<Pair> pairsFor(size_t count,int evaluation,bool near=false) {
  std::vector<Pair> out;for(size_t i=0;i<count;++i) out.push_back(pairAt(i,evaluation,near));return out;
}
struct Errors {double h=0,g=0,cost=0,summary_h=0,summary_g=0,summary_cost=0;};
void compare(const Hessian& actual,const Hessian& expected,double& h,double& g,double& cost) {
  if(!actual.allFinite() || !expected.allFinite()) throw std::runtime_error("Nonfinite Hessian validation");
  h=std::max(h,(actual.topLeftCorner<12,12>()-expected.topLeftCorner<12,12>()).norm()/std::max(1.,expected.topLeftCorner<12,12>().norm()));
  g=std::max(g,(actual.block<12,1>(0,12)-expected.block<12,1>(0,12)).norm()/std::max(1.,expected.block<12,1>(0,12).norm()));
  cost=std::max(cost,std::abs(actual(12,12)-expected(12,12))/std::max(1.,std::abs(expected(12,12))));
}
Errors validate(direct_experiment::Batch& gpu,const Matrices& input) {
  const auto reference=cpuSummaries(input);Errors errors;gpu.upload(views(input));
  for(bool near:{false,true}) {
    const auto pairs=pairsFor(input.size(),0,near);const auto poses=packPoses(pairs);
    const auto h=gpu.evaluate(poses),cost=gpu.evaluate(poses,true);
    for(size_t i=0;i<input.size();++i) {
      const auto raw=rawHessian(input[i],pairs[i]);
      const Hessian actual=Eigen::Map<const Eigen::Matrix<double,13,13,Eigen::RowMajor>>(h.data()+i*169);
      compare(actual,raw,errors.h,errors.g,errors.cost);
      compare(reference[i].augmentedHessian(pairs[i].i,pairs[i].j),raw,errors.summary_h,errors.summary_g,errors.summary_cost);
      if(!std::isfinite(cost[i])) throw std::runtime_error("Nonfinite cost validation");
      errors.cost=std::max(errors.cost,std::abs(cost[i]-raw(12,12))/std::max(1.,std::abs(raw(12,12))));
      errors.summary_cost=std::max(errors.summary_cost,std::abs(reference[i].squaredError(pairs[i].i,pairs[i].j)-raw(12,12))/std::max(1.,std::abs(raw(12,12))));
    }
  }
  if(std::max({errors.h,errors.g,errors.cost,errors.summary_h,errors.summary_g,errors.summary_cost})>1e-9)
    throw std::runtime_error("Direct Hessian/gradient/cost or summary disagrees with raw CPU by >1e-9");
  return errors;
}
double elapsed(Clock::time_point start) {return std::chrono::duration<double,std::micro>(Clock::now()-start).count();}
double mean(const std::vector<double>& values){return std::accumulate(values.begin(),values.end(),0.)/values.size();}
volatile double checksum=0;
}
int main(int argc,char** argv) {
  try {
    std::filesystem::path directory="benchmarks/results/qr-stairs200";int repeats=10,qr_first=32,qr_reduction=32,qr_threads=128;bool validate_only=false;
    for(int i=1;i<argc;++i) {std::string arg=argv[i];if(arg=="--validate-only"){validate_only=true;continue;}
      if(i+1==argc)throw std::runtime_error("Missing argument");
      if(arg=="--input-dir")directory=argv[++i];else if(arg=="--repeat")repeats=std::stoi(argv[++i]);
      else if(arg=="--qr-first-rows")qr_first=std::stoi(argv[++i]);
      else if(arg=="--qr-reduction-rows")qr_reduction=std::stoi(argv[++i]);
      else if(arg=="--qr-threads")qr_threads=std::stoi(argv[++i]);else throw std::runtime_error("Unknown argument");}
    if(repeats<1)throw std::runtime_error("Invalid repeat count");
    direct_experiment::Batch direct;form::BatchedCudaQr qr(true,qr_first,qr_reduction,qr_threads);
    Matrices edge;
    for(int cols:{7,13})for(int rows:{0,1,31,32,33,129}) {
      Eigen::MatrixXd m=Eigen::MatrixXd::Random(rows,cols);
      if(rows>1)m.row(rows-1)=m.row(0);if(cols==7 && rows)m.col(0).setOnes();edge.push_back(m);
    }
    auto error=validate(direct,edge);
    std::cerr<<"Edge validation: H="<<error.h<<" gradient="<<error.g<<" cost="<<error.cost<<'\n';
    std::vector<std::filesystem::path> files;for(const auto& e:std::filesystem::directory_iterator(directory))
      if(e.path().extension()==".formqr")files.push_back(e.path());std::sort(files.begin(),files.end());
    if(files.empty())throw std::runtime_error("No captured input");
    std::cout<<std::setprecision(12)<<"batch,kind,evaluations,matrices,feature_values,repeats,qr_first_rows,qr_reduction_rows,qr_threads,direct_upload_us,direct_resident_us,direct_total_us,qr_summary_total_us,summary_resident_us,h_error,gradient_error,cost_error,summary_h_error,summary_gradient_error,summary_cost_error\n";
    for(const auto& file:files) {
      const auto input=form_benchmark::readBatch(file);const auto input_views=views(input);
      const auto errors=validate(direct,input);size_t values=0;for(const auto& a:input)values+=a.size();
      std::cerr<<file.filename()<<": H="<<errors.h<<" gradient="<<errors.g<<" cost="<<errors.cost<<'\n';
      if(validate_only)continue;
      const auto cached=summaries(qr.compute(input));
      for(bool cost_only:{false,true})for(int evaluations:{1,2,4}) {
        std::vector<std::vector<Pair>> pairs;
        for(int k=0;k<evaluations;++k)pairs.push_back(pairsFor(input.size(),k));
        std::vector<double> uploads,resident,total,baseline,summary_only;
        auto cpuEval=[&](const auto& summaries) {
          for(int k=0;k<evaluations;++k)for(size_t i=0;i<input.size();++i) {
            if(cost_only)checksum+=summaries[i].squaredError(pairs[k][i].i,pairs[k][i].j);
            else checksum+=summaries[i].augmentedHessian(pairs[k][i].i,pairs[k][i].j)(12,12);
          }
        };
        auto runDirect=[&]() {
          auto begin=Clock::now();direct.upload(input_views);uploads.push_back(elapsed(begin));
          auto eval_start=Clock::now();
          for(int k=0;k<evaluations;++k){auto out=direct.evaluate(packPoses(pairs[k]),cost_only);checksum+=out.back();}
          resident.push_back(elapsed(eval_start));total.push_back(elapsed(begin));
        };
        auto runQr=[&]() {auto begin=Clock::now();const auto ready=summaries(qr.compute(input));cpuEval(ready);baseline.push_back(elapsed(begin));};
        for(int r=0;r<repeats;++r) {
          if(r%2){runQr();runDirect();}else{runDirect();runQr();}
          auto begin=Clock::now();cpuEval(cached);summary_only.push_back(elapsed(begin));
        }
        std::cout<<file.filename().string()<<','<<(cost_only?"trial_cost":"hessian")<<','<<evaluations<<','<<input.size()<<','<<values<<','<<repeats<<','<<qr_first<<','<<qr_reduction<<','<<qr_threads<<','
                 <<mean(uploads)<<','<<mean(resident)<<','<<mean(total)<<','<<mean(baseline)<<','<<mean(summary_only)<<','
                 <<errors.h<<','<<errors.g<<','<<errors.cost<<','<<errors.summary_h<<','<<errors.summary_g<<','<<errors.summary_cost<<'\n';
      }
    }
    std::cerr<<"checksum="<<checksum<<"; timings include pose transforms, pose upload, batch kernels and synchronized result copy; no per-factor sync.\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
