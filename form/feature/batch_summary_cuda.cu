#include <optional>
#include "form/optimization/diagnostics.hpp"
// MIT License; see factor.hpp for copyright and license text.
#include "form/feature/batch_summary_cuda.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
namespace form {
namespace {
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
template<class T,bool Host=false> struct Buffer {
  T* data=nullptr;size_t capacity=0;
  explicit Buffer(size_t count) {reserve(count);}
  void reserve(size_t count) {
    count=std::max<size_t>(1,count);if(count<=capacity)return;
    if(data) {if constexpr(Host)check(cudaFreeHost(data));else check(cudaFree(data));data=nullptr;capacity=0;}
    if constexpr(Host)check(cudaMallocHost(reinterpret_cast<void**>(&data),count*sizeof(T)));
    else check(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T)));
    capacity=count;
  }
  ~Buffer() {if constexpr(Host) cudaFreeHost(data);else cudaFree(data);}
  Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
__device__ double skew(const double* p,int r,int c) {
  if(r==c)return 0.;if(r==0)return c==1?-p[2]:p[1];if(r==1)return c==0?p[2]:-p[0];return c==0?-p[1]:p[0];
}
// 34 residual rows per edge: 7 point roots x 3 components + 13 plane roots.
// 128 threads give four warps for the 169 Hessian entries, with 3.5 KiB shared storage.
template<bool Cost>
__global__ void evaluateEdges(const BatchRoot* roots,const BatchPose* poses,double* partial) {
  const int e=blockIdx.x,tid=threadIdx.x;const auto& root=roots[e];
  const auto& pi=poses[root.i];const auto& pj=poses[root.j];
  __shared__ double a[34*13];
  __shared__ double rel[9],delta[9],t[3],dt[3];
  if(tid<3) {dt[tid]=pj.translation[tid]-pi.translation[tid];double sum=0.;for(int k=0;k<3;++k) sum+=pi.rotation[k*3+tid]*(pj.translation[k]-pi.translation[k]);t[tid]=sum;}
  if(tid<9) {int r=tid/3,c=tid%3;double s=0.,d=0.;for(int k=0;k<3;++k) {s+=pi.rotation[k*3+r]*pj.rotation[k*3+c];d+=pi.rotation[k*3+r]*(pj.rotation[k*3+c]-pi.rotation[k*3+c]);}rel[tid]=s;delta[tid]=d;}
  __syncthreads();
  if(tid<34) {
    double* out=a+13*tid;
    double residual=0.;
    if(tid<21) {
      const int row=tid/3,c=tid%3;double p[3],q[3],d[3];
      for(int k=0;k<3;++k) {p[k]=root.point[row+7*(1+k)];d[k]=root.point[row+7*(4+k)];q[k]=p[k]+d[k];}
      double rotation_term=0.,difference_term=0.;
      for(int k=0;k<3;++k) {rotation_term+=(pj.rotation[3*c+k]-pi.rotation[3*c+k])*p[k];difference_term+=pj.rotation[3*c+k]*d[k];}
      double weight=root.point[row];residual=rotation_term+difference_term+weight*dt[c];
      if constexpr(!Cost) for(int j=0;j<3;++j) {
        double ji=0.,jj=0.;for(int k=0;k<3;++k) {ji+=pi.rotation[3*c+k]*skew(p,k,j);jj-=pj.rotation[3*c+k]*skew(q,k,j);}
        out[j]=ji;out[3+j]=-weight*pi.rotation[3*c+j];out[6+j]=jj;out[9+j]=weight*pj.rotation[3*c+j];
      }
    } else {
      const int row=tid-21;double q[13];for(int k=0;k<13;++k) q[k]=root.plane[row+13*k];
      for(int k=0;k<9;++k) residual+=q[k]*delta[k];
      double normal_term=0.;for(int k=0;k<3;++k) normal_term+=q[9+k]*t[k];residual+=normal_term;residual+=q[12];
      if constexpr(!Cost) for(int axis=0;axis<3;++axis) {
        int x=(axis+1)%3,y=(axis+2)%3;
        double ji=q[9+x]*t[y]-q[9+y]*t[x],jj=0.,jt=0.;
        for(int k=0;k<3;++k) {ji+=q[3*x+k]*rel[3*y+k]-q[3*y+k]*rel[3*x+k];jj+=q[3*k+x]*rel[3*k+y]-q[3*k+y]*rel[3*k+x];jt+=q[9+k]*rel[3*k+axis];}
        out[axis]=ji;out[3+axis]=-q[9+axis];out[6+axis]=jj;out[9+axis]=jt;
      }
    }
    out[12]=-residual;
  }
  __syncthreads();
  if constexpr(Cost) {
    if(tid==0) {double sum=0.;for(int r=0;r<34;++r) sum+=a[r*13+12]*a[r*13+12];partial[e]=root.weight*sum;}
  } else {
    for(int entry=tid;entry<169;entry+=blockDim.x) {
      int r=entry%13,c=entry/13;double sum=0.;for(int k=0;k<34;++k) sum+=a[k*13+r]*a[k*13+c];partial[e*169+entry]=root.weight*sum;
    }
  }
}
__global__ void assemble(const double* partial,const int* offsets,const int* indices,double* output,int count) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
  double sum=0.;for(int p=offsets[k];p<offsets[k+1];++p) sum+=partial[indices[p]];output[k]=sum;
}
__global__ void sumCosts(const double* partial,double* output,int edges) {
  // A single deterministic sum; feature graphs here contain hundreds of edges.
  if(threadIdx.x==0) {double sum=0.;for(int k=0;k<edges;++k)sum+=partial[k];output[0]=sum;}
}
void solverCheck(cusolverStatus_t status) {
  if(status!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("Resident cuSolver failure: "+std::to_string(status));
}
struct FrozenDescriptor {int matrix, n, delta; bool anchor;};
struct ResidentData {
  int dimension=0, frozen_count=0, delta_count=0, auxiliary_count=0, workspace_count=0;
  bool ready=false;
  cusolverDnHandle_t solver=nullptr;
  Buffer<FrozenDescriptor> frozen{0};
  Buffer<double> matrices{0},constant{0},displacements{0},shifted{0},products{0},costs{0},auxiliary{0};
  Buffer<int> rhs_offsets{0},rhs_indices{0},aux_offsets{0},aux_indices{0};
  Buffer<double> damped{0},solution{0},workspace{0},scalars{3};
  Buffer<int> info{2};
  Buffer<double,true> host_displacements{0},host_auxiliary{0},host_solution{0},host_scalars{3};
  Buffer<int,true> host_info{2};
  explicit ResidentData(cudaStream_t stream) {solverCheck(cusolverDnCreate(&solver));solverCheck(cusolverDnSetStream(solver,stream));}
  ~ResidentData(){if(solver)cusolverDnDestroy(solver);}
};
// Frozen factors retain their original tangent-space Hessian. Only b and f shift.
__global__ void shiftFrozen(const FrozenDescriptor* terms,const double* matrices,const double* delta,
                            double* products,double* shifted,double* costs) {
  const auto term=terms[blockIdx.x];const int n=term.n;const double* h=matrices+term.matrix;
  for(int r=threadIdx.x;r<n;r+=blockDim.x) {
    double gd=0.;if(term.anchor)for(int c=0;c<n;++c)gd+=h[r+(n+1)*c]*delta[term.delta+c];
    products[term.delta+r]=gd;shifted[term.delta+r]=h[r+(n+1)*n]-gd;
  }
  __syncthreads();
  if(threadIdx.x==0) {
    double f=h[n+(n+1)*n];
    if(term.anchor)for(int r=0;r<n;++r)f+=delta[term.delta+r]*(products[term.delta+r]-2*h[r+(n+1)*n]);
    costs[blockIdx.x]=f;
  }
}
__global__ void mergeResident(double* model,const double* constant,int n,const double* shifted,const double* costs,int terms,
                              const int* rhs_offsets,const int* rhs_indices,const double* auxiliary,const int* aux_offsets,const int* aux_indices) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=(n+1)*(n+1))return;
  int r=k%(n+1),c=k/(n+1);double sum=model[k]+constant[k];
  if(r==n && c==n)for(int t=0;t<terms;++t)sum+=costs[t];
  else if(r==n || c==n) {int row=r==n?c:r;for(int p=rhs_offsets[row];p<rhs_offsets[row+1];++p)sum+=shifted[rhs_indices[p]];}
  for(int p=aux_offsets[k];p<aux_offsets[k+1];++p)sum+=auxiliary[aux_indices[p]];
  model[k]=sum;
}
__global__ void residentCost(const double* feature,int edges,const FrozenDescriptor* terms,const double* costs,int count,double* result) {
  if(threadIdx.x==0) {double sum=0.;for(int e=0;e<edges;++e)sum+=feature[e];for(int t=0;t<count;++t)if(terms[t].anchor)sum+=costs[t];result[2]=.5*sum;}
}
__global__ void prepareDamped(const double* model,int n,double lambda,bool diagonal,double minimum,double maximum,double* matrix,double* rhs) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  if(k<n*n) {int r=k%n,c=k/n;double v=model[r+(n+1)*c];
    if(r==c){double a=1./(1./sqrt(lambda));if(diagonal)a*=sqrt(fmin(maximum,fmax(minimum,v)));v+=a*a;}matrix[k]=v;
  }
  if(k<n)rhs[k]=model[k+(n+1)*n];
}
__global__ void linearErrors(const double* model,int n,const double* delta,double* scalars) {
  // Row products run in parallel; final accumulation has a fixed order.
  __shared__ double sums[256];double sum=0.;
  for(int r=threadIdx.x;r<n;r+=blockDim.x) {double gd=0.;for(int c=0;c<n;++c)gd+=model[r+(n+1)*c]*delta[c];sum+=delta[r]*(gd-2*model[r+(n+1)*n]);}
  sums[threadIdx.x]=sum;__syncthreads();
  if(threadIdx.x==0) {double change=0.;for(int k=0;k<blockDim.x;++k)change+=sums[k];scalars[0]=.5*model[n+(n+1)*n];scalars[1]=scalars[0]+.5*change;}
}
template<class T> void upload(Buffer<T>& buffer,const std::vector<T>& values,cudaStream_t stream) {
  buffer.reserve(values.size());if(!values.empty())check(cudaMemcpyAsync(buffer.data,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
}
void buildCsr(const std::vector<std::pair<int,int>>& entries,int size,std::vector<int>& offsets,std::vector<int>& indices) {
  offsets.assign(size+1,0);for(auto item:entries)++offsets[item.first+1];
  for(int k=1;k<=size;++k)offsets[k]+=offsets[k-1];indices.resize(entries.size());auto cursor=offsets;
  for(auto item:entries)indices[cursor[item.first]++]=item.second;
}
}
struct CudaSummaryBatch::Impl {
  int poses,edges,entries;cudaStream_t stream=nullptr;
  std::unique_ptr<ResidentData> resident;
  Buffer<BatchRoot> roots;Buffer<BatchPose> pose_data;Buffer<int> offsets,indices;
  Buffer<double> partial,output;Buffer<BatchPose,true> host_poses;Buffer<double,true> host_output;
  Impl(int n,size_t e,size_t os,size_t is):poses(n),edges(e),entries((6*n+1)*(6*n+1)),roots(e),pose_data(n),offsets(os),indices(is),partial(e*169),output(entries),host_poses(n),host_output(entries) {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
  ~Impl(){if(stream){cudaStreamSynchronize(stream);resident.reset();cudaStreamDestroy(stream);}}
};
CudaSummaryBatch::CudaSummaryBatch(int poses,const std::vector<BatchRoot>& roots,const std::vector<int>& offsets,const std::vector<int>& indices):impl_(std::make_unique<Impl>(poses,roots.size(),offsets.size(),indices.size())) {
  reset(poses,roots,offsets,indices);
}
void CudaSummaryBatch::reset(int poses,const std::vector<BatchRoot>& roots,const std::vector<int>& offsets,const std::vector<int>& indices) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::cuda_batch_reset);
  auto& s=*impl_;check(cudaStreamSynchronize(s.stream));if(s.resident){s.resident->ready=false;s.resident->dimension=0;}s.poses=poses;s.edges=roots.size();s.entries=(6*poses+1)*(6*poses+1);
  s.roots.reserve(roots.size());s.pose_data.reserve(poses);s.offsets.reserve(offsets.size());s.indices.reserve(indices.size());
  s.partial.reserve(roots.size()*169);s.output.reserve(s.entries);s.host_poses.reserve(poses);s.host_output.reserve(s.entries);
  if(!roots.empty())check(cudaMemcpyAsync(s.roots.data,roots.data(),roots.size()*sizeof(BatchRoot),cudaMemcpyHostToDevice,s.stream));
  check(cudaMemcpyAsync(s.offsets.data,offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice,s.stream));
  if(!indices.empty())check(cudaMemcpyAsync(s.indices.data,indices.data(),indices.size()*sizeof(int),cudaMemcpyHostToDevice,s.stream));
  check(cudaStreamSynchronize(s.stream));
}
CudaSummaryBatch::~CudaSummaryBatch()=default;
void CudaSummaryBatch::evaluate(const std::vector<BatchPose>& poses,double* output,bool cost) {
  auto& s=*impl_;check(cudaStreamSynchronize(s.stream));if(s.resident)s.resident->ready=false;std::copy(poses.begin(),poses.end(),s.host_poses.data);
  check(cudaMemcpyAsync(s.pose_data.data,s.host_poses.data,poses.size()*sizeof(BatchPose),cudaMemcpyHostToDevice,s.stream));
  if(s.edges) {
    if(cost)evaluateEdges<true><<<s.edges,128,0,s.stream>>>(s.roots.data,s.pose_data.data,s.partial.data);
    else evaluateEdges<false><<<s.edges,128,0,s.stream>>>(s.roots.data,s.pose_data.data,s.partial.data);
    check(cudaGetLastError());
  }
  if(cost)sumCosts<<<1,32,0,s.stream>>>(s.partial.data,s.output.data,s.edges);
  else assemble<<<(s.entries+255)/256,256,0,s.stream>>>(s.partial.data,s.offsets.data,s.indices.data,s.output.data,s.entries);
  check(cudaGetLastError());const size_t count=cost?1:s.entries;
  check(cudaMemcpyAsync(s.host_output.data,s.output.data,count*sizeof(double),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));std::copy_n(s.host_output.data,count,output);
}
void CudaSummaryBatch::configureResident(const std::vector<FrozenSystem>& frozen,const std::vector<std::vector<int>>& auxiliary_poses) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::cuda_configure);
  auto& s=*impl_;check(cudaStreamSynchronize(s.stream));
  if(!s.resident)s.resident=std::make_unique<ResidentData>(s.stream);
  auto& w=*s.resident;w.ready=false;w.dimension=0;
  std::optional<diagnostics::Scope> phase; phase.emplace(diagnostics::Stage::cuda_configure_host);
  const int n=6*s.poses,na=n+1;
  std::vector<double> matrices,constant(s.entries,0.);
  std::vector<FrozenDescriptor> terms;
  std::vector<std::pair<int,int>> rhs_entries,aux_entries;
  int displacement_count=0,auxiliary_count=0;
  auto mapping=[&](const std::vector<int>& poses) {
    std::vector<int> result;std::vector<bool> seen(s.poses,false);
    for(int p:poses) {
      if(p<0 || p>=s.poses || seen[p])throw std::invalid_argument("Invalid resident factor pose indices");
      seen[p]=true;for(int k=0;k<6;++k)result.push_back(6*p+k);
    }
    result.push_back(n);return result;
  };
  for(const auto& term:frozen) {
    auto map=mapping(term.pose_indices);int local=static_cast<int>(map.size())-1;
    if(term.augmented.size()!=map.size()*map.size())throw std::invalid_argument("Invalid frozen augmented matrix size");
    for(double value:term.augmented)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite frozen matrix");
    terms.push_back({static_cast<int>(matrices.size()),local,displacement_count,term.has_anchor});
    matrices.insert(matrices.end(),term.augmented.begin(),term.augmented.end());
    for(int r=0;r<local;++r) {
      rhs_entries.emplace_back(map[r],displacement_count+r);
      for(int c=0;c<local;++c)constant[map[r]+na*map[c]]+=term.augmented[r+(local+1)*c];
    }
    displacement_count+=local;
  }
  for(const auto& poses:auxiliary_poses) {
    auto map=mapping(poses);int local=map.size();
    for(int c=0;c<local;++c)for(int r=0;r<local;++r)aux_entries.emplace_back(map[r]+na*map[c],auxiliary_count+r+local*c);
    auxiliary_count+=local*local;
  }
  std::vector<int> rhs_offsets,rhs_indices,aux_offsets,aux_indices;
  buildCsr(rhs_entries,n,rhs_offsets,rhs_indices);buildCsr(aux_entries,s.entries,aux_offsets,aux_indices);
  phase.emplace(diagnostics::Stage::cuda_configure_upload);
  upload(w.frozen,terms,s.stream);upload(w.matrices,matrices,s.stream);upload(w.constant,constant,s.stream);
  upload(w.rhs_offsets,rhs_offsets,s.stream);upload(w.rhs_indices,rhs_indices,s.stream);
  upload(w.aux_offsets,aux_offsets,s.stream);upload(w.aux_indices,aux_indices,s.stream);
  w.displacements.reserve(displacement_count);w.host_displacements.reserve(displacement_count);
  w.shifted.reserve(displacement_count);w.products.reserve(displacement_count);w.costs.reserve(terms.size());
  w.auxiliary.reserve(auxiliary_count);w.host_auxiliary.reserve(auxiliary_count);
  w.damped.reserve(static_cast<size_t>(n)*n);w.solution.reserve(n);w.host_solution.reserve(n);
  solverCheck(cusolverDnDpotrf_bufferSize(w.solver,CUBLAS_FILL_MODE_LOWER,n,w.damped.data,n,&w.workspace_count));
  w.workspace.reserve(w.workspace_count);
  check(cudaStreamSynchronize(s.stream));
  w.dimension=n;w.delta_count=displacement_count;w.auxiliary_count=auxiliary_count;w.frozen_count=terms.size();
}
void CudaSummaryBatch::residentLinearize(const std::vector<BatchPose>& poses,const std::vector<double>& deltas,const std::vector<double>& auxiliary) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::cuda_linearize);
  auto& s=*impl_;
  if(!s.resident || !s.resident->dimension)throw std::logic_error("Resident system is not configured");auto& w=*s.resident;
  if(poses.size()!=static_cast<size_t>(s.poses) || deltas.size()!=static_cast<size_t>(w.delta_count) || auxiliary.size()!=static_cast<size_t>(w.auxiliary_count))throw std::invalid_argument("Resident linearize input size mismatch");
  check(cudaStreamSynchronize(s.stream));w.ready=false;
  std::copy(poses.begin(),poses.end(),s.host_poses.data);std::copy(deltas.begin(),deltas.end(),w.host_displacements.data);std::copy(auxiliary.begin(),auxiliary.end(),w.host_auxiliary.data);
  check(cudaMemcpyAsync(s.pose_data.data,s.host_poses.data,poses.size()*sizeof(BatchPose),cudaMemcpyHostToDevice,s.stream));
  if(!deltas.empty())check(cudaMemcpyAsync(w.displacements.data,w.host_displacements.data,deltas.size()*sizeof(double),cudaMemcpyHostToDevice,s.stream));
  if(!auxiliary.empty())check(cudaMemcpyAsync(w.auxiliary.data,w.host_auxiliary.data,auxiliary.size()*sizeof(double),cudaMemcpyHostToDevice,s.stream));
  if(s.edges)evaluateEdges<false><<<s.edges,128,0,s.stream>>>(s.roots.data,s.pose_data.data,s.partial.data);
  assemble<<<(s.entries+255)/256,256,0,s.stream>>>(s.partial.data,s.offsets.data,s.indices.data,s.output.data,s.entries);
  if(w.frozen_count)shiftFrozen<<<w.frozen_count,256,0,s.stream>>>(w.frozen.data,w.matrices.data,w.displacements.data,w.products.data,w.shifted.data,w.costs.data);
  mergeResident<<<(s.entries+255)/256,256,0,s.stream>>>(s.output.data,w.constant.data,w.dimension,w.shifted.data,w.costs.data,w.frozen_count,w.rhs_offsets.data,w.rhs_indices.data,w.auxiliary.data,w.aux_offsets.data,w.aux_indices.data);
  check(cudaGetLastError());w.ready=true;
}
double CudaSummaryBatch::residentError(const std::vector<BatchPose>& poses,const std::vector<double>& deltas) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::cuda_error);
  auto& s=*impl_;
  if(!s.resident || !s.resident->dimension)throw std::logic_error("Resident system is not configured");auto& w=*s.resident;
  if(poses.size()!=static_cast<size_t>(s.poses) || deltas.size()!=static_cast<size_t>(w.delta_count))throw std::invalid_argument("Resident error input size mismatch");
  check(cudaStreamSynchronize(s.stream));
  std::copy(poses.begin(),poses.end(),s.host_poses.data);std::copy(deltas.begin(),deltas.end(),w.host_displacements.data);
  check(cudaMemcpyAsync(s.pose_data.data,s.host_poses.data,poses.size()*sizeof(BatchPose),cudaMemcpyHostToDevice,s.stream));
  if(!deltas.empty())check(cudaMemcpyAsync(w.displacements.data,w.host_displacements.data,deltas.size()*sizeof(double),cudaMemcpyHostToDevice,s.stream));
  if(s.edges)evaluateEdges<true><<<s.edges,128,0,s.stream>>>(s.roots.data,s.pose_data.data,s.partial.data);
  if(w.frozen_count)shiftFrozen<<<w.frozen_count,256,0,s.stream>>>(w.frozen.data,w.matrices.data,w.displacements.data,w.products.data,w.shifted.data,w.costs.data);
  residentCost<<<1,32,0,s.stream>>>(s.partial.data,s.edges,w.frozen.data,w.costs.data,w.frozen_count,w.scalars.data);
  check(cudaGetLastError());
  check(cudaMemcpyAsync(w.host_scalars.data+2,w.scalars.data+2,sizeof(double),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));return w.host_scalars.data[2];
}
bool CudaSummaryBatch::residentSolve(double lambda,bool diagonal,double minimum,double maximum,std::vector<double>& delta,double& old_error,double& new_error) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::cuda_solve);
  auto& s=*impl_;
  if(!s.resident || !s.resident->ready)throw std::logic_error("Resident solve requires a linear model");auto& w=*s.resident;const int n=w.dimension;
  if(!std::isfinite(lambda) || lambda<0 || (diagonal && (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum<=0 || maximum<minimum)))throw std::invalid_argument("Invalid resident damping parameters");
  prepareDamped<<<(n*n+255)/256,256,0,s.stream>>>(s.output.data,n,lambda,diagonal,minimum,maximum,w.damped.data,w.solution.data);
  check(cudaGetLastError());
  solverCheck(cusolverDnDpotrf(w.solver,CUBLAS_FILL_MODE_LOWER,n,w.damped.data,n,w.workspace.data,w.workspace_count,w.info.data));
  solverCheck(cusolverDnDpotrs(w.solver,CUBLAS_FILL_MODE_LOWER,n,1,w.damped.data,n,w.solution.data,n,w.info.data+1));
  linearErrors<<<1,256,0,s.stream>>>(s.output.data,n,w.solution.data,w.scalars.data);check(cudaGetLastError());
  check(cudaMemcpyAsync(w.host_info.data,w.info.data,2*sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  check(cudaMemcpyAsync(w.host_solution.data,w.solution.data,n*sizeof(double),cudaMemcpyDeviceToHost,s.stream));
  check(cudaMemcpyAsync(w.host_scalars.data,w.scalars.data,2*sizeof(double),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  if(w.host_info.data[0]!=0 || w.host_info.data[1]!=0)return false;
  if(!std::isfinite(w.host_scalars.data[0]) || !std::isfinite(w.host_scalars.data[1]))return false;
  for(int k=0;k<n;++k)if(!std::isfinite(w.host_solution.data[k]))return false;
  delta.assign(w.host_solution.data,w.host_solution.data+n);old_error=w.host_scalars.data[0];new_error=w.host_scalars.data[1];return true;
}
void CudaSummaryBatch::residentHessian(double* output) {
  auto& s=*impl_;if(!s.resident || !s.resident->ready)throw std::logic_error("Resident model is not linearized");
  check(cudaMemcpyAsync(s.host_output.data,s.output.data,s.entries*sizeof(double),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));std::copy_n(s.host_output.data,s.entries,output);
}

}
