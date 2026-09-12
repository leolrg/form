// MIT License; see factor.hpp for copyright and license text.
#include "form/feature/batch_summary_cuda.hpp"
#include <cuda_runtime.h>
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
}
struct CudaSummaryBatch::Impl {
  int poses,edges,entries;cudaStream_t stream=nullptr;
  Buffer<BatchRoot> roots;Buffer<BatchPose> pose_data;Buffer<int> offsets,indices;
  Buffer<double> partial,output;Buffer<BatchPose,true> host_poses;Buffer<double,true> host_output;
  Impl(int n,size_t e,size_t os,size_t is):poses(n),edges(e),entries((6*n+1)*(6*n+1)),roots(e),pose_data(n),offsets(os),indices(is),partial(e*169),output(entries),host_poses(n),host_output(entries) {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
  ~Impl(){if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
CudaSummaryBatch::CudaSummaryBatch(int poses,const std::vector<BatchRoot>& roots,const std::vector<int>& offsets,const std::vector<int>& indices):impl_(std::make_unique<Impl>(poses,roots.size(),offsets.size(),indices.size())) {
  reset(poses,roots,offsets,indices);
}
void CudaSummaryBatch::reset(int poses,const std::vector<BatchRoot>& roots,const std::vector<int>& offsets,const std::vector<int>& indices) {
  auto& s=*impl_;s.poses=poses;s.edges=roots.size();s.entries=(6*poses+1)*(6*poses+1);
  s.roots.reserve(roots.size());s.pose_data.reserve(poses);s.offsets.reserve(offsets.size());s.indices.reserve(indices.size());
  s.partial.reserve(roots.size()*169);s.output.reserve(s.entries);s.host_poses.reserve(poses);s.host_output.reserve(s.entries);
  if(!roots.empty())check(cudaMemcpyAsync(s.roots.data,roots.data(),roots.size()*sizeof(BatchRoot),cudaMemcpyHostToDevice,s.stream));
  check(cudaMemcpyAsync(s.offsets.data,offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice,s.stream));
  if(!indices.empty())check(cudaMemcpyAsync(s.indices.data,indices.data(),indices.size()*sizeof(int),cudaMemcpyHostToDevice,s.stream));
  check(cudaStreamSynchronize(s.stream));
}
CudaSummaryBatch::~CudaSummaryBatch()=default;
void CudaSummaryBatch::evaluate(const std::vector<BatchPose>& poses,double* output,bool cost) {
  auto& s=*impl_;std::copy(poses.begin(),poses.end(),s.host_poses.data);
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
}
