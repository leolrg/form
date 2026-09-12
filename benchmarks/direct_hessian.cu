#include "direct_hessian.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
namespace direct_experiment {
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
template<class T,bool Host=false> struct Buffer {
  T* data=nullptr;size_t capacity=0;
  ~Buffer() {if(data) {if constexpr(Host) cudaFreeHost(data);else cudaFree(data);}}
  void reserve(size_t n) {
    if(n<=capacity) return;
    if(data) {if constexpr(Host) check(cudaFreeHost(data));else check(cudaFree(data));}
    data=nullptr;capacity=0;
    if constexpr(Host) check(cudaMallocHost(reinterpret_cast<void**>(&data),n*sizeof(T)));
    else check(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(T)));
    capacity=n;
  }
};
struct Matrix {uint64_t offset;int rows,cols,first,tiles;};
struct Tile {int matrix,row;};
__device__ double skewEntry(const double* p,int row,int col) {
  if(row==col) return 0.;
  if(row==0) return col==1?-p[2]:p[1];
  if(row==1) return col==0?p[2]:-p[0];
  return col==0?-p[1]:p[0];
}
template<bool Cost>
__device__ void evaluateRow(const double* a,const Matrix& m,int row,const PoseData& p,double* out) {
  double q[13];
  for(int k=0;k<m.cols;++k) q[k]=a[m.offset+row+static_cast<uint64_t>(k)*m.rows];
  if(m.cols==7) {
    double pi[3]={q[1],q[2],q[3]},pj[3]={q[1]+q[4],q[2]+q[5],q[3]+q[6]};
    for(int c=0;c<3;++c) {
      double residual=q[0]*p.dt[c];
      for(int k=0;k<3;++k) residual+=(p.Rj[3*c+k]-p.Ri[3*c+k])*pi[k]+p.Rj[3*c+k]*q[4+k];
      if constexpr(Cost) out[c]=residual;
      else {
        for(int j=0;j<3;++j) {
          double ji=0,jj=0;
          for(int k=0;k<3;++k) {ji+=p.Ri[3*c+k]*skewEntry(pi,k,j);jj-=p.Rj[3*c+k]*skewEntry(pj,k,j);}
          out[13*c+j]=ji;out[13*c+3+j]=-q[0]*p.Ri[3*c+j];
          out[13*c+6+j]=jj;out[13*c+9+j]=q[0]*p.Rj[3*c+j];
        }
        out[13*c+12]=-residual;
      }
    }
  } else {
    double residual=q[12];
    for(int k=0;k<9;++k) residual+=q[k]*p.D[k];
    for(int k=0;k<3;++k) residual+=q[9+k]*p.t[k];
    if constexpr(Cost) out[0]=residual;
    else {
      for(int axis=0;axis<3;++axis) {
        const int x=(axis+1)%3,y=(axis+2)%3;
        double ji=q[9+x]*p.t[y]-q[9+y]*p.t[x],jj=0,jt=0;
        for(int k=0;k<3;++k) {
          ji+=q[3*x+k]*p.R[3*y+k]-q[3*y+k]*p.R[3*x+k];
          jj+=q[3*k+x]*p.R[3*k+y]-q[3*k+y]*p.R[3*k+x];
          jt+=q[9+k]*p.R[3*k+axis];
        }
        out[axis]=ji;out[3+axis]=-q[9+axis];out[6+axis]=jj;out[9+axis]=jt;
      }
      out[12]=-residual;
    }
  }
}
template<bool Cost>
__global__ void partials(const double* input,const Matrix* matrices,const Tile* tiles,
                         const PoseData* poses,double* partial) {
  const int tile=blockIdx.x,lane=threadIdx.x;
  const Tile t=tiles[tile];const Matrix m=matrices[t.matrix];
  if constexpr(Cost) {
    if(lane<32) {
      double residual[3]={0,0,0};
      if(t.row+lane<m.rows) evaluateRow<true>(input,m,t.row+lane,poses[t.matrix],residual);
      double sum=residual[0]*residual[0]+residual[1]*residual[1]+residual[2]*residual[2];
      for(int offset=16;offset;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
      if(!lane) partial[tile]=sum;
    }
  } else {
    __shared__ double augmented[32*39];
    if(lane<32) {
      double* a=augmented+lane*39;
      for(int k=0;k<39;++k) a[k]=0.;
      if(t.row+lane<m.rows) evaluateRow<false>(input,m,t.row+lane,poses[t.matrix],a);
    }
    __syncthreads();
    if(lane<169) {
      const int r=lane/13,c=lane%13,components=m.cols==7?3:1;double sum=0;
      for(int row=0;row<32;++row) for(int k=0;k<components;++k)
        sum+=augmented[row*39+k*13+r]*augmented[row*39+k*13+c];
      partial[static_cast<uint64_t>(tile)*169+lane]=sum;
    }
  }
}
template<bool Cost>
__global__ void finish(const Matrix* matrices,const double* partial,double* result) {
  const int factor=blockIdx.x,entry=threadIdx.x,width=Cost?1:169;
  const Matrix m=matrices[factor];
  if(entry<width) {
    double sum=0;
    for(int k=0;k<m.tiles;++k) sum+=partial[static_cast<uint64_t>(m.first+k)*width+entry];
    result[static_cast<uint64_t>(factor)*width+entry]=sum;
  }
}
}
struct Batch::Impl {
  cudaStream_t stream=nullptr;size_t count=0,tiles=0;
  Buffer<double> inputs,partial,output;Buffer<Matrix> matrices;Buffer<Tile> tasks;Buffer<PoseData> poses;
  Buffer<double,true> host_input,host_output;Buffer<PoseData,true> host_poses;
  Impl(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
  ~Impl(){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}
};
Batch::Batch():impl_(std::make_unique<Impl>()){}
Batch::~Batch()=default;
void Batch::upload(const std::vector<MatrixInput>& inputs) {
  if(inputs.empty()) throw std::runtime_error("Empty direct-Hessian batch");
  auto& s=*impl_;std::vector<Matrix> matrices;std::vector<Tile> tiles;size_t count=0;
  for(const auto& a:inputs) {
    if(a.rows<0 || (a.cols!=7 && a.cols!=13)) throw std::runtime_error("Invalid direct-Hessian input shape");
    const int n=std::max(1,(a.rows+31)/32),index=matrices.size();
    matrices.push_back({count,a.rows,a.cols,static_cast<int>(tiles.size()),n});
    for(int k=0;k<n;++k) tiles.push_back({index,k*32});
    count+=static_cast<size_t>(a.rows)*a.cols;
  }
  s.host_input.reserve(std::max<size_t>(1,count));size_t offset=0;
  for(const auto& a:inputs) {const size_t n=static_cast<size_t>(a.rows)*a.cols;
    if(n) std::copy(a.values,a.values+n,s.host_input.data+offset);offset+=n;}
  s.inputs.reserve(std::max<size_t>(1,count));s.partial.reserve(tiles.size()*169);
  s.output.reserve(inputs.size()*169);s.host_output.reserve(inputs.size()*169);s.poses.reserve(inputs.size());s.host_poses.reserve(inputs.size());
  s.matrices.reserve(matrices.size());s.tasks.reserve(tiles.size());s.count=inputs.size();s.tiles=tiles.size();
  if(count) check(cudaMemcpyAsync(s.inputs.data,s.host_input.data,count*8,cudaMemcpyHostToDevice,s.stream));
  check(cudaMemcpyAsync(s.matrices.data,matrices.data(),matrices.size()*sizeof(Matrix),cudaMemcpyHostToDevice,s.stream));
  check(cudaMemcpyAsync(s.tasks.data,tiles.data(),tiles.size()*sizeof(Tile),cudaMemcpyHostToDevice,s.stream));
  check(cudaStreamSynchronize(s.stream));
}
std::vector<double> Batch::evaluate(const std::vector<PoseData>& poses,bool cost_only) {
  auto& s=*impl_;if(poses.size()!=s.count) throw std::runtime_error("Pose count mismatch");
  std::copy(poses.begin(),poses.end(),s.host_poses.data);
  check(cudaMemcpyAsync(s.poses.data,s.host_poses.data,poses.size()*sizeof(PoseData),cudaMemcpyHostToDevice,s.stream));
  if(cost_only) {partials<true><<<s.tiles,256,0,s.stream>>>(s.inputs.data,s.matrices.data,s.tasks.data,s.poses.data,s.partial.data);
    finish<true><<<s.count,256,0,s.stream>>>(s.matrices.data,s.partial.data,s.output.data);}
  else {partials<false><<<s.tiles,256,0,s.stream>>>(s.inputs.data,s.matrices.data,s.tasks.data,s.poses.data,s.partial.data);
    finish<false><<<s.count,256,0,s.stream>>>(s.matrices.data,s.partial.data,s.output.data);}
  check(cudaGetLastError());const size_t values=s.count*(cost_only?1:169);
  check(cudaMemcpyAsync(s.host_output.data,s.output.data,values*8,cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  return {s.host_output.data,s.host_output.data+values};
}
}
