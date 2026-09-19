#include <optional>
#include "form/optimization/diagnostics.hpp"
#include "form/optimization/cuda_matcher.hpp"
#include "form/feature/cuda_qr.hpp"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace form {
namespace {
void check(cudaError_t code) {
  if(code!=cudaSuccess) throw std::runtime_error(std::string("CUDA matcher: ")+cudaGetErrorString(code));
}
template<class T> struct Buffer {
  T* data=nullptr; size_t capacity=0;
  ~Buffer() { if(data) cudaFree(data); }
  void reserve(size_t n) {
    if(n<=capacity) return;
    T* next=nullptr; const size_t size=std::max(n,capacity+capacity/2);
    check(cudaMalloc(reinterpret_cast<void**>(&next),size*sizeof(T)));
    if(data) check(cudaFree(data));
    data=next; capacity=size;
  }
  void upload(const std::vector<T>& values,cudaStream_t stream) {
    reserve(values.size());
    if(!values.empty()) check(cudaMemcpyAsync(data,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
  }
};
struct Bucket { int x,y,z,begin,count; };
struct Pose { double matrix[12]; };
struct Group { int first,rows; size_t offset; };
__host__ __device__ uint32_t hash(int x,int y,int z) {
  return uint32_t(x)*73856093u ^ uint32_t(y)*19349669u ^ uint32_t(z)*83492791u;
}
// Each coordinate uses two bits per neighbor (shift + 1), in FORM's exact
// visit order. Immediate masks avoid serialized divergent constant-memory reads.
__device__ int neighborShift(unsigned long long coordinates,int neighbor) {
  return int((coordinates>>(2*neighbor))&3)-1;
}

__global__ void extractWorld(CudaMatcher::MapPoint* points,int count,
                             size_t stride,double* world,const Pose* inverse_poses,
                             const int* pose_indices,int* invalid) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)) return;
  for(int axis=0;axis<4;++axis) world[size_t(axis)*stride+i]=points[i].world[axis];
  if(inverse_poses) {
    const Pose pose=inverse_poses[pose_indices[i]];
    const double x=points[i].world[0],y=points[i].world[1],z=points[i].world[2];
    const double nx=points[i].normal[0],ny=points[i].normal[1],nz=points[i].normal[2];
    for(int axis=0;axis<3;++axis) {
      const double* r=pose.matrix+4*axis;
      const double local=((r[0]*x+r[1]*y)+r[2]*z)+r[3];
      const double normal=(r[0]*nx+r[1]*ny)+r[2]*nz;
      points[i].local[axis]=local; points[i].normal[axis]=normal;
      if(!isfinite(local) || !isfinite(normal)) atomicExch(invalid,1);
    }
  }
}

// One warp owns one query. Dense voxel rows are scanned cooperatively; the
// lexicographic (distance, neighbor order, point order) reduction retains FORM's
// first-hit tie rule even when different lanes discover equal-distance points.
__global__ void nearest(const Bucket* buckets,int mask,const double* __restrict__ points,size_t point_stride,
                        const CudaMatcher::Query* queries,int count,double width,Pose pose,
                        CudaMatcher::Result* results) {
  const int lane=threadIdx.x%32;
  const int i=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(i>=count) return; // Whole warp exits, including the final partial block.
  const auto q=queries[i];
  double world[3],coords[3];
  for(int axis=0;axis<3;++axis) {
    const double* r=pose.matrix+4*axis;
    world[axis]=((r[0]*q.point[0]+r[1]*q.point[1])+r[2]*q.point[2])+r[3];
    coords[axis]=floor(world[axis]/width);
    if(!isfinite(coords[axis]) || coords[axis]<-2147483647. || coords[axis]>2147483646.) {
      if(lane==0) results[i]={0.,-2};
      return;
    }
  }
  const int x=int(coords[0]), y=int(coords[1]), z=int(coords[2]);
  double best=__longlong_as_double(0x7fefffffffffffffLL);
  unsigned long long rank=~0ULL;
  // Independent hash probes occupy 27 lanes instead of serializing every
  // lookup through lane zero. Divergent collision chains reconverge at ballot.
  int neighbor_begin=-1,neighbor_length=0;
  if(lane<27) {
    const int a=x+neighborShift(0x2a9542829549ULL,lane);
    const int b=y+neighborShift(0x2828295489495ULL,lane);
    const int c=z+neighborShift(0x8888888954955ULL,lane);
    int slot=hash(a,b,c)&mask;
    Bucket voxel=buckets[slot];
    while(voxel.begin!=-1 && (voxel.x!=a || voxel.y!=b || voxel.z!=c)) {
      slot=(slot+1)&mask; voxel=buckets[slot];
    }
    neighbor_begin=voxel.begin; neighbor_length=voxel.count;
  }
  unsigned occupied=__ballot_sync(0xffffffff,neighbor_length>0);
  while(occupied) {
    const int n=__ffs(occupied)-1;
    occupied&=occupied-1;
    const int begin=__shfl_sync(0xffffffff,neighbor_begin,n);
    const int length=__shfl_sync(0xffffffff,neighbor_length,n);
    for(size_t offset=lane;offset<size_t(length);offset+=32) {
      const int j=begin+int(offset);
      const double dx=points[j]-world[0],dy=points[point_stride+j]-world[1];
      const double dz=points[2*point_stride+j]-world[2],dw=points[3*point_stride+j]-q.point[3];
      const double distance=(dx*dx+dz*dz)+(dy*dy+dw*dw);
      if(distance<best) { best=distance; rank=(static_cast<unsigned long long>(n)<<32)|unsigned(j); }
    }
  }
  for(int shift=16;shift;shift/=2) {
    const double other=__shfl_down_sync(0xffffffff,best,shift);
    const auto other_rank=__shfl_down_sync(0xffffffff,rank,shift);
    if(other<best || (other==best && other_rank<rank)) { best=other; rank=other_rank; }
  }
  if(lane==0) results[i]={best,rank==~0ULL?-1:int(unsigned(rank))};
}

// Anchors change only after an uncertified full search, including in Audit.
struct SearchAnchor {
  double world[3],second;
  int cell[3],index,previous_index;
  bool empty,valid;
};
enum ReuseFlag : unsigned char { Certified=1, Searched=2, CellFallback=4,
  GapFallback=8, Unchanged=16, Mismatch=32, OracleSearched=64, InvalidQuery=128 };
__device__ double exactDistance(const double* points,size_t stride,int j,
                                const double* world,double padding) {
  const double dx=points[j]-world[0],dy=points[stride+j]-world[1];
  const double dz=points[2*stride+j]-world[2],dw=points[3*stride+j]-padding;
  return (dx*dx+dz*dz)+(dy*dy+dw*dw);
}
// See certified_rematching.md for the floating-point enclosure proof.
template<bool Squared>
__device__ bool gapCertified(const SearchAnchor& anchor,const double* world,double winner) {
  constexpr double error=0x1p-46; // 64 * binary64 epsilon
  constexpr double tiny=0x1p-1069; // 32 * binary64 denorm_min
  if(!isfinite(winner) || winner>=__longlong_as_double(0x7fefffffffffffffLL)) return false;
  double displacement2=0.;
  for(int axis=0;axis<3;++axis) {
    const double delta=fmax(fabs(__dsub_rd(world[axis],anchor.world[axis])),
                            fabs(__dsub_ru(world[axis],anchor.world[axis])));
    displacement2=__dadd_ru(displacement2,__dmul_ru(delta,delta));
  }
  if(!isfinite(displacement2)) return false;
  if constexpr(Squared) {
    // Exact binary64 reciprocal enclosures: 1-E <= 1/(1+E),
    // and 1+E+epsilon >= 1/(1-E). Clamp before multiplying the lower bound.
    const double reference=__dmul_rd(fmax(0.,__dsub_rd(anchor.second,tiny)),1.-error);
    const double threshold=__dmul_ru(__dadd_ru(winner,tiny),1.+error+0x1p-52);
    if(!isfinite(threshold)) return false;
    const double gap=__dsub_rd(__dsub_rd(reference,displacement2),threshold);
    if(!(gap>0.)) return false;
    // A positive gap and gap^2 > 4*D*W prove sqrt(R)>sqrt(D)+sqrt(W).
    // Underflow and overflow may weaken this test, but directed rounding
    // preserves the strict computed-distance ordering whenever it succeeds.
    return __dmul_rd(gap,gap)>__dmul_ru(4.,__dmul_ru(displacement2,threshold));
  } else {
    const double old2=fmax(0.,__ddiv_rd(__dsub_rd(anchor.second,tiny),1.+error));
    const double lower_norm=fmax(0.,__dsub_rd(__dsqrt_rd(old2),__dsqrt_ru(displacement2)));
    const double lower=__dsub_rd(__dmul_rd(1.-error,__dmul_rd(lower_norm,lower_norm)),tiny);
    return winner<lower;
  }
}
struct RankedPair {
  double best,second;
  unsigned long long rank,second_rank;
  __device__ void insert(double distance,unsigned long long candidate) {
    if(candidate==~0ULL) return;
    // Overlapping explicit voxel ranges may contain the same point index.
    if(rank!=~0ULL && unsigned(candidate)==unsigned(rank)) {
      if(candidate<rank) rank=candidate;
      return;
    }
    if(second_rank!=~0ULL && unsigned(candidate)==unsigned(second_rank)) {
      if(candidate>=second_rank) return;
      second_rank=candidate;
    }
    if(distance<best || (distance==best && candidate<rank && rank!=~0ULL)) {
      second=best; second_rank=rank; best=distance; rank=candidate;
    } else if(distance<second || (distance==second && candidate<second_rank && second_rank!=~0ULL)) {
      second=distance; second_rank=candidate;
    }
  }
};

// One thread owns one certificate, so a warp evaluates 32 queries. It never
// mutates anchors: Audit still needs the previous index, and all retained bounds
// must remain tied to the last uncertified full search.
template<bool Squared>
__global__ void certifyQueries(const double* __restrict__ points,size_t point_stride,
                               const CudaMatcher::Query* queries,int count,double width,Pose pose,
                               const SearchAnchor* anchors,CudaMatcher::Result* results,
                               unsigned char* flags) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)) return;
  const auto q=queries[i];
  double world[3],coords[3];
  for(int axis=0;axis<3;++axis) {
    const double* r=pose.matrix+4*axis;
    world[axis]=((r[0]*q.point[0]+r[1]*q.point[1])+r[2]*q.point[2])+r[3];
    coords[axis]=floor(world[axis]/width);
    if(!isfinite(coords[axis]) || coords[axis]<-2147483647. || coords[axis]>2147483646.) {
      results[i].distance=0.; results[i].index=-2; flags[i]=Searched|InvalidQuery;
      return;
    }
  }
  const int x=int(coords[0]),y=int(coords[1]),z=int(coords[2]);
  unsigned char status=0;
  bool certified=false;
  CudaMatcher::Result reused{__longlong_as_double(0x7fefffffffffffffLL),-1};
  if(anchors[i].valid) {
    const SearchAnchor& anchor=anchors[i];
    const bool same_cell=anchor.cell[0]==x && anchor.cell[1]==y && anchor.cell[2]==z;
    if(!same_cell) status|=CellFallback;
    else {
      const bool identical=world[0]==anchor.world[0] && world[1]==anchor.world[1] && world[2]==anchor.world[2];
      reused.index=anchor.index;
      if(reused.index>=0) reused.distance=exactDistance(points,point_stride,reused.index,world,q.point[3]);
      certified=identical || anchor.empty || (anchor.index>=0 && gapCertified<Squared>(anchor,world,reused.distance));
      if(!certified) status|=GapFallback;
    }
    if(certified) status|=Certified;
  }
  if(certified) { results[i].distance=reused.distance; results[i].index=reused.index; }
  flags[i]=status;
}

template<bool Occurrences,bool Split,bool Squared>
__global__ void nearestReuse(const Bucket* buckets,int mask,const double* __restrict__ points,size_t point_stride,
                        const CudaMatcher::Query* queries,int count,double width,Pose pose,
                        CudaMatcher::Result* results,SearchAnchor* anchors,unsigned char* flags,
                        bool anchors_valid,bool audit) {
  const int lane=threadIdx.x%32;
  const int i=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(i>=count) return; // Whole warp exits, including the final partial block.
  const double maximum=__longlong_as_double(0x7fefffffffffffffLL);
  unsigned char status=0;
  bool certified=false;
  CudaMatcher::Result reused{maximum,-1};
  if constexpr(Split) {
    status=flags[i];
    if(status&InvalidQuery) return; // Prepass already emitted the original -2 result.
    certified=status&Certified;
    if(certified) {
      if(lane==0) { reused.distance=results[i].distance; reused.index=results[i].index; }
      if(!audit) {
        if(lane==0) anchors[i].previous_index=reused.index;
        return; // Certified warps avoid even repeating the transform.
      }
    }
  }
  const auto q=queries[i];
  double world[3],coords[3];
  for(int axis=0;axis<3;++axis) {
    const double* r=pose.matrix+4*axis;
    world[axis]=((r[0]*q.point[0]+r[1]*q.point[1])+r[2]*q.point[2])+r[3];
    coords[axis]=floor(world[axis]/width);
    if(!isfinite(coords[axis]) || coords[axis]<-2147483647. || coords[axis]>2147483646.) {
      if(lane==0) { results[i]={0.,-2}; flags[i]=Searched; }
      return;
    }
  }
  const int x=int(coords[0]), y=int(coords[1]), z=int(coords[2]);
  if constexpr(!Split) {
    // Lane zero alone evaluates the bound and cached point; the warp branches uniformly.
    if(lane==0 && anchors_valid && anchors[i].valid) {
      const SearchAnchor& anchor=anchors[i];
      const bool same_cell=anchor.cell[0]==x && anchor.cell[1]==y && anchor.cell[2]==z;
      if(!same_cell) status|=CellFallback;
      else {
        const bool identical=world[0]==anchor.world[0] && world[1]==anchor.world[1] && world[2]==anchor.world[2];
        reused.index=anchor.index;
        if(reused.index>=0) reused.distance=exactDistance(points,point_stride,reused.index,world,q.point[3]);
        certified=identical || anchor.empty || (anchor.index>=0 && gapCertified<Squared>(anchor,world,reused.distance));
        if(!certified) status|=GapFallback;
      }
      if(certified) status|=Certified;
    }
    certified=__shfl_sync(0xffffffff,int(certified),0);
    if(certified && !audit) {
      if(lane==0) { results[i]=reused; flags[i]=status; anchors[i].previous_index=reused.index; }
      return;
    }
  }
  RankedPair pair{maximum,maximum,~0ULL,~0ULL};
  // Independent hash probes occupy 27 lanes instead of serializing every
  // lookup through lane zero. Divergent collision chains reconverge at ballot.
  int neighbor_begin=-1,neighbor_length=0;
  if(lane<27) {
    const int a=x+neighborShift(0x2a9542829549ULL,lane);
    const int b=y+neighborShift(0x2828295489495ULL,lane);
    const int c=z+neighborShift(0x8888888954955ULL,lane);
    int slot=hash(a,b,c)&mask;
    Bucket voxel=buckets[slot];
    while(voxel.begin!=-1 && (voxel.x!=a || voxel.y!=b || voxel.z!=c)) {
      slot=(slot+1)&mask; voxel=buckets[slot];
    }
    neighbor_begin=voxel.begin; neighbor_length=voxel.count;
  }
  unsigned occupied=__ballot_sync(0xffffffff,neighbor_length>0);
  const bool empty=occupied==0;
  while(occupied) {
    const int n=__ffs(occupied)-1;
    occupied&=occupied-1;
    const int begin=__shfl_sync(0xffffffff,neighbor_begin,n);
    const int length=__shfl_sync(0xffffffff,neighbor_length,n);
    for(size_t offset=lane;offset<size_t(length);offset+=32) {
      const int j=begin+int(offset);
      const double dx=points[j]-world[0],dy=points[point_stride+j]-world[1];
      const double dz=points[2*point_stride+j]-world[2],dw=points[3*point_stride+j]-q.point[3];
      const double distance=(dx*dx+dz*dz)+(dy*dy+dw*dw);
      const auto candidate=(static_cast<unsigned long long>(n)<<32)|unsigned(j);
      if constexpr(Occurrences) {
        // Second-smallest occurrence is a lower bound for every distinct
        // competitor, even if explicit voxel ranges repeat the winner index.
        pair.second=fmin(pair.second,fmax(pair.best,distance));
        if(distance<pair.best) { pair.best=distance; pair.rank=candidate; }
      } else pair.insert(distance,candidate);
    }
  }
  for(int shift=16;shift;shift/=2) {
    const double other=__shfl_down_sync(0xffffffff,pair.best,shift);
    const auto other_rank=__shfl_down_sync(0xffffffff,pair.rank,shift);
    const double other_second=__shfl_down_sync(0xffffffff,pair.second,shift);
    if constexpr(Occurrences) {
      pair.second=fmin(fmin(pair.second,other_second),fmax(pair.best,other));
      if(other<pair.best || (other==pair.best && other_rank<pair.rank)) {
        pair.best=other; pair.rank=other_rank;
      }
    } else {
      const auto other_second_rank=__shfl_down_sync(0xffffffff,pair.second_rank,shift);
      pair.insert(other,other_rank); pair.insert(other_second,other_second_rank);
    }
  }
  if(lane==0) {
    const CudaMatcher::Result result{pair.best,pair.rank==~0ULL?-1:int(unsigned(pair.rank))};
    status|=Searched;
    if(audit && anchors_valid && result.index==anchors[i].previous_index) status|=Unchanged;
    if(certified && (result.index!=reused.index || result.distance!=reused.distance)) status|=Mismatch;
    results[i]=result; flags[i]=status;
    if(!certified || (status&Mismatch)) {
      auto& anchor=anchors[i];
      for(int axis=0;axis<3;++axis) { anchor.world[axis]=world[axis]; anchor.cell[axis]=int(coords[axis]); }
      anchor.index=result.index; anchor.second=pair.second; anchor.empty=empty; anchor.valid=true;
    }
    anchors[i].previous_index=result.index;
  }
}

// Audit uses the untouched original kernel as an independent oracle on the
// identical snapshot. Return oracle results, and invalidate any disagreeing anchor.
__global__ void auditOriginal(const CudaMatcher::Result* oracle,CudaMatcher::Result* results,
                              SearchAnchor* anchors,unsigned char* flags,int count) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)) return;
  unsigned char status=flags[i]|OracleSearched;
  if(results[i].index!=oracle[i].index || results[i].distance!=oracle[i].distance) {
    status|=Mismatch; anchors[i].valid=false;
  }
  flags[i]=status;
  // Memberwise copy avoids reading Result's uninitialized trailing ABI padding.
  results[i].distance=oracle[i].distance; results[i].index=oracle[i].index;
}

__global__ void classify(const CudaMatcher::Result* results,int count,
                         const int* target_groups,int group_count,double threshold,
                         int* keys,int* sequence,int* counts) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)) return;
  const auto result=results[i];
  int group=group_count; // Rejected queries sort after every real group.
  if(result.index==-2) atomicExch(counts+group_count,1);
  if(result.index>=0 && result.distance<threshold) {
    const int target=target_groups[result.index];
    if(target>=0) { group=target; atomicAdd(counts+group,1); }
  }
  keys[i]=group;
  sequence[i]=int(i);
}

__global__ void pack(const CudaMatcher::MapPoint* points,const CudaMatcher::Query* queries,
                     const CudaMatcher::Result* results,const int* indices,const Group* groups,
                     bool plane,double* output) {
  const Group group=groups[blockIdx.y];
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=group.rows) return;
  const int q=indices[group.first+row];
  const auto p=points[results[q].index];
  const auto query=queries[q];
  double v[7];
  if(plane) {
    for(int k=0;k<3;++k) { v[k]=query.point[k]; v[3+k]=p.normal[k]; }
    // Eigen Vector3 dot: x + (y + z).
    v[6]=p.normal[0]*(query.point[0]-p.local[0])+
      (p.normal[1]*(query.point[1]-p.local[1])+p.normal[2]*(query.point[2]-p.local[2]));
  } else {
    v[0]=1.;
    for(int k=0;k<3;++k) { v[1+k]=p.local[k]; v[4+k]=query.point[k]-p.local[k]; }
  }
  for(int k=0;k<7;++k) output[group.offset+row+size_t(k)*group.rows]=v[k];
}

// Stable slots belong to a target scan. Pose changes alone never dirty a row:
// local feature coordinates are immutable for this snapshot.
__global__ void updateStableRows(const CudaMatcher::Result* results,const int* keys,int count,
                                int groups,int stride,int* query_slots,int* slot_queries,
                                int* previous_targets,int* dirty) {
  const int q=blockIdx.x*blockDim.x+threadIdx.x;
  if(q>=count) return;
  const int slot=query_slots[q],target=results[q].index;
  const int group=keys[q]<groups?keys[q]:-1;
  if(slot>=0 && (group!=slot/stride || target!=previous_targets[q])) {
    atomicExch(dirty+slot/64,1);
    if(group!=slot/stride) { slot_queries[slot]=-1; query_slots[q]=-1; }
  }
  previous_targets[q]=target;
}

// Queue order may vary across warps. Each accepted query owns exactly one slot,
// so row multiset and first-hole highwater remain independent of arrival order.
// Filling runs in a separate kernel: all removals and queue writes must finish
// before any block can reuse another block's former slot.
__global__ void queueStableRows(const CudaMatcher::Result* results,const int* keys,int count,
                               int groups,int stride,int* query_slots,int* slot_queries,
                               int* previous_targets,int* dirty,int* incoming,int* arrivals) {
  const int q=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%32;
  const unsigned active=__ballot_sync(0xffffffffu,q<count);
  if(q>=count) return;
  int slot=query_slots[q];
  const int target=results[q].index,group=keys[q]<groups?keys[q]:-1;
  if(slot>=0 && (group!=slot/stride || target!=previous_targets[q])) {
    atomicExch(dirty+slot/64,1);
    if(group!=slot/stride) {
      slot_queries[slot]=-1; query_slots[q]=-1; slot=-1;
    }
  }
  previous_targets[q]=target;
  const bool fresh=group>=0 && slot<0;
  const unsigned arriving=__ballot_sync(active,fresh);
  if(!fresh) return;
#if __CUDA_ARCH__ >= 700
  const unsigned peers=__match_any_sync(arriving,group);
  const int leader=__ffs(peers)-1;
  int offset=0;
  if(lane==leader) offset=atomicAdd(arrivals+group,__popc(peers));
  offset=__shfl_sync(peers,offset,leader);
  offset+=__popc(peers&((1u<<lane)-1));
#else
  // Queue order is unconstrained; retain compilation for pre-Volta targets.
  const int offset=atomicAdd(arrivals+group,1);
#endif
  incoming[group*stride+offset]=q;
}

__global__ void fillQueuedRows(int stride,int* query_slots,int* slot_queries,
                               const int* incoming,const int* arrival_counts,
                               int* highwater,int* dirty) {
  using Scan=cub::BlockScan<int,256>;
  __shared__ typename Scan::TempStorage temporary;
  __shared__ int holes,last;
  const int g=blockIdx.x,t=threadIdx.x,arrivals=arrival_counts[g];
  if(t==0) { holes=0; last=highwater[g]; }
  __syncthreads();
  const int end=highwater[g]+min(stride-highwater[g],arrivals);
  for(int start=0;start<end && holes<arrivals;start+=256) {
    const int slot=start+t;
    const int vacant=slot<end && slot_queries[g*stride+slot]<0;
    int prefix,total;
    Scan(temporary).ExclusiveSum(vacant,prefix,total);
    if(vacant && holes+prefix<arrivals) {
      const int q=incoming[g*stride+holes+prefix];
      slot_queries[g*stride+slot]=q; query_slots[q]=g*stride+slot;
      atomicExch(dirty+(g*stride+slot)/64,1);
      atomicMax(&last,slot+1);
    }
    __syncthreads();
    if(t==0) holes+=total;
    __syncthreads();
  }
  if(t==0) highwater[g]=last;
}

// One block per target scan compacts arriving rows and fills its first holes.
// Block scans preserve query order and avoid global allocation atomics.
__global__ void fillStableRows(const Group* groups,const int* sorted,int stride,
                              int* query_slots,int* slot_queries,int* incoming,
                              int* highwater,int* dirty) {
  using Scan=cub::BlockScan<int,256>;
  __shared__ typename Scan::TempStorage temporary;
  __shared__ int arrivals,holes,last;
  const int g=blockIdx.x,t=threadIdx.x;
  const Group group=groups[g];
  if(t==0) { arrivals=0; holes=0; last=highwater[g]; }
  __syncthreads();
  for(int start=0;start<group.rows;start+=256) {
    const int row=start+t;
    const int q=row<group.rows?sorted[group.first+row]:-1;
    const int fresh=q>=0 && query_slots[q]<0;
    int prefix,total;
    Scan(temporary).ExclusiveSum(fresh,prefix,total);
    if(fresh) incoming[g*stride+arrivals+prefix]=q;
    __syncthreads();
    if(t==0) arrivals+=total;
    __syncthreads();
  }
  const int end=highwater[g]+min(stride-highwater[g],arrivals);
  for(int start=0;start<end && holes<arrivals;start+=256) {
    const int slot=start+t;
    const int vacant=slot<end && slot_queries[g*stride+slot]<0;
    int prefix,total;
    Scan(temporary).ExclusiveSum(vacant,prefix,total);
    if(vacant && holes+prefix<arrivals) {
      const int q=incoming[g*stride+holes+prefix];
      slot_queries[g*stride+slot]=q; query_slots[q]=g*stride+slot;
      atomicExch(dirty+(g*stride+slot)/64,1);
      atomicMax(&last,slot+1);
    }
    __syncthreads();
    if(t==0) holes+=total;
    __syncthreads();
  }
  if(t==0) highwater[g]=last;
}

__global__ void packStableLeaves(const CudaMatcher::MapPoint* points,const CudaMatcher::Query* queries,
                                const CudaMatcher::Result* results,const int* slots,const int* highwater,
                                const int* dirty,unsigned char* flags,int stride,bool plane,double* output) {
  const int g=blockIdx.y;
  for(int leaf=blockIdx.x;leaf*64<highwater[g];leaf+=gridDim.x) {
  const int node=g*(stride/64)+leaf,row=threadIdx.x;
  const bool changed=dirty[node]!=0;
  if(row==0) flags[node]=changed;
  if(!changed) continue;
  const int q=slots[g*stride+leaf*64+row];
  double v[7]={};
  if(q>=0) {
    const auto p=points[results[q].index];
    const auto query=queries[q];
    if(plane) {
      for(int k=0;k<3;++k) { v[k]=query.point[k]; v[3+k]=p.normal[k]; }
      v[6]=p.normal[0]*(query.point[0]-p.local[0])+
        (p.normal[1]*(query.point[1]-p.local[1])+p.normal[2]*(query.point[2]-p.local[2]));
    } else {
      v[0]=1.;
      for(int k=0;k<3;++k) { v[1+k]=p.local[k]; v[4+k]=query.point[k]-p.local[k]; }
    }
  }
  for(int k=0;k<7;++k) output[size_t(node)*64*7+row+k*64]=v[k];
  }
}
}
struct CudaMatcher::Impl {
  cudaStream_t stream=nullptr;
  Buffer<Bucket> buckets;
  Buffer<MapPoint> points;
  Buffer<Pose> inverse_poses;
  Buffer<int> pose_indices,snapshot_error;
  Buffer<double> search_positions;
  size_t point_stride=0;
  Buffer<Query> queries;
  Buffer<Result> results,oracle_results;
  Buffer<SearchAnchor> anchors;
  Buffer<unsigned char> reuse_flags;
  CudaMatcher::ReuseMode reuse_mode=CudaMatcher::ReuseMode::Disabled;
  bool anchors_valid=false,stats_ready=false;
  CudaMatcher::SearchKernel search_kernel=CudaMatcher::SearchKernel::Fused;
  bool occurrence_bound=false,split_prepass=false,squared_certificate=false;
  Buffer<int> indices;
  Buffer<Group> groups;
  Buffer<double> packed;
  Buffer<int> target_groups,group_counts,sort_keys,sorted_keys,sequence;
  Buffer<unsigned char> sort_storage;
  std::vector<int> host_counts;
  int group_count=0,point_count=0,sort_bits=1;
  size_t sort_bytes=0;
  bool groups_ready=false,host_results_ready=false;
  std::vector<Result> host_results;
  int mask=0,count=0;
  double width=1.;
  bool ready=false,searched=false;
  BatchedCudaQr qr;
  bool incremental_summaries=false,summary_valid=false,summary_plane=false,summary_audit=false,summary_tree=false,summary_bounds=false,summary_row_queue=false;
  size_t summary_checks=0;
  double summary_relative_error=0.;
  Buffer<int> query_slots,slot_queries,previous_targets,incoming,highwater,dirty_leaves;
  Buffer<unsigned char> dirty_flags;
  Buffer<double> stable_packed;
  std::vector<int> host_highwater;
  int slot_stride=0;
  void invalidateSummaries() { summary_valid=false; qr.resetIncremental(); }
  std::vector<Eigen::MatrixXd> packIncremental(const std::vector<Group>& descriptors,bool plane);
  void launchSearch(const std::array<double,12>& matrix);
  std::vector<Eigen::MatrixXd> packSummaries(const std::vector<Group>& descriptors,
      const std::vector<BatchedCudaQr::DeviceInput>& shapes,size_t total,int max_rows,bool plane);
  Impl() { check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
  ~Impl() { if(stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
};
CudaMatcher::CudaMatcher():impl_(std::make_unique<Impl>()) {
  if(const char* value=std::getenv("FORM_CUDA_SUMMARY_REUSE")) {
    const std::string mode(value);
    if(mode=="blocks64" || mode=="audit64" || mode=="tree64" || mode=="audit-tree64") {
      setIncrementalSummaries(true);
      setSummaryTree(mode=="tree64" || mode=="audit-tree64");
      impl_->summary_audit=mode=="audit64" || mode=="audit-tree64";
    } else if(mode!="off") throw std::invalid_argument("Invalid FORM_CUDA_SUMMARY_REUSE (off|blocks64|audit64|tree64|audit-tree64)");
  }
  if(const char* value=std::getenv("FORM_CUDA_TREE_WARPS")) {
    const std::string cap(value);
    if(cap=="32") setSummaryTreeWarpCap(32);
    else if(cap=="128") setSummaryTreeWarpCap(128);
    else if(cap=="256") setSummaryTreeWarpCap(256);
    else throw std::invalid_argument("Invalid FORM_CUDA_TREE_WARPS (32|128|256)");
  }
  if(const char* value=std::getenv("FORM_CUDA_TREE_BOUNDS")) {
    const std::string enabled(value);
    if(enabled=="1") setSummaryTreeBounds(true);
    else if(enabled!="0") throw std::invalid_argument("Invalid FORM_CUDA_TREE_BOUNDS (0|1)");
  }
  if(const char* value=std::getenv("FORM_CUDA_SUMMARY_ROWS")) {
    const std::string rows(value);
    if(rows=="queue") setSummaryRowQueue(true);
    else if(rows!="sorted") throw std::invalid_argument("Invalid FORM_CUDA_SUMMARY_ROWS (sorted|queue)");
  }
  if(const char* value=std::getenv("FORM_CUDA_MATCH_REUSE")) {
    const std::string mode(value);
    if(mode=="audit") setReuseMode(ReuseMode::Audit);
    else if(mode=="certified") setReuseMode(ReuseMode::Certified);
    else if(mode!="off") throw std::invalid_argument("Invalid FORM_CUDA_MATCH_REUSE (off|audit|certified)");
  }
  if(const char* value=std::getenv("FORM_CUDA_MATCH_KERNEL")) {
    const std::string kernel(value);
    if(kernel=="split") setSearchKernel(SearchKernel::Split);
    else if(kernel!="fused") throw std::invalid_argument("Invalid FORM_CUDA_MATCH_KERNEL (fused|split)");
  }
  if(const char* value=std::getenv("FORM_CUDA_MATCH_CERTIFICATE")) {
    const std::string certificate(value);
    if(certificate=="squared") setSquaredCertificate(true);
    else if(certificate!="norm") throw std::invalid_argument("Invalid FORM_CUDA_MATCH_CERTIFICATE (norm|squared)");
  }
  if(const char* value=std::getenv("FORM_CUDA_MATCH_TOP2")) {
    const std::string bound(value);
    if(bound=="occurrences") setOccurrenceBound(true);
    else if(bound!="distinct") throw std::invalid_argument("Invalid FORM_CUDA_MATCH_TOP2 (distinct|occurrences)");
  }
}
CudaMatcher::~CudaMatcher()=default;
void CudaMatcher::setIncrementalSummaries(bool enabled) {
  impl_->invalidateSummaries(); impl_->incremental_summaries=enabled; impl_->summary_audit=false; impl_->summary_tree=false; impl_->summary_bounds=false; impl_->summary_row_queue=false;
}
void CudaMatcher::setSummaryTree(bool enabled) {
  impl_->invalidateSummaries(); impl_->summary_tree=enabled;
  if(enabled) impl_->incremental_summaries=true;
}
void CudaMatcher::setSummaryRowQueue(bool enabled) {
  impl_->invalidateSummaries(); impl_->summary_row_queue=enabled;
}
void CudaMatcher::setSummaryTreeBounds(bool enabled) {
  impl_->invalidateSummaries(); impl_->summary_bounds=enabled;
}
void CudaMatcher::setSummaryTreeWarpCap(size_t cap) { impl_->qr.setIncrementalTreeWarpCap(cap); }
CudaMatcher::SummaryStats CudaMatcher::summaryStats() {
  if(!impl_->incremental_summaries || !impl_->summary_valid) return {};
  const auto stats=impl_->summary_tree?impl_->qr.incrementalTreeStats():impl_->qr.incrementalStats();
  return {stats.active_leaves,size_t(stats.dirty_leaves),impl_->summary_checks,impl_->summary_relative_error,impl_->summary_tree,impl_->summary_tree && impl_->summary_bounds,impl_->summary_row_queue};
}
void CudaMatcher::setSearchKernel(SearchKernel kernel) {
  auto& state=*impl_;
  state.anchors_valid=false; state.stats_ready=false;
  if(kernel!=SearchKernel::Fused && kernel!=SearchKernel::Split)
    throw std::invalid_argument("Invalid CUDA search kernel");
  state.search_kernel=kernel;
}
void CudaMatcher::setSquaredCertificate(bool enabled) {
  impl_->anchors_valid=false; impl_->stats_ready=false; impl_->squared_certificate=enabled;
}
void CudaMatcher::setOccurrenceBound(bool enabled) {
  impl_->anchors_valid=false; impl_->stats_ready=false; impl_->occurrence_bound=enabled;
}
void CudaMatcher::setReuseMode(ReuseMode mode) {
  auto& s=*impl_;
  s.anchors_valid=false; s.stats_ready=false;
  if(mode!=ReuseMode::Disabled && mode!=ReuseMode::Audit && mode!=ReuseMode::Certified)
    throw std::invalid_argument("Invalid CUDA reuse mode");
  s.reuse_mode=mode;
}
CudaMatcher::ReuseStats CudaMatcher::reuseStats() {
  auto& s=*impl_;
  ReuseStats stats;
  if(!s.stats_ready) return stats;
  stats.total=s.count; stats.split_queries=s.split_prepass?s.count:0;
  if(s.reuse_mode==ReuseMode::Disabled) { stats.searched=s.count; return stats; }
  std::vector<unsigned char> flags(s.count);
  if(s.count) check(cudaMemcpyAsync(flags.data(),s.reuse_flags.data,s.count,cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  for(auto flag:flags) {
    stats.certified+=bool(flag&Certified); stats.searched+=bool(flag&Searched);
    stats.cell_fallback+=bool(flag&CellFallback); stats.gap_fallback+=bool(flag&GapFallback);
    stats.unchanged+=bool(flag&Unchanged); stats.mismatches+=bool(flag&Mismatch);
    stats.oracle_searched+=bool(flag&OracleSearched);
  }
  return stats;
}
void CudaMatcher::reset(const std::vector<Voxel>& voxels,const std::vector<MapPoint>& points,
                        const std::vector<Query>& queries,double width,
                        const std::vector<std::array<double,12>>& inverse_poses,
                        const std::vector<int>& point_pose_indices) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::matcher_reset);
  auto& s=*impl_;
  s.invalidateSummaries();
  s.anchors_valid=false; s.stats_ready=false;
  s.ready=false; s.searched=false; s.groups_ready=false; s.host_results_ready=false;
  if(!std::isfinite(width) || width<=0) throw std::invalid_argument("Invalid CUDA voxel width");
  constexpr size_t limit=std::numeric_limits<int>::max();
  if(points.size()>limit || queries.size()>limit || voxels.size()>limit/4)
    throw std::invalid_argument("CUDA matcher snapshot too large");
  if(inverse_poses.size()>limit ||
     (inverse_poses.empty()?!point_pose_indices.empty():point_pose_indices.size()!=points.size()))
    throw std::invalid_argument("Invalid CUDA inverse pose shape");
  std::vector<Pose> poses(inverse_poses.size());
  for(size_t p=0;p<inverse_poses.size();++p) {
    for(double v:inverse_poses[p]) if(!std::isfinite(v))
      throw std::invalid_argument("Nonfinite CUDA inverse pose");
    std::copy(inverse_poses[p].begin(),inverse_poses[p].end(),poses[p].matrix);
  }
  for(int index:point_pose_indices) if(index<0 || size_t(index)>=inverse_poses.size())
    throw std::invalid_argument("Invalid CUDA inverse pose index");
  for(const auto& p:points) {
    for(double v:p.world) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA map point");
    for(double v:p.local) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA local point");
    for(double v:p.normal) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA map normal");
  }
  for(const auto& q:queries) for(double v:q.point)
    if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA query");
  size_t size=1; while(size<std::max<size_t>(1,voxels.size()*2)) size*=2;
  std::vector<Bucket> buckets(size,Bucket{0,0,0,-1,0});
  for(const auto& v:voxels) {
    if(v.begin<0 || v.count<0 || size_t(v.begin)+v.count>points.size())
      throw std::invalid_argument("Invalid CUDA voxel point range");
    size_t index=hash(v.coords[0],v.coords[1],v.coords[2])&(size-1);
    while(buckets[index].begin!=-1) {
      const auto& b=buckets[index];
      if(b.x==v.coords[0] && b.y==v.coords[1] && b.z==v.coords[2])
        throw std::invalid_argument("Duplicate CUDA voxel");
      index=(index+1)&(size-1);
    }
    buckets[index]={v.coords[0],v.coords[1],v.coords[2],v.begin,v.count};
  }
  check(cudaStreamSynchronize(s.stream));
  s.buckets.upload(buckets,s.stream); s.points.upload(points,s.stream); s.queries.upload(queries,s.stream);
  int invalid_snapshot=0;
  if(!poses.empty()) {
    s.inverse_poses.upload(poses,s.stream); s.pose_indices.upload(point_pose_indices,s.stream);
    s.snapshot_error.reserve(1); check(cudaMemsetAsync(s.snapshot_error.data,0,sizeof(int),s.stream));
  }
  // Keep the public map records for summary packing. Nearest only reads four
  // world coordinates; plane starts are aligned for coalesced warp loads.
  s.point_stride=(points.size()+31)/32*32;
  s.search_positions.reserve(4*s.point_stride);
  if(!points.empty()) {
    extractWorld<<<1+(points.size()-1)/256,256,0,s.stream>>>(s.points.data,int(points.size()),s.point_stride,s.search_positions.data,
      poses.empty()?nullptr:s.inverse_poses.data,s.pose_indices.data,s.snapshot_error.data);
    check(cudaGetLastError());
  }
  s.results.reserve(queries.size()); s.host_results.resize(queries.size());
  s.mask=int(size-1); s.count=int(queries.size()); s.point_count=int(points.size()); s.width=width;
  if(!poses.empty()) check(cudaMemcpyAsync(&invalid_snapshot,s.snapshot_error.data,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  if(invalid_snapshot) throw std::invalid_argument("Nonfinite CUDA transformed map feature");
  s.ready=true;
}
void CudaMatcher::Impl::launchSearch(const std::array<double,12>& matrix) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_search_launch);
  auto& s=*this;
  const bool anchors_valid=s.anchors_valid;
  s.anchors_valid=false; s.stats_ready=false; s.split_prepass=false;
  s.searched=false; s.host_results_ready=false;
  if(!s.ready) throw std::logic_error("CUDA matcher must be reset before search");
  for(auto v:matrix) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite CUDA query pose");
  if(s.count) {
    Pose pose; std::copy(matrix.begin(),matrix.end(),pose.matrix);
    // Two queries per block: A100 sweeps over 64/128/256/512 favor 64 for
    // sparse and dense maps, with 62.5% occupancy and less block tail work.
    constexpr int search_threads=64;
    if(s.reuse_mode==ReuseMode::Disabled) {
      nearest<<<1+(s.count-1)/(search_threads/32),search_threads,0,s.stream>>>(s.buckets.data,s.mask,s.search_positions.data,s.point_stride,s.queries.data,s.count,s.width,pose,s.results.data);
    } else {
      s.anchors.reserve(s.count); s.reuse_flags.reserve(s.count);
      // The first full search cannot certify anything; omit the split prepass.
      s.split_prepass=s.search_kernel==SearchKernel::Split && anchors_valid;
      if(s.split_prepass) {
        auto certify=[&](auto squared) {
          certifyQueries<decltype(squared)::value><<<1+(s.count-1)/128,128,0,s.stream>>>(s.search_positions.data,s.point_stride,
            s.queries.data,s.count,s.width,pose,s.anchors.data,s.results.data,s.reuse_flags.data);
        };
        if(s.squared_certificate) certify(std::true_type{});
        else certify(std::false_type{});
        check(cudaGetLastError());
      }
      auto launch=[&](auto occurrences,auto split,auto squared) {
        nearestReuse<decltype(occurrences)::value,decltype(split)::value,decltype(squared)::value>
          <<<1+(s.count-1)/(search_threads/32),search_threads,0,s.stream>>>(s.buckets.data,s.mask,
          s.search_positions.data,s.point_stride,s.queries.data,s.count,s.width,pose,s.results.data,
          s.anchors.data,s.reuse_flags.data,anchors_valid,s.reuse_mode==ReuseMode::Audit);
      };
      auto launch_bound=[&](auto occurrences,auto split) {
        if(s.squared_certificate) launch(occurrences,split,std::true_type{});
        else launch(occurrences,split,std::false_type{});
      };
      if(s.occurrence_bound) {
        if(s.split_prepass) launch_bound(std::true_type{},std::true_type{});
        else launch_bound(std::true_type{},std::false_type{});
      } else {
        if(s.split_prepass) launch_bound(std::false_type{},std::true_type{});
        else launch_bound(std::false_type{},std::false_type{});
      }
      check(cudaGetLastError());
      if(s.reuse_mode==ReuseMode::Audit) {
        s.oracle_results.reserve(s.count);
        nearest<<<1+(s.count-1)/(search_threads/32),search_threads,0,s.stream>>>(s.buckets.data,s.mask,s.search_positions.data,s.point_stride,s.queries.data,s.count,s.width,pose,s.oracle_results.data);
        check(cudaGetLastError());
        auditOriginal<<<1+(s.count-1)/256,256,0,s.stream>>>(s.oracle_results.data,s.results.data,s.anchors.data,s.reuse_flags.data,s.count);
      }
    }
    check(cudaGetLastError());
  }
  s.anchors_valid=true; s.stats_ready=true;
}
const std::vector<CudaMatcher::Result>& CudaMatcher::search(const std::array<double,12>& matrix) try {
  impl_->launchSearch(matrix);
  impl_->searched=true;
  return downloadResults();
} catch(...) { impl_->anchors_valid=false; impl_->stats_ready=false; throw; }
const std::vector<CudaMatcher::Result>& CudaMatcher::downloadResults() {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_download);
  auto& s=*impl_;
  if(!s.searched) throw std::logic_error("CUDA results require a completed search");
  if(!s.host_results_ready) {
    if(s.count) check(cudaMemcpyAsync(s.host_results.data(),s.results.data,size_t(s.count)*sizeof(Result),cudaMemcpyDeviceToHost,s.stream));
    check(cudaStreamSynchronize(s.stream));
    for(const auto& r:s.host_results) if(r.index==-2) {
      s.searched=false; s.anchors_valid=false; s.stats_ready=false;
      throw std::invalid_argument("CUDA transformed query exceeds voxel coordinate range");
    }
    s.host_results_ready=true;
  }
  return s.host_results;
}
void CudaMatcher::setGroups(const std::vector<int>& target_groups,size_t group_count) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_group_setup);
  auto& s=*impl_;
  s.invalidateSummaries();
  s.groups_ready=false;
  if(!s.ready) throw std::logic_error("CUDA matcher must be reset before grouping");
  if(target_groups.size()!=size_t(s.point_count) || group_count>65535)
    throw std::invalid_argument("Invalid CUDA target group shape");
  for(int group:target_groups) if(group < -1 || (group>=0 && size_t(group)>=group_count))
    throw std::invalid_argument("Invalid CUDA target group index");
  s.target_groups.upload(target_groups,s.stream);
  s.group_count=int(group_count);
  s.group_counts.reserve(group_count+1); s.host_counts.resize(group_count+1);
  s.sort_keys.reserve(s.count); s.sorted_keys.reserve(s.count);
  s.sequence.reserve(s.count); s.indices.reserve(s.count);
  s.sort_bits=1;
  while((1u<<s.sort_bits)<=group_count) ++s.sort_bits;
  if(s.count) {
    check(cub::DeviceRadixSort::SortPairs(nullptr,s.sort_bytes,s.sort_keys.data,s.sorted_keys.data,
          s.sequence.data,s.indices.data,s.count,0,s.sort_bits,s.stream));
    s.sort_storage.reserve(s.sort_bytes);
  }
  check(cudaStreamSynchronize(s.stream));
  s.groups_ready=true;
}
CudaMatcher::GroupedSummary CudaMatcher::searchGrouped(const std::array<double,12>& matrix,
                                                       double threshold_squared,bool plane) try {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_search_grouped);
  auto& s=*impl_;
  s.searched=false; s.host_results_ready=false;
  if(!s.groups_ready) throw std::logic_error("CUDA target groups must be configured before grouped search");
  if(!std::isfinite(threshold_squared) || threshold_squared<=0)
    throw std::invalid_argument("Invalid CUDA squared match threshold");
  std::optional<diagnostics::Scope> phase; phase.emplace(diagnostics::Stage::match_classify_wait);
  s.launchSearch(matrix);
  check(cudaMemsetAsync(s.group_counts.data,0,s.host_counts.size()*sizeof(int),s.stream));
  if(s.count) {
    classify<<<1+(s.count-1)/256,256,0,s.stream>>>(s.results.data,s.count,s.target_groups.data,s.group_count,
      threshold_squared,s.sort_keys.data,s.sequence.data,s.group_counts.data);
    check(cudaGetLastError());
  }
  check(cudaMemcpyAsync(s.host_counts.data(),s.group_counts.data,s.host_counts.size()*sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  check(cudaStreamSynchronize(s.stream));
  phase.emplace(diagnostics::Stage::match_group_sort);
  if(s.host_counts.back()) throw std::invalid_argument("CUDA transformed query exceeds voxel coordinate range");
  s.searched=true;
  GroupedSummary result;
  std::vector<Group> descriptors;
  std::vector<BatchedCudaQr::DeviceInput> shapes;
  int first=0,max_rows=0;
  size_t total=0;
  for(int group=0;group<s.group_count;++group) {
    const int rows=s.host_counts[group];
    descriptors.push_back({first,rows,total}); shapes.push_back({size_t(rows),plane});
    result.counts.push_back(size_t(rows));
    first+=rows; total+=size_t(rows)*7; max_rows=std::max(max_rows,rows);
  }
  const bool queue_only=s.incremental_summaries && s.summary_row_queue &&
    !s.summary_audit && !BatchedCudaQr::hasInputObserver();
  if(first && !queue_only) {
    // Full QR, summary audits, and observers retain the original sorted order.
    // CUB's stable radix sort retains ascending input query order within each
    // group, independent of scheduling of the histogram's integer atomics.
    check(cub::DeviceRadixSort::SortPairs(s.sort_storage.data,s.sort_bytes,s.sort_keys.data,s.sorted_keys.data,
          s.sequence.data,s.indices.data,s.count,0,s.sort_bits,s.stream));
  }
  phase.reset();
  s.summary_checks=0; s.summary_relative_error=0.;
  if(s.incremental_summaries && !BatchedCudaQr::hasInputObserver()) {
    result.roots=s.packIncremental(descriptors,plane);
    if(s.summary_audit) {
      auto fresh=s.packSummaries(descriptors,shapes,total,max_rows,plane);
      for(size_t group=0;group<fresh.size();++group) {
        const Eigen::MatrixXd reference=fresh[group].transpose()*fresh[group];
        const Eigen::MatrixXd actual=result.roots[group].transpose()*result.roots[group];
        const double error=(actual-reference).norm()/(1.+reference.norm());
        if(!std::isfinite(error) || error>1e-10)
          throw std::runtime_error("Incremental summary differs from full QR reconstruction");
        s.summary_relative_error=std::max(s.summary_relative_error,error); ++s.summary_checks;
      }
      result.roots=std::move(fresh); // Keep the original solver feedback in audits.
    }
  }
  else {
    s.invalidateSummaries(); // Diagnostic observers require logical raw row order.
    result.roots=s.packSummaries(descriptors,shapes,total,max_rows,plane);
  }
  return result;
} catch(...) { impl_->anchors_valid=false; impl_->stats_ready=false; impl_->invalidateSummaries(); throw; }

std::vector<Eigen::MatrixXd> CudaMatcher::Impl::packIncremental(const std::vector<Group>& descriptors,bool plane) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_pack);
  auto& s=*this;
  if(!s.summary_valid || s.summary_plane!=plane) {
    s.invalidateSummaries();
    const size_t stride=std::max<size_t>(64,((size_t(s.count)+63)/64)*64);
    const size_t slots=stride*size_t(s.group_count);
    if(stride>size_t(std::numeric_limits<int>::max()) || slots>size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Incremental summary slot dimensions exceed int range");
    s.slot_stride=int(stride);
    s.query_slots.reserve(s.count); s.previous_targets.reserve(s.count);
    s.slot_queries.reserve(slots); s.incoming.reserve(slots);
    s.highwater.reserve(s.group_count); s.host_highwater.assign(s.group_count,0);
    s.dirty_leaves.reserve(slots/64+(s.summary_row_queue?size_t(s.group_count):0));
    s.dirty_flags.reserve(slots/64);
    s.stable_packed.reserve(slots*7);
    if(s.count) check(cudaMemsetAsync(s.query_slots.data,0xff,size_t(s.count)*sizeof(int),s.stream));
    if(slots) check(cudaMemsetAsync(s.slot_queries.data,0xff,slots*sizeof(int),s.stream));
    if(s.group_count) check(cudaMemsetAsync(s.highwater.data,0,size_t(s.group_count)*sizeof(int),s.stream));
    s.summary_plane=plane;
  }
  const size_t leaves=size_t(s.group_count)*(s.slot_stride/64);
  const size_t dirty_count=leaves+(s.summary_row_queue?size_t(s.group_count):0);
  if(dirty_count) check(cudaMemsetAsync(s.dirty_leaves.data,0,dirty_count*sizeof(int),s.stream));
  // Counters share the dirty allocation and its existing clear; derive their
  // address only after reserve above. Each group's queue fits count <= stride.
  int* arrivals=s.summary_row_queue && s.group_count?s.dirty_leaves.data+leaves:nullptr;
  if(!s.summary_row_queue) s.groups.upload(descriptors,s.stream);
  if(s.count) {
    if(s.summary_row_queue)
      queueStableRows<<<1+(s.count-1)/256,256,0,s.stream>>>(s.results.data,s.sort_keys.data,s.count,s.group_count,
        s.slot_stride,s.query_slots.data,s.slot_queries.data,s.previous_targets.data,s.dirty_leaves.data,
        s.incoming.data,arrivals);
    else
      updateStableRows<<<1+(s.count-1)/256,256,0,s.stream>>>(s.results.data,s.sort_keys.data,s.count,s.group_count,
        s.slot_stride,s.query_slots.data,s.slot_queries.data,s.previous_targets.data,s.dirty_leaves.data);
    check(cudaGetLastError());
  }
  if(s.group_count) {
    if(s.summary_row_queue)
      fillQueuedRows<<<s.group_count,256,0,s.stream>>>(s.slot_stride,s.query_slots.data,s.slot_queries.data,
        s.incoming.data,arrivals,s.highwater.data,s.dirty_leaves.data);
    else
      fillStableRows<<<s.group_count,256,0,s.stream>>>(s.groups.data,s.indices.data,s.slot_stride,
        s.query_slots.data,s.slot_queries.data,s.incoming.data,s.highwater.data,s.dirty_leaves.data);
    check(cudaGetLastError());
    if(!s.summary_tree)
      check(cudaMemcpyAsync(s.host_highwater.data(),s.highwater.data,size_t(s.group_count)*sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  }
  if(s.summary_tree) {
    if(s.group_count) {
      const int grid=std::min(s.slot_stride/64,32);
      packStableLeaves<<<dim3(grid,s.group_count),64,0,s.stream>>>(s.points.data,s.queries.data,s.results.data,
        s.slot_queries.data,s.highwater.data,s.dirty_leaves.data,s.dirty_flags.data,s.slot_stride,plane,s.stable_packed.data);
      check(cudaGetLastError());
    }
    size_t maximum=std::numeric_limits<size_t>::max();
    if(s.summary_bounds) {
      maximum=0;
      for(size_t group=0;group<descriptors.size();++group) {
        // First-hole allocation gives h' = max(h,current accepted count).
        // Reset/plane/group/failure invalidation clears this mirror above.
        s.host_highwater[group]=std::max(s.host_highwater[group],descriptors[group].rows);
        maximum=std::max(maximum,(size_t(s.host_highwater[group])+63)/64);
      }
    }
    auto roots=s.qr.computeDevicePackedIncrementalTree(s.stable_packed.data,s.dirty_flags.data,
      s.slot_stride/64,s.group_count,s.highwater.data,plane,s.stream,maximum);
    s.summary_valid=true;
    return roots;
  }
  check(cudaStreamSynchronize(s.stream));
  std::vector<size_t> active;
  int maximum=0;
  for(int high:s.host_highwater) { const int n=(high+63)/64; active.push_back(n); maximum=std::max(maximum,n); }
  if(maximum) {
    packStableLeaves<<<dim3(maximum,s.group_count),64,0,s.stream>>>(s.points.data,s.queries.data,s.results.data,
      s.slot_queries.data,s.highwater.data,s.dirty_leaves.data,s.dirty_flags.data,s.slot_stride,plane,s.stable_packed.data);
    check(cudaGetLastError());
  }
  auto roots=s.qr.computeDevicePackedIncremental(s.stable_packed.data,s.dirty_flags.data,s.slot_stride/64,active,plane,s.stream);
  s.summary_valid=true;
  return roots;
}
std::vector<Eigen::MatrixXd> CudaMatcher::Impl::packSummaries(const std::vector<Group>& descriptors,
    const std::vector<BatchedCudaQr::DeviceInput>& shapes,size_t total,int max_rows,bool plane) {
  diagnostics::Scope diagnostic_scope(diagnostics::Stage::match_pack);
  auto& s=*this;
  s.groups.upload(descriptors,s.stream); s.packed.reserve(total);
  if(max_rows) {
    pack<<<dim3(1+(max_rows-1)/128,descriptors.size()),128,0,s.stream>>>(s.points.data,s.queries.data,s.results.data,s.indices.data,s.groups.data,plane,s.packed.data);
    check(cudaGetLastError());
  }
  return s.qr.computeDevicePacked(s.packed.data,shapes,s.stream);
}
std::vector<Eigen::MatrixXd> CudaMatcher::summarize(const std::vector<std::vector<int>>& groups,bool plane) {
  auto& s=*impl_;
  if(!s.searched) throw std::logic_error("CUDA summary requires a completed search");
  downloadResults();
  if(groups.size()>65535) throw std::invalid_argument("Too many CUDA match groups");
  std::vector<int> indices;
  std::vector<Group> descriptors;
  std::vector<BatchedCudaQr::DeviceInput> shapes;
  size_t total=0; int max_rows=0;
  for(const auto& g:groups) {
    if(g.size()+indices.size()>size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Too many CUDA summary rows");
    for(int q:g) if(q<0 || q>=s.count || s.host_results[q].index<0)
      throw std::invalid_argument("Invalid CUDA summary query index");
    descriptors.push_back({int(indices.size()),int(g.size()),total});
    shapes.push_back({g.size(),plane});
    indices.insert(indices.end(),g.begin(),g.end()); total+=g.size()*7;
    max_rows=std::max(max_rows,int(g.size()));
  }
  s.indices.upload(indices,s.stream);
  return s.packSummaries(descriptors,shapes,total,max_rows,plane);
}
}
