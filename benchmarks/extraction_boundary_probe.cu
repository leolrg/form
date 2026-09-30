// MIT License
// Copyright (c) 2025 Easton Potokar, Taylor Pool, and Michael Kaess
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Diagnostic adaptation of extraction.tpp, not an estimator backend. Mirrors its
// float rules, including cap+1 behavior and suppression across sector boundaries.
#include <cuda_runtime.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <stdexcept>

struct Point { float x,y,z,w; };
struct Key { float value; int index; };
struct Params {
  int rows=128,cols=1024,sectors=6,neighbors=5,spacing=5,planes=50,points=3;
  double min2=.1*.1,max2=50.*50.,threshold=1.;
  __host__ __device__ int count() const {return rows*cols;}
  __host__ __device__ int stride() const {return 2+sectors*(planes+1)+sectors*(points+1);}
};
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Buffer {
  T* data=nullptr; size_t size;
  explicit Buffer(size_t n):size(n){check(cudaMalloc(&data,n*sizeof(T)));}
  ~Buffer(){cudaFree(data);}
  void upload(const std::vector<T>& v){if(v.size()!=size)throw std::runtime_error("upload shape");check(cudaMemcpy(data,v.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> download(){std::vector<T> v(size);check(cudaMemcpy(v.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__host__ __device__ float norm2(Point p) {return (p.x*p.x+p.z*p.z)+(p.y*p.y+p.w*p.w);}
__host__ __device__ bool interior(int c,Params p){return c>=p.neighbors && c<p.cols-p.neighbors;}
struct Masks {std::vector<bool> planar,point;};
Masks cpuMasks(const std::vector<Point>& scan,Params p) {
  Masks m{std::vector<bool>(p.count(),true),std::vector<bool>(p.count(),true)};
  for(int i=0;i<p.count();++i) {
    if(!interior(i%p.cols,p)){m.planar[i]=m.point[i]=false;continue;}
    double n=norm2(scan[i]);
    if(n<p.min2 || n>p.max2) {
      m.planar[i]=m.point[i]=false;
      for(int k=1;k<=p.neighbors;++k)m.planar[i-k]=m.planar[i+k]=false;
    }
  }
  return m;
}
std::vector<Key> curvature(const std::vector<Point>& scan,const Masks& m,Params p) {
  std::vector<Key> keys(p.count());
  for(int i=0;i<p.count();++i) {
    double v=FLT_MAX;
    if(m.planar[i]) {
      double x=-(2.*p.neighbors)*scan[i].x,y=-(2.*p.neighbors)*scan[i].y,z=-(2.*p.neighbors)*scan[i].z;
      for(int n=1;n<=p.neighbors;++n){x=x+scan[i-n].x+scan[i+n].x;y=y+scan[i-n].y+scan[i+n].y;z=z+scan[i-n].z+scan[i+n].z;}
      v=x*x+y*y+z*z;
    }
    keys[i]={float(v),i};
  }
  return keys;
}
std::vector<int> cpuSelect(const Masks& m,std::vector<Key> keys,Params p,bool stable,
                           std::vector<int>* sorted=nullptr) {
  std::vector<unsigned char> used(m.planar.begin(),m.planar.end());
  std::vector<int> out(p.rows*p.stride());
  tbb::parallel_for(0,p.rows,[&](int row){
    int* o=out.data()+row*p.stride();
    for(int s=0;s<p.sectors;++s){
      int first=row*p.cols+s*(p.cols/p.sectors),end=s+1==p.sectors?(row+1)*p.cols:first+p.cols/p.sectors;
      std::sort(keys.begin()+first,keys.begin()+end,[&](Key a,Key b){return a.value<b.value || (stable && a.value==b.value && a.index<b.index);});
      int count=0;
      for(int j=first;j<end;++j){int i=keys[j].index;
        if(used[i] && keys[j].value<p.threshold){o[2+o[0]++]=i;for(int k=0;k<p.spacing;++k)used[i+k]=used[i-k]=0;++count;}
        if(count>p.planes)break;
      }
    }
  });
  auto valid=m.point;
  for(int i=0;i<p.count();++i)valid[i]=(bool(used[i])==m.planar[i]) && valid[i];
  for(int row=0;row<p.rows;++row){int* o=out.data()+row*p.stride();
    for(int s=0;s<p.sectors;++s){
      if(!p.points)continue;
      int first=row*p.cols+s*(p.cols/p.sectors),end=s+1==p.sectors?(row+1)*p.cols:first+p.cols/p.sectors;
      std::vector<int> unused;for(int i=first;i<end;++i)if(valid[i])unused.push_back(i);
      int factor=1+int(unused.size())/p.points,count=0;
      for(int offset=0;offset<factor && count<=p.points;++offset)
        for(int j=offset;j<int(unused.size());j+=factor){int i=unused[j];
          if(valid[i]){o[2+p.sectors*(p.planes+1)+o[1]++]=i;for(int k=0;k<p.spacing;++k)valid[i+k]=valid[i-k]=false;++count;}
          if(count>p.points)break;
        }
    }
  }
  if(sorted){sorted->resize(keys.size());for(size_t i=0;i<keys.size();++i)(*sorted)[i]=keys[i].index;}
  return out;
}
// Implemented after the parity harness: range masks preserve the CPU's NaN
// comparisons and ignore edge samples as sources of neighborhood invalidation.
__global__ void rawMask(const Point* scan,unsigned char* point,Params p) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=p.count())return;
  double n=norm2(scan[i]);point[i]=interior(i%p.cols,p) && !(n<p.min2 || n>p.max2);
}
__global__ void dilateMask(const unsigned char* point,unsigned char* planar,Params p) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=p.count())return;
  int c=i%p.cols;bool good=interior(c,p);
  for(int k=-p.neighbors;k<=p.neighbors && good;++k)
    if(interior(c+k,p) && !point[i+k])good=false;
  planar[i]=good;
}
__global__ void sortSectors(const Key* input,int* indices,Params p) {
  __shared__ Key keys[256];
  int sector=blockIdx.x%p.sectors,row=blockIdx.x/p.sectors,t=threadIdx.x;
  int first=row*p.cols+sector*(p.cols/p.sectors),end=sector+1==p.sectors?(row+1)*p.cols:first+p.cols/p.sectors;
  keys[t]=first+t<end?input[first+t]:Key{FLT_MAX,INT_MAX};__syncthreads();
  for(int k=2;k<=256;k*=2)for(int j=k/2;j;j/=2){
    int other=t^j;
    if(other>t){Key a=keys[t],b=keys[other];bool less=a.value<b.value || (a.value==b.value && a.index<b.index);
      bool equal=a.value==b.value && a.index==b.index;
      if(!equal && (less==bool(t&k))){keys[t]=b;keys[other]=a;}
    }
    __syncthreads();
  }
  if(first+t<end)indices[first+t]=keys[t].index;
}
__global__ void selectRows(const Key* input,const int* sorted,const unsigned char* planar,
    unsigned char* point,unsigned char* used,int* scratch,int* output,Params p) {
  int row=blockIdx.x;
  for(int c=threadIdx.x;c<p.cols;c+=blockDim.x)used[row*p.cols+c]=planar[row*p.cols+c];
  __syncthreads();if(threadIdx.x)return;
  int* o=output+row*p.stride();o[0]=o[1]=0;
  for(int s=0;s<p.sectors;++s){
    int first=row*p.cols+s*(p.cols/p.sectors),end=s+1==p.sectors?(row+1)*p.cols:first+p.cols/p.sectors,count=0;
    for(int j=first;j<end;++j){int i=sorted[j];
      if(used[i] && input[i].value<p.threshold){o[2+o[0]++]=i;for(int k=0;k<p.spacing;++k)used[i+k]=used[i-k]=0;++count;}
      if(count>p.planes)break;
    }
  }
  for(int i=row*p.cols;i<(row+1)*p.cols;++i)point[i]=(used[i]==planar[i]) && point[i];
  for(int s=0;s<p.sectors;++s){
    if(!p.points)continue;
    int first=row*p.cols+s*(p.cols/p.sectors),end=s+1==p.sectors?(row+1)*p.cols:first+p.cols/p.sectors,n=0,count=0;
    for(int i=first;i<end;++i)if(point[i])scratch[first+n++]=i;
    int factor=1+n/p.points;
    for(int offset=0;offset<factor && count<=p.points;++offset)
      for(int j=offset;j<n;j+=factor){int i=scratch[first+j];
        if(point[i]){o[2+p.sectors*(p.planes+1)+o[1]++]=i;for(int k=0;k<p.spacing;++k)point[i+k]=point[i-k]=0;++count;}
        if(count>p.points)break;
      }
  }
}
struct Gpu {
  Params p;Buffer<Point> scan;Buffer<Key> keys;Buffer<int> sorted,scratch,out;
  Buffer<unsigned char> point,planar,used;
  explicit Gpu(Params q):p(q),scan(q.count()),keys(q.count()),sorted(q.count()),scratch(q.count()),out(q.rows*q.stride()),point(q.count()),planar(q.count()),used(q.count()){}
  void masks(){rawMask<<<(p.count()+255)/256,256>>>(scan.data,point.data,p);dilateMask<<<(p.count()+255)/256,256>>>(point.data,planar.data,p);}
  void select(bool sort){masks();if(sort)sortSectors<<<p.rows*p.sectors,256>>>(keys.data,sorted.data,p);selectRows<<<p.rows,32>>>(keys.data,sorted.data,planar.data,point.data,used.data,scratch.data,out.data,p);check(cudaGetLastError());}
};
bool same(const std::vector<int>& a,const std::vector<int>& b,Params p){
  for(int row=0;row<p.rows;++row){int base=row*p.stride();if(a[base]!=b[base] || a[base+1]!=b[base+1])return false;
    for(int j=0;j<a[base];++j)if(a[base+2+j]!=b[base+2+j])return false;
    for(int j=0;j<a[base+1];++j)if(a[base+2+p.sectors*(p.planes+1)+j]!=b[base+2+p.sectors*(p.planes+1)+j])return false;
  }return true;
}
bool sameSets(const std::vector<int>& a,const std::vector<int>& b,Params p){
  for(int row=0;row<p.rows;++row)for(int type=0;type<2;++type){
    int base=row*p.stride(),offset=base+2+(type?p.sectors*(p.planes+1):0);
    if(a[base+type]!=b[base+type])return false;
    std::vector<int> x(a.begin()+offset,a.begin()+offset+a[base+type]);
    std::vector<int> y(b.begin()+offset,b.begin()+offset+b[base+type]);
    std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());if(x!=y)return false;
  }return true;
}
template<class F> double timeUs(F f){auto start=std::chrono::steady_clock::now();f();return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}
void verify(const std::vector<Point>& scan,Params p,Gpu& gpu,const std::vector<Key>& keys,const Masks& m) {
  gpu.scan.upload(scan);gpu.keys.upload(keys);gpu.masks();
  auto pm=gpu.planar.download(),qm=gpu.point.download();
  for(int i=0;i<p.count();++i)if(bool(pm[i])!=m.planar[i] || bool(qm[i])!=m.point[i])throw std::runtime_error("mask mismatch");
  std::vector<int> sorted;auto ref=cpuSelect(m,keys,p,false,&sorted);gpu.sorted.upload(sorted);gpu.select(false);
  if(!same(ref,gpu.out.download(),p))throw std::runtime_error("ordered selection mismatch");
  auto stable=cpuSelect(m,keys,p,true);gpu.select(true);
  if(!same(stable,gpu.out.download(),p))throw std::runtime_error("stable GPU selection mismatch");
}
int main(int argc,char** argv) {
  try {
    if(argc<2)throw std::runtime_error("Usage: probe INPUT [SCAN_LIMIT=1190] [STRIDE=20] [REPEATS=8; 0=parity only]");
    tbb::global_control threads(tbb::global_control::max_allowed_parallelism,32);
    Params p; p.rows=2;p.cols=41;
    std::vector<Point> synthetic(p.count(),Point{5,0,0,0});
    synthetic[0].x=100;synthetic[20].x=0;synthetic[30].w=60;
    synthetic[60].x=NAN;
    // Synthetic equal curvature ties deliberately expose unspecified std::sort order.
    auto sm=cpuMasks(synthetic,p);auto sk=curvature(synthetic,sm,p);
    for(auto& k:sk)if(!std::isfinite(k.value))k.value=0; // mask NaN checked; sort NaNs unsupported
    Gpu sg(p);verify(synthetic,p,sg,sk,sm);
    std::ifstream in(argv[1],std::ios::binary);char magic[8];uint32_t rows,cols;
    in.read(magic,8);in.read(reinterpret_cast<char*>(&rows),4);in.read(reinterpret_cast<char*>(&cols),4);
    if(!in || std::memcmp(magic,"FORMPC01",8) || !rows || cols<30 || rows>2048 || cols>1536)throw std::runtime_error("input shape");
    p.rows=rows;p.cols=cols;
    if(p.cols/p.sectors+p.cols%p.sectors>256)throw std::runtime_error("prototype sort width >256");
    int limit=argc>2?std::stoi(argv[2]):1190,stride=argc>3?std::stoi(argv[3]):20,repeats=argc>4?std::stoi(argv[4]):8;
    if(limit<1 || stride<1 || repeats<0)throw std::runtime_error("invalid limits");
    std::cout<<std::setprecision(12)<<"scan,dense,repeat,cpu_masks_us,cpu_masks_selection_us,gpu_masks_resident_us,gpu_masks_roundtrip_us,gpu_ordered_resident_us,gpu_stable_resident_us,gpu_stable_roundtrip_us,stable_changes_sequence,stable_changes_membership\n";
    Gpu current(p);Params dense=p;dense.spacing=2;dense.planes=100;dense.points=6;Gpu more(dense);
    size_t checked=0,changed=0,membership=0;uint64_t checksum=0;
    for(int scanIndex=0;scanIndex<limit;++scanIndex){
      int64_t stamp;uint32_t n;in.read(reinterpret_cast<char*>(&stamp),8);
      if(in.eof() && in.gcount()==0)break;
      in.read(reinterpret_cast<char*>(&n),4);if(!in || n!=uint32_t(p.count()))throw std::runtime_error("scan header");
      std::vector<Point> scan(n);in.read(reinterpret_cast<char*>(scan.data()),n*sizeof(Point));if(!in)throw std::runtime_error("truncated scan");
      if(scanIndex%stride)continue;
      auto m=cpuMasks(scan,p);auto keys=curvature(scan,m,p);
      for(auto k:keys)if(!std::isfinite(k.value))throw std::runtime_error("real scan has nonfinite sort key");
      for(int d=0;d<2;++d){Params q=d?dense:p;Gpu& g=d?more:current;verify(scan,q,g,keys,m);
        std::vector<int> sorted;auto original=cpuSelect(m,keys,q,false,&sorted);auto stable=cpuSelect(m,keys,q,true);
        bool differs=!same(original,stable,q),differentSet=!sameSets(original,stable,q);++checked;changed+=differs;membership+=differentSet;
        if(!repeats){std::cout<<scanIndex<<','<<d<<",-1,0,0,0,0,0,0,0,"<<differs<<','<<differentSet<<'\n';continue;}
        for(int r=-2;r<repeats;++r){double times[7]{};
          // Reverse the operation order on alternate repeats; all buffers are warm.
          for(int j=0;j<7;++j){int op=(r%2)?6-j:j;
            if(op==4)g.sorted.upload(sorted);
            times[op]=timeUs([&]{switch(op){
              case 0:{auto v=cpuMasks(scan,q);checksum+=v.planar[10];break;}
              case 1:{auto mm=cpuMasks(scan,q);auto v=cpuSelect(mm,keys,q,false);checksum+=v[0];break;}
              case 2:g.masks();check(cudaDeviceSynchronize());break;
              case 3:g.scan.upload(scan);g.masks();{auto a=g.planar.download(),b=g.point.download();checksum+=a[10]+b[10];}break;
              case 4:g.select(false);check(cudaDeviceSynchronize());break;
              case 5:g.select(true);check(cudaDeviceSynchronize());break;
              case 6:g.scan.upload(scan);g.keys.upload(keys);g.select(true);{auto v=g.out.download();checksum+=v[0];}break;
            }});
          }
          if(r>=0){std::cout<<scanIndex<<','<<d<<','<<r;for(auto t:times)std::cout<<','<<t;std::cout<<','<<differs<<','<<differentSet<<'\n';}
        }
      }
    }
    std::cerr<<"verified_config_scans="<<checked<<" stable_changed_sequence="<<changed<<" stable_changed_membership="<<membership<<" checksum="<<checksum<<'\n';
    if(!checked)throw std::runtime_error("no scans tested");
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
