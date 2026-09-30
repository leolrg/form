// Internal kernels for the opt-in resident extractor. Preserve legacy ordering
// by default; explicit curvature/index ties remain available for experiments.
template<class T>
__global__ void selectionRawMask(const Point<T>* scan,unsigned char* point,int count,
    CudaExtraction::Selection p,CudaExtraction::Reduction reduction) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  const int c=i%p.columns;
  double n=double(distanceSquared(scan[i],Point<T>{0,0,0,0},reduction));
  point[i]=c>=p.neighbors && c<p.columns-p.neighbors && !(n<p.min_squared || n>p.max_squared);
}
__global__ void selectionPlanarMask(const unsigned char* point,unsigned char* valid,
    int count,int columns,int neighbors) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  const int c=i%columns;bool good=c>=neighbors && c<columns-neighbors;
  for(int k=-neighbors;k<=neighbors && good;++k)
    if(c+k>=neighbors && c+k<columns-neighbors && !point[i+k])good=false;
  valid[i]=good;
}
__global__ void selectionSort(const double* curvature,int* sorted,int* error,
    int columns,int sectors,bool stable_order) {
  __shared__ double keys[1024];__shared__ int indices[1024];
  int sector=blockIdx.x%sectors,row=blockIdx.x/sectors,t=threadIdx.x;
  int first=row*columns+sector*(columns/sectors),end=sector+1==sectors?(row+1)*columns:first+columns/sectors;
  keys[t]=first+t<end?curvature[first+t]:DBL_MAX;
  indices[t]=first+t<end?first+t:INT_MAX;
  if(first+t<end && !isfinite(keys[t]))atomicExch(error,1);
  __syncthreads();
  if(!stable_order) {
    if(!t) {
      bool finite=true;for(int i=0;i<end-first;++i)finite=finite && isfinite(keys[i]);
      // Sort local indices against shared keys, then restore the global offset.
      for(int i=0;i<end-first;++i)indices[i]=i;
      if(finite)detail::legacySelectionSort(indices,keys,end-first);
      for(int i=0;i<end-first;++i)indices[i]+=first;
    }
    __syncthreads();
  } else for(int k=2;k<=blockDim.x;k*=2)for(int j=k/2;j;j/=2){
    int other=t^j;
    if(other>t){double a=keys[t],b=keys[other];int ai=indices[t],bi=indices[other];
      bool less=a<b || (a==b && ai<bi),equal=a==b && ai==bi;
      if(!equal && less==bool(t&k)){keys[t]=b;keys[other]=a;indices[t]=bi;indices[other]=ai;}
    }__syncthreads();
  }
  if(first+t<end)sorted[first+t]=indices[t];
}
__global__ void selectionRows(const double* curvature,const int* sorted,
    const unsigned char* valid,unsigned char* point,unsigned char* used,
    int* scratch,int* planes,int* points,int* counts,const int* error,CudaExtraction::Selection p) {
  int row=blockIdx.x,base=row*p.columns;
  // Reject malformed sort input before any potentially padded index is read.
  if(*error){if(!threadIdx.x){counts[2*row]=0;counts[2*row+1]=0;}return;}
  for(int c=threadIdx.x;c<p.columns;c+=blockDim.x)used[base+c]=valid[base+c];
  __syncthreads();if(threadIdx.x)return;
  int np=0,nq=0;
  for(int s=0;s<p.sectors;++s){
    int first=base+s*(p.columns/p.sectors),end=s+1==p.sectors?base+p.columns:first+p.columns/p.sectors;
    size_t count=0;
    for(int j=first;j<end;++j){int i=sorted[j];
      if(used[i] && curvature[i]<p.threshold){planes[base+np++]=i;for(int k=0;k<p.spacing;++k)used[i+k]=used[i-k]=0;++count;}
      if(count>p.planes)break;
    }
  }
  for(int i=base;i<base+p.columns;++i)point[i]=(used[i]==valid[i]) && point[i];
  for(int s=0;s<p.sectors;++s){
    if(!p.points)continue;
    int first=base+s*(p.columns/p.sectors),end=s+1==p.sectors?base+p.columns:first+p.columns/p.sectors,n=0;
    for(int i=first;i<end;++i)if(point[i])scratch[first+n++]=i;
    size_t factor=1+size_t(n)/p.points,count=0;
    for(size_t offset=0;offset<factor && count<=p.points;++offset)
      for(size_t j=offset;j<size_t(n);j+=factor){int i=scratch[first+j];
        if(point[i]){points[base+nq++]=i;for(int k=0;k<p.spacing;++k)point[i+k]=point[i-k]=0;++count;}
        if(count>p.points)break;
      }
  }
  counts[2*row]=np;counts[2*row+1]=nq;
}
__global__ void selectionCompact(const int* planes,const int* points,const int* counts,
    int* plane_output,int* point_output,int* totals,int rows,int columns) {
  const int row=blockIdx.x;int po=0,qo=0;
  for(int r=0;r<row;++r){po+=counts[2*r];qo+=counts[2*r+1];}
  for(int i=threadIdx.x;i<counts[2*row];i+=blockDim.x)plane_output[po+i]=planes[row*columns+i];
  for(int i=threadIdx.x;i<counts[2*row+1];i+=blockDim.x)point_output[qo+i]=points[row*columns+i];
  if(row==rows-1 && threadIdx.x==0){totals[0]=po+counts[2*row];totals[1]=qo+counts[2*row+1];}
}
