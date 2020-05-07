#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#define N 16777216 
#define THREADS_PER_BLOCK 1024
 
__global__ void KMeansClustering(double *centroids,double *data,double *clstr1,double *clstr2,double *clstr3,int n,int *noOfCPoints)
{
 
int tid = (blockIdx.x*blockDim.x +threadIdx.x)*3;
 
if(tid<3*n){
  
   int index = (blockIdx.x*blockDim.x +threadIdx.x)*2;
   //__shared__ int clusters_point_count[3][THREADS_PER_BLOCK];
   __shared__ int s_cluster[3];
   s_cluster[0] = 0;
   s_cluster[1] = 0;
   s_cluster[2] = 0;
  // for(int i=0;i<blockDim.x;i++){
    //  clusters_point_count[0][i] = 0;
    //  clusters_point_count[1][i] = 0;
    //  clusters_point_count[2][i] = 0;
   //}
   __syncthreads();
  
   double data_x = data[tid];
   double data_y = data[tid+1];
 
   double *cluster[3];
   
   double d_1 = pow(data_x-centroids[0],2)+pow(data_y-centroids[1],2);
   double d_2 = pow(data_x-centroids[2],2)+pow(data_y-centroids[3],2);
   double d_3 = pow(data_x-centroids[4],2)+pow(data_y-centroids[5],2);
    
   cluster[0] = clstr1;
   cluster[1] = clstr2;
   cluster[2] = clstr3;
   int clusterIndex = d_1 > d_2 ? d_2 > d_3 ? 2 : 1  : d_1 < d_3 ? 0: 2 ;    
   
   for(int i=0;i<3;i++){
      if(i!=clusterIndex){ 
        double * clusterPtr = cluster[i];
        clusterPtr[index]   = 0.0;
        clusterPtr[index+1] = 0.0;
      }else{
        double * clusterPtr = cluster[clusterIndex];
        clusterPtr[index] = data_x;
        clusterPtr[index+1] = data_y;
      }
   }
  
   atomicAdd(&s_cluster[clusterIndex],1);
   __syncthreads();
      
    if(threadIdx.x < 3){
      atomicAdd(&noOfCPoints[threadIdx.x],s_cluster[threadIdx.x]);     
    }
   }
}
 

__global__ void sumCluster(double *cluster1,double *cluster2,double *cluster3,int n){
int tid = (blockIdx.x*blockDim.x +threadIdx.x)*2;

__shared__ double shared_data_1[THREADS_PER_BLOCK*2];
__shared__ double shared_data_2[THREADS_PER_BLOCK*2];
__shared__ double shared_data_3[THREADS_PER_BLOCK*2];
 
if(tid < n){ 
shared_data_1[2*threadIdx.x] = cluster1[tid];
shared_data_1[2*threadIdx.x+1] = cluster1[tid+1];
 
shared_data_2[2*threadIdx.x] = cluster2[tid];
shared_data_2[2*threadIdx.x+1] = cluster2[tid+1];
 
shared_data_3[2*threadIdx.x] = cluster3[tid];
shared_data_3[2*threadIdx.x+1] = cluster3[tid+1];
__syncthreads();
}
 
 
  int stride = blockDim.x; 
  while((stride >= 2) && (threadIdx.x < stride/2)){
   
    shared_data_1[2*threadIdx.x] += shared_data_1[2*threadIdx.x+stride];
    //addition for y
    shared_data_1[2*threadIdx.x+1]+=shared_data_1[2*threadIdx.x+stride+1];   
    
    //addition for x
    shared_data_2[2*threadIdx.x]+=shared_data_2[2*threadIdx.x+stride];
    //addition for y
    shared_data_2[2*threadIdx.x+1]+=shared_data_2[2*threadIdx.x+stride+1];
 
    //addition for x
    shared_data_3[2*threadIdx.x]+=shared_data_3[2*threadIdx.x+stride];
    //addition for y
    shared_data_3[2*threadIdx.x+1]+=shared_data_3[2*threadIdx.x+stride+1];
    __syncthreads();
    stride =  stride>>1;
  }
 
if(threadIdx.x == 0){
  cluster1[blockIdx.x*2] = shared_data_1[threadIdx.x];
  cluster1[blockIdx.x*2+1] = shared_data_1[threadIdx.x+1];
 
  cluster2[blockIdx.x*2] = shared_data_2[threadIdx.x];
  cluster2[blockIdx.x*2+1] = shared_data_2[threadIdx.x+1];
 
  cluster3[blockIdx.x*2] = shared_data_3[threadIdx.x];
  cluster3[blockIdx.x*2+1] = shared_data_3[threadIdx.x+1];

}
}
 
void checkCudaError(cudaError_t error,int lineNo){
      if (error !=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,lineNo);
        exit(EXIT_FAILURE);
     }
 
}
 
int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    FILE *inFile = fopen("16777216_CLUSTER_DATA.csv", "r");
    if(inFile == NULL){
        printf("Unable to read the data from the file");
        exit(1);
    }
    
    //Host memory allocation
    double *host_data = (double *)malloc(sizeof(double)*N*3);
    //CUDA memory allocation
    double *dev_data;
    cudaError_t error = cudaMalloc(&dev_data,N*3*sizeof(double));
    checkCudaError(error,__LINE__-1);
    for(int i =0;i<N;i++){
        fscanf(inFile, "%lf,%lf,%lf\n", &host_data[i*3],&host_data[i*3+1],&host_data[i*3+2]);
    }

   double *host_cluster_1 = (double *)calloc(N*2,sizeof(double));
   double *host_cluster_2 = (double *)calloc(N*2,sizeof(double));
   double *host_cluster_3 = (double *)calloc(N*2,sizeof(double));
   
   double *dev_c_1;
   double *dev_c_2;
   double *dev_c_3;
   error = cudaMalloc((void**)&dev_c_1,N*2*sizeof(double));
   checkCudaError(error,__LINE__-1);
   error = cudaMalloc((void**)&dev_c_2,N*2*sizeof(double));
   checkCudaError(error,__LINE__-1);
   error = cudaMalloc((void**)&dev_c_3,N*2*sizeof(double));
   checkCudaError(error,__LINE__-1);
   
 
   double* host_centroids = (double*)malloc(6*sizeof(double));
   double* dev_centroids;
   error = cudaMalloc((void**)&dev_centroids,6*sizeof(double));
   checkCudaError(error,__LINE__-1);  
   //Randomly initialising K centroids for the clusters
   srand(41);
 
   int index1 = (rand() % N )*3;
  
   host_centroids[0] = host_data[index1];
   host_centroids[1] = host_data[index1+1];
   int index2 = (rand() % N)*3;
  
   host_centroids[2] = host_data[index2];
   host_centroids[3] = host_data[index2+1];
   int index3 = (rand() % N)*3;
   
   host_centroids[4] = host_data[index3];
   host_centroids[5] = host_data[index3+1];
   printf("Initial Centroid Estimate\n");
   for(int i=0;i<=4;i+=2){
          printf("centroid[%d][0] = %lf centroid[%d][1] = %lf\n",i,host_centroids[i],i,host_centroids[i+1]);
   }
   //Data transfer to GPU
  /* error = cudaMemcpy(dev_centroids,host_centroids,6*sizeof(double),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);*/
   error = cudaMemcpy(dev_data,host_data,N*3*sizeof(double),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);
   
   int *h_noOfCPoints = (int*)calloc(3,sizeof(int));
   int *c_noOfCPoints;
   error = cudaMalloc((void**)&c_noOfCPoints,3*sizeof(int));
   checkCudaError(error,__LINE__-1);
   error = cudaMemcpy(c_noOfCPoints,h_noOfCPoints,3*sizeof(int),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);
 
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   
   double previous_centroids[6];
   int noOfIterations = 0;
   while(1){
   noOfIterations++;
   for(int i=0;i<6;i++){
       previous_centroids[i] = host_centroids[i] ;
   }
   error = cudaMemcpy(dev_centroids,host_centroids,6*sizeof(double),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);
   for(int i=0;i<3;i++){
       h_noOfCPoints[i] = 0 ;
   }
   error = cudaMemcpy(c_noOfCPoints,h_noOfCPoints,3*sizeof(int),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);

   KMeansClustering<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_centroids,dev_data,dev_c_1,dev_c_2,dev_c_3,N,c_noOfCPoints);
   error = cudaGetLastError();
   checkCudaError(error,__LINE__-2);
   
   error = cudaMemcpy(h_noOfCPoints,c_noOfCPoints,3*sizeof(int),cudaMemcpyDeviceToHost);
   checkCudaError(error,__LINE__-1);
   //printf("\ncluster points %d %d %d\n",h_noOfCPoints[0],h_noOfCPoints[1],h_noOfCPoints[2]);  
   int blockSize = THREADS_PER_BLOCK;
   int temp = N;
   
   while(1){
   
      if(temp>blockSize){        
          sumCluster<<<temp/blockSize,blockSize>>>(dev_c_1,dev_c_2,dev_c_3,temp*2);
          error = cudaGetLastError();
          checkCudaError(error,__LINE__-2);
      }
      else if (temp >= 32){
          sumCluster<<<1,temp>>>(dev_c_1,dev_c_2,dev_c_3,temp*2);
          error = cudaGetLastError();
          //printf("%d,%d\n",temp,blockSize);
          checkCudaError(error,__LINE__-2);
          break;
      }
      else{
          error = cudaMemcpy(host_cluster_1,dev_c_1,N*2*sizeof(double),cudaMemcpyDeviceToHost);
          checkCudaError(error,__LINE__-1);
          error = cudaMemcpy(host_cluster_2,dev_c_2,N*2*sizeof(double),cudaMemcpyDeviceToHost);
          checkCudaError(error,__LINE__-1);
          error = cudaMemcpy(host_cluster_3,dev_c_3,N*2*sizeof(double),cudaMemcpyDeviceToHost);
          checkCudaError(error,__LINE__-1);
          for(int i = 1 ; i < temp ; i++){
              host_cluster_1[0] += host_cluster_1[2*i];
              host_cluster_1[1] += host_cluster_1[2*i+1];
              host_cluster_2[0] += host_cluster_2[2*i];
              host_cluster_2[1] += host_cluster_2[2*i+1];
              host_cluster_3[0] += host_cluster_3[2*i];
              host_cluster_3[1] += host_cluster_3[2*i+1];       
          }
        break;    
      }
     if(temp > blockSize){
     temp = temp/blockSize;
     }      
   }
   if(temp>=32){
       error = cudaMemcpy(host_cluster_1,dev_c_1,N*2*sizeof(double),cudaMemcpyDeviceToHost);
       checkCudaError(error,__LINE__-1);
       error = cudaMemcpy(host_cluster_2,dev_c_2,N*2*sizeof(double),cudaMemcpyDeviceToHost);
       checkCudaError(error,__LINE__-1);
       error = cudaMemcpy(host_cluster_3,dev_c_3,N*2*sizeof(double),cudaMemcpyDeviceToHost);
       checkCudaError(error,__LINE__-1);       
   }
     

   double sumXcluster1 = host_cluster_1[0];
   double sumYcluster1 = host_cluster_1[1];
 
   double sumXcluster2 = host_cluster_2[0];
   double sumYcluster2 = host_cluster_2[1];
 
   double sumXcluster3 = host_cluster_3[0];
   double sumYcluster3 = host_cluster_3[1];
    
 
   host_centroids[0] = sumXcluster1/(double)h_noOfCPoints[0];
   host_centroids[1] = sumYcluster1/(double)h_noOfCPoints[0];
 
   host_centroids[2] = sumXcluster2/(double)h_noOfCPoints[1];
   host_centroids[3] = sumYcluster2/(double)h_noOfCPoints[1];
 
   host_centroids[4] = sumXcluster3/(double)h_noOfCPoints[2];
   host_centroids[5] = sumYcluster3/(double)h_noOfCPoints[2];
   
   //for(int i=0;i<=4;i+=2){
   //   printf("centroid[%d][0] = %lf centroid[%d][1] = %lf\n",i,host_centroids[i],i,host_centroids[i+1]);
   //}
   int count = 0;
   for(int i=0;i<6;i++){
      if(host_centroids[i] != previous_centroids[i]){
        break;
      }
      count++;
   }
   if(count == 6){
     break;
   }
 }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   for(int i=0;i<=4;i+=2){
      printf("centroid[%d][0] = %lf centroid[%d][1] = %lf\n",i,host_centroids[i],i,host_centroids[i+1]);
   }
   
   //Total no computations for one iteration
   //16 * N for kernel 1
   //2 * N for kernel 2
   double throughput = (24  * 2.0 * noOfIterations) *N/(1000*milliseconds);
   printf("\nThroughput is %lf MFLOPS",throughput);
   printf("\nTime is %f ms",milliseconds);

  return 0; 
}
 
 


