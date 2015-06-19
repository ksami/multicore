//HJM_Securities.cpp
//Routines to compute various security prices using HJM framework (via Simulation).
//Authors: Mark Broadie, Jatin Dewanwala
//Collaborator: Mikhail Smelyanskiy, Jike Chong, Intel

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "nr_routines.h"
#include "HJM.h"
#include "HJM_Securities.h"
#include "HJM_type.h"

#ifdef ENABLE_MPI
#include "mpi.h"
#endif

#ifdef ENABLE_THREADS
#include <pthread.h>
#define MAX_THREAD 1024

#ifdef TBB_VERSION
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/cache_aligned_allocator.h"
tbb::cache_aligned_allocator<FTYPE> memory_ftype;
tbb::cache_aligned_allocator<parm> memory_parm;
#define TBB_GRAINSIZE 1
#endif // TBB_VERSION
#endif //ENABLE_THREADS


#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

int NUM_TRIALS = DEFAULT_NUM_TRIALS;
int nThreads = 1;
int nSwaptions = 1;
int iN = 11; 
FTYPE dYears = 5.5; 
int iFactors = 3; 
parm *swaptions;

// =================================================
FTYPE *dSumSimSwaptionPrice_global_ptr;
FTYPE *dSumSquareSimSwaptionPrice_global_ptr;
int chunksize;


#ifdef TBB_VERSION
struct Worker {
    Worker(){}
    void operator()(const tbb::blocked_range<int> &range) const {
        FTYPE pdSwaptionPrice[2];
        int begin = range.begin();
        int end   = range.end();

        for(int i=begin; i!=end; i++) {
            int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
                swaptions[i].dCompounding, swaptions[i].dMaturity, 
                swaptions[i].dTenor, swaptions[i].dPaymentInterval,
                swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
                swaptions[i].pdYield, swaptions[i].ppdFactors,
                100, NUM_TRIALS, BLOCK_SIZE, 0);
            assert(iSuccess == 1);
            swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
            swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];

        }
    }
};

#endif //TBB_VERSION


void * worker(void *arg){
    int tid = *((int *)arg);
    FTYPE pdSwaptionPrice[2];
    FTYPE *pdZ = (FTYPE *)malloc(3*16*11*sizeof(FTYPE));
    FTYPE *randZ = (FTYPE *)malloc(3*16*11*sizeof(FTYPE));

#ifdef ENABLE_MPI
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int chunksize = nSwaptions/numprocs;
    int beg = tid*chunksize;
    int end = (tid+1)*chunksize;
    if(tid == numprocs -1 )
        end = nSwaptions;
    printf("id: %d, beg: %d, end: %d\n", tid, beg, end);
#else
    int chunksize = nSwaptions/nThreads;
    int beg = tid*chunksize;
    int end = (tid+1)*chunksize;
    if(tid == nThreads -1 )
        end = nSwaptions;
#endif

    for(int i=beg; i < end; i++) {
        int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
            swaptions[i].dCompounding, swaptions[i].dMaturity, 
            swaptions[i].dTenor, swaptions[i].dPaymentInterval,
            swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
            swaptions[i].pdYield, swaptions[i].ppdFactors,
            100, NUM_TRIALS, BLOCK_SIZE, 0, pdZ, randZ);
        assert(iSuccess == 1);
        swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
        swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
    }
    free(pdZ);
    free(randZ);

    return NULL;    
}




//Please note: Whenever we type-cast to (int), we add 0.5 to ensure that the value is rounded to the correct number. 
//For instance, if X/Y = 0.999 then (int) (X/Y) will equal 0 and not 1 (as (int) rounds down).
//Adding 0.5 ensures that this does not happen. Therefore we use (int) (X/Y + 0.5); instead of (int) (X/Y);

int main(int argc, char *argv[])
{
    int iSuccess = 0;
    int i,j;
    
    FTYPE **factors=NULL;

#ifdef ENABLE_MPI
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(myid==0)
    {
#endif

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
    printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n"); 
    fflush(NULL);
#else
    printf("PARSEC Benchmark Suite\n");
    fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_begin(__parsec_swaptions);
#endif

    if(argc == 1)
    {
        fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
        exit(1);
    }

    for (int j=1; j<argc; j++) {
        if (!strcmp("-sm", argv[j])) {NUM_TRIALS = atoi(argv[++j]);}
        else if (!strcmp("-nt", argv[j])) {nThreads = atoi(argv[++j]);} 
        else if (!strcmp("-ns", argv[j])) {nSwaptions = atoi(argv[++j]);} 
        else {
            fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
        }
    }

    if(nSwaptions < nThreads) {
        nSwaptions = nThreads; 
    }

    printf("Number of Simulations: %d,  Number of threads: %d Number of swaptions: %d\n", NUM_TRIALS, nThreads, nSwaptions);

#ifdef ENABLE_THREADS

#ifdef TBB_VERSION
    tbb::task_scheduler_init init(nThreads);
#else
    pthread_t      *threads;
    pthread_attr_t  pthread_custom_attr;

    if ((nThreads < 1) || (nThreads > MAX_THREAD))
    {
        fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
        exit(1);
    }
    threads = (pthread_t *) malloc(nThreads * sizeof(pthread_t));
    pthread_attr_init(&pthread_custom_attr);

#endif // TBB_VERSION

    if ((nThreads < 1) || (nThreads > MAX_THREAD))
    {
      fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
      exit(1);
    }

#else
    if (nThreads != 1)
    {
      fprintf(stderr,"Number of threads must be 1 (serial version)\n");
      exit(1);
    }
#endif //ENABLE_THREADS

#ifdef ENABLE_MPI
    } //myid==0
    printf("bcasting\n");
    MPI_Bcast(&nSwaptions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("bcast done\n");
#endif

    // initialize input dataset
    factors = dmatrix(0, iFactors-1, 0, iN-2);
    //the three rows store vol data for the three factors
    factors[0][0]= .01;
    factors[0][1]= .01;
    factors[0][2]= .01;
    factors[0][3]= .01;
    factors[0][4]= .01;
    factors[0][5]= .01;
    factors[0][6]= .01;
    factors[0][7]= .01;
    factors[0][8]= .01;
    factors[0][9]= .01;

    factors[1][0]= .009048;
    factors[1][1]= .008187;
    factors[1][2]= .007408;
    factors[1][3]= .006703;
    factors[1][4]= .006065;
    factors[1][5]= .005488;
    factors[1][6]= .004966;
    factors[1][7]= .004493;
    factors[1][8]= .004066;
    factors[1][9]= .003679;

    factors[2][0]= .001000;
    factors[2][1]= .000750;
    factors[2][2]= .000500;
    factors[2][3]= .000250;
    factors[2][4]= .000000;
    factors[2][5]= -.000250;
    factors[2][6]= -.000500;
    factors[2][7]= -.000750;
    factors[2][8]= -.001000;
    factors[2][9]= -.001250;

    // setting up multiple swaptions
    swaptions = 
#ifdef TBB_VERSION
    (parm *)memory_parm.allocate(sizeof(parm)*nSwaptions, NULL);
#else
    (parm *)malloc(sizeof(parm)*nSwaptions);
#endif

#ifdef ENABLE_MPI
    int chunksize = nSwaptions/numprocs;
    int beg = myid*chunksize;
    int end = (myid+1)*chunksize;
    if(myid == numprocs -1 )
        end = nSwaptions;
#else
    int beg = 0;
    int end = nSwaptions;
#endif

    int k;
    for (i = beg; i < end; i++) {
        swaptions[i].Id = i;
        swaptions[i].iN = iN;
        swaptions[i].iFactors = iFactors;
        swaptions[i].dYears = dYears;

        swaptions[i].dStrike =  (double)i / (double)nSwaptions; 
        swaptions[i].dCompounding =  0;
        swaptions[i].dMaturity =  1;
        swaptions[i].dTenor =  2.0;
        swaptions[i].dPaymentInterval =  1.0;

        swaptions[i].pdYield = dvector(0,iN-1);;
        swaptions[i].pdYield[0] = .1;
        for(j=1;j<=swaptions[i].iN-1;++j)
            swaptions[i].pdYield[j] = swaptions[i].pdYield[j-1]+.005;

        swaptions[i].ppdFactors = dmatrix(0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
        for(k=0;k<=swaptions[i].iFactors-1;++k)
            for(j=0;j<=swaptions[i].iN-2;++j)
                swaptions[i].ppdFactors[k][j] = factors[k][j];
        }


    // **********Calling the Swaption Pricing Routine*****************
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

#ifdef ENABLE_THREADS

#ifdef TBB_VERSION
    Worker w;
    tbb::parallel_for(tbb::blocked_range<int>(0,nSwaptions,TBB_GRAINSIZE),w);
#else

    int threadIDs[nThreads];
    for (i = 0; i < nThreads; i++) {
        threadIDs[i] = i;
        pthread_create(&threads[i], &pthread_custom_attr, worker, &threadIDs[i]);
    }
    for (i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);

#endif // TBB_VERSION   

#else
    #ifdef ENABLE_MPI
        printf("%d working\n", myid);
        worker(&myid);
        printf("%d done\n", myid);
        const int dest = 0;
        const int tag = 1000;
        int idx=0;
        /*
        MPI_Status status;

        for(int i=nSwaptions/numprocs; i<nSwaptions; i++)
        {
            if(myid==(i/(end-beg))){
                MPI_Sendrecv(&swaptions[i].dSimSwaptionMeanPrice, 1, MPI_DOUBLE, dest, tag, &swaptions[i].dSimSwaptionMeanPrice, 1, MPI_DOUBLE, (i/(end-beg)), tag, MPI_COMM_WORLD, &status);
                MPI_Sendrecv(&swaptions[i].dSimSwaptionStdError, 1, MPI_DOUBLE, dest, tag, &swaptions[i].dSimSwaptionStdError, 1, MPI_DOUBLE, (i/(end-beg)), tag, MPI_COMM_WORLD, &status);                
            }
        }
*/
        printf("start sendrecv\n");
        //TODO: change to multi device opencl
         if(myid==0){
             //MPI_recv()x2x(numprocs-end)
             MPI_Request request[2*(nSwaptions-end)];
             MPI_Status status[2*(nSwaptions-end)];
             for(int i=end; i<nSwaptions; i++){
                 printf("i: %d\n", i);
                 MPI_Irecv(&swaptions[i].dSimSwaptionMeanPrice, 1, MPI_DOUBLE, i/(end-beg), tag, MPI_COMM_WORLD, &request[idx++]);
                 MPI_Irecv(&swaptions[i].dSimSwaptionStdError, 1, MPI_DOUBLE, i/(end-beg), tag, MPI_COMM_WORLD, &request[idx++]);                
             }
             printf("waiting\n");
             MPI_Waitall(2*(nSwaptions-end), request, status);
             printf("done waiting\n");

         }
         else{
             //MPI_send()x2x(beg-end)
             MPI_Request request[2*(end-beg)];
             MPI_Status status[2*(end-beg)];
             for(int i=beg; i<end; i++){
                 MPI_Isend(&swaptions[i].dSimSwaptionMeanPrice, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &request[idx++]);
                 MPI_Isend(&swaptions[i].dSimSwaptionStdError, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &request[idx++]);
             }
             MPI_Waitall(2*(end-beg), request, status);
         }
    #else
        int threadID=0;
        worker(&threadID);
    #endif
#endif //ENABLE_THREADS

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif


#ifdef ENABLE_MPI
    if(myid==0)
    {
#endif

    for (i = 0; i < nSwaptions; i++) {
        fprintf(stderr,"Swaption%d: [SwaptionPrice: %.10lf StdError: %.10lf] \n", 
            i, swaptions[i].dSimSwaptionMeanPrice, swaptions[i].dSimSwaptionStdError);

    }

#ifdef ENABLE_MPI
    }  //myid==0
#endif

    for (i = beg; i < end; i++) {
        free_dvector(swaptions[i].pdYield, 0, swaptions[i].iN-1);
        free_dmatrix(swaptions[i].ppdFactors, 0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
    }


#ifdef TBB_VERSION
    memory_parm.deallocate(swaptions, sizeof(parm));
#else
    free(swaptions);
#endif // TBB_VERSION

    //***********************************************************

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

#ifdef ENABLE_MPI
    MPI_Finalize();
#endif

    return iSuccess;
}
