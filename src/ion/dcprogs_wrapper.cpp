#include <iostream>
#include "limits.h"
#include "math.h"
#include "likelihood.h"
#include <gmcmc/gmcmc_ion_missed_events.h>

extern "C" int dcprogs_likelihood(const gmcmc_burst_dataset * dataset, double * qmatrix, int k , size_t ldq,int kA, double tau, double tcrit, size_t  set_no, int useChs, double * likelihood) {
    //first, unwrap the data
   

    DCProgs::t_Bursts bursts;
    
    size_t m = gmcmc_dataset_get_num_bursts(dataset,set_no);
    double burst_time = 0;
    int interval_count=0;
    //printf("Set %zu has %zu bursts\n", set_no, m);
    for (size_t b = 0; b < m; b++){
        size_t * burst_length = gmcmc_dataset_get_burst_number(dataset,b,set_no);
        //printf("Burst %zu -> %zu elements\n", set_no, *burst_length);
 
        //put the bursts in the vector format required by dcprogs
        
        DCProgs::t_Burst dburst;
        size_t elem;
        const double * burst = gmcmc_dataset_get_burst(dataset , b, set_no); 
        for ( elem = 0; elem < *burst_length; elem++){
            dburst.push_back(burst[elem]);
            burst_time+=burst[elem];
            interval_count++;
            //printf("za%f\n", burst[elem]);    
        }
        bursts.push_back(dburst);
        free(burst_length); //instantiated with malloc in the C code
    }

    //fill in the q_matrix
    //printf("In wrapper: ldq -> %zu \n",ldq);
    DCProgs::t_rmatrix matrix(k , k);
    for (int i=0; i < k; i++){
        for (int j=0; j < k;j++){
            matrix(i,j) = qmatrix[(j*ldq)+i];
            //printf("[%d][%d] = %.16f\n",i,j,matrix(i,j));
        }
    }
    // printf("End of Q\n");
    if (! useChs > 0)
    	tcrit=-1;//-tcrit;        

    DCProgs::Log10Likelihood dc_likelihood(bursts, kA, tau, tcrit);
    DCProgs::t_real result = 0;
    int error = 0;
    try {
        result = dc_likelihood(matrix);
    }
    catch (std::exception& e) {
        printf("Exception thrown\n");
        printf(e.what());
        printf("Q-matrix in dcprogs\n");
        std::cout << matrix << std::endl;
        error = -1;
    }

    if (fabs(result) == std::numeric_limits<double>::infinity() || std::isnan(result)){
        error=-1;
    }

    *likelihood = result*log(10);
    /*printf ("parsed bursts %zu\n", bursts.size());
    printf("t-crit %.10f\n",tcrit);
    printf("t-res %.10f\n",tau);
    printf("kA %i\n",kA);
    printf("Burst time %.16f\n",burst_time);
    printf("Interval count %i\n",interval_count);
    printf("Likelihood %.17f\n",*likelihood);
    */    
    return error;
}
