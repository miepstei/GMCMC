//
//  ION_TwoState_PopMCMC_MH.c
//  
//
//  Created by Michael Epstein on 05/03/2014.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <getopt.h>
#include <time.h>

#include <sys/time.h>

#include <mpi.h>

#include <gmcmc/gmcmc_ion.h>
#include <gmcmc/gmcmc_popmcmc.h>
#include <gmcmc/gmcmc_ion_missed_events.h>
#include <gmcmc/gmcmc_proposal.h>

#include "common.h"

#define MPI_ERROR_CHECK(call, msg) \
do { \
int error = (call); \
if (error != MPI_SUCCESS) { \
fprintf(stderr, "%s\n%s returned %d in %s (%s:%d):\n", \
msg, #call, error, __func__, __FILE__, __LINE__); \
char string[MPI_MAX_ERROR_STRING]; \
int length; \
int errorerror = MPI_Error_string(error, string, &length); \
if (errorerror != MPI_SUCCESS) \
fprintf(stderr, "\tadditionally, MPI_Error_string returned %d when looking up the error code\n", errorerror);\
else \
fprintf(stderr, "\t%s\n", string); \
return error; \
} \
} while (false)

struct ion_args {
  bool log10space;
  bool precondition;
};

static int parse_extra(int c, const char * optarg, void * extra) {
  (void)optarg;
  struct ion_args * args = (struct ion_args *)extra;
  switch (c) {
    case 1000:
      args->log10space = true;
      return 0;
    case 1001:
      args->precondition = true;
      return 0;
  }
  return '?';
}

static void print_extra(FILE * stream) {
  fprintf(stream, "Ion Channel options:\n");
  fprintf(stream, "  --log10space  infer the parameters in log space\n");
  fprintf(stream, "  --precondition   precondition the RWMH proposal\n");
}

static struct ion_args extra;   // extra arguments are used in calculate_Q_matrix so need to be file scope
static void precondition_covariance(double *, size_t );

/**
 * Calculates the Q matrix from the current parameter values.
 *
 * @param [in]  params  the parameter values
 * @param [out] Q       the Q matrix, initialised to zero
 * @param [in]  ldq     the leading dimension of the Q matrix
 */
static void calculate_Q_matrix(const double *, double *, size_t, double);

int main(int argc, char * argv[]) {
    // Since we are using MPI for parallel processing initialise it here before
    // parsing the arguments for our program
    MPI_ERROR_CHECK(MPI_Init(&argc, &argv), "Failed to initialise MPI");
    
    // Handle MPI errors ourselves
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    
    // Get the MPI process ID and number of cores
    int rank, size;
    MPI_ERROR_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Unable to get MPI rank");
    MPI_ERROR_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), "Unable to get MPI communicator size");
    
    // Default dataset file
    const char * data_file = "data/Ach_Synthetic_Burst_Data.mat";
    extra.log10space = false;
    extra.precondition = false;
    
    struct option ext_longopts[] = {
      { "log10space", no_argument, NULL, 1000 },{"precondition" , no_argument,NULL, 1001},
      { NULL, 0, NULL, 0 }
    };

    /*
     * Set up default MCMC options
     */
    gmcmc_popmcmc_options mcmc_options;
    
    // Set number of tempered distributions to use
    mcmc_options.num_temperatures = 5;
    
    // Set number of burn-in and posterior samples
    mcmc_options.num_burn_in_samples   = 10000;
    mcmc_options.num_posterior_samples = 10000;
    
    // Set iteration interval for adapting stepsizes
    mcmc_options.adapt_rate            =  25;
    mcmc_options.upper_acceptance_rate =   0.5;
    mcmc_options.lower_acceptance_rate =   0.1;
    
    // Callbacks
    mcmc_options.acceptance = acceptance_monitor;
    mcmc_options.burn_in_writer = NULL;
    mcmc_options.posterior_writer = NULL;
    
  int error;
  size_t num_blocks = 0, * block_sizes = NULL, * blocks = NULL;
  if ((error = parse_options(argc, argv, NULL, ext_longopts,
                             parse_extra, print_extra, &extra,
                             &mcmc_options, &data_file,
                             &num_blocks, &block_sizes, &blocks)) != 0) {
        free(block_sizes);
        free(blocks);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return error;
    }
   
    double * temperatures = malloc(mcmc_options.num_temperatures * sizeof(double));
    if (temperatures == NULL) {
        free(block_sizes);
        free(blocks);
        fputs("Unable to allocate temperature schedule\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -2;
    }

    // Avoid divide by zero in temperature scale
    if (mcmc_options.num_temperatures == 1)
        temperatures[0] = 1.0;
    else {
        for (unsigned int i = 0; i < mcmc_options.num_temperatures; i++)
            temperatures[i] = pow(i * (1.0 / (mcmc_options.num_temperatures - 1.0)), 5.0);
    }
    mcmc_options.temperatures = temperatures;
    
    // Print out MCMC options on node 0
    if (rank == 0) {
        fprintf(stdout, "Number of cores: %d\n", size);
        print_options(stdout, &mcmc_options);
        print_blocks(stdout, num_blocks, block_sizes, blocks);
    }
    
    
    /*
     * Model settings
     */
    const unsigned int num_params = 10;
    
    // Set up priors for each of the parameters
    gmcmc_distribution ** priors;
    if ((priors = malloc(num_params * sizeof(gmcmc_distribution *))) == NULL) {
        fputs("Failed to allocate space for priors\n", stderr);
        free(temperatures);
        free(block_sizes);
        free(blocks);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -2;
    }

    if(extra.log10space){
        // Set up priors for log space
        fputs("Unable to create priors (Unimplemented)\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -3;
    } else {
        // Set up priors for standard space
        //gamma parameterisation is 1/\alpha, 1/\beta...
        double nap = (1.0e+6)/4;
        double ap = (1.0e+10)/4;

        double max_priors[] = {nap, nap , nap , nap , nap, nap, ap , nap , nap , ap };
        for (unsigned int i = 0; i < num_params; i++) {
            if ((error = gmcmc_distribution_create_gamma(&priors[i], 1, max_priors[i])) != 0) {
                // Clean up
                for (unsigned int j = 0; j < i; j++)
                    gmcmc_distribution_destroy(priors[i]);
                
                free(priors);
                free(temperatures);
                free(block_sizes);
                free(blocks);
                fputs("Unable to create priors\n", stderr);
                MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
                return -3;
            }
        }
    }
    
    // Load the dataset
    gmcmc_burst_dataset * dataset;
    if ((error = gmcmc_dataset_create_matlab_ion_burst(&dataset, data_file)) != 0) {
        // Clean up
        for (unsigned int i = 0; i < num_params; i++)
            gmcmc_distribution_destroy(priors[i]);
        free(priors);
        free(temperatures);
        free(block_sizes);
        free(blocks);
        fputs("Unable to load dataset\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -4;
    }
    
    // Create the model
    gmcmc_model * model;
    if ((error = gmcmc_model_create(&model, num_params, priors)) != 0) {
        // Clean up
        for (unsigned int i = 0; i < num_params; i++)
            gmcmc_distribution_destroy(priors[i]);
        free(priors);
        free(block_sizes);
        free(blocks);
        free(temperatures);
        gmcmc_burst_dataset_destroy(dataset);
        fputs("Unable to create model\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -4;
    }
    
    // Priors have been copied into model so don't need them here any more
    for (unsigned int i = 0; i < num_params; i++)
        gmcmc_distribution_destroy(priors[i]);
    free(priors);
   


    if (num_blocks > 0) {
        if ((error = gmcmc_model_set_blocking(model, num_blocks, block_sizes)) != 0) {
            free(temperatures);
            free(block_sizes);
            free(blocks);
            gmcmc_burst_dataset_destroy(dataset);
            fputs("Unable to set blocking\n", stderr);
            MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
            return error;
        }
        free(block_sizes);

        if ((error = gmcmc_model_set_blocks(model, blocks)) != 0) {
            free(temperatures);
            free(blocks);
            gmcmc_burst_dataset_destroy(dataset);
            fputs("Unable to set blocks\n", stderr);
            MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
            return error;
        }
        free(blocks);
    }
 
 
    // Set up starting values for all temperatures
    if (extra.log10space) {
        const double params[] = { 3.321159,4.667956,3.906631,0.058638,5.074879,2.148782,7.590858,2.133920,4.228501,7.856480  };
        if ((error = gmcmc_model_set_params(model, params)) != 0) {
            // Clean up
            free(temperatures);
            gmcmc_burst_dataset_destroy(dataset);
            gmcmc_model_destroy(model);
            fputs("Unable to set initial parameter values\n", stderr);
            MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
            return -5;
        }
    }
    else {
        const double params[] = { 2094.878953,46553.944113,8065.490054,1.144559,118817.180140,140.858203,38981469.780476,136.119501,16923.924772,71858735.800358 };
        if ((error = gmcmc_model_set_params(model, params)) != 0) {
            // Clean up
            free(temperatures);
            gmcmc_burst_dataset_destroy(dataset);
            gmcmc_model_destroy(model);
            fputs("Unable to set initial parameter values\n", stderr);
            MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
            return -5;
        }
    }

   
    // Sea the step size
    gmcmc_model_set_stepsize(model, 1);
    gmcmc_model_set_stepsize_bounds(model, 1.0e-08, 1.0e+05);

    /*
     * ION model settings
     */
    gmcmc_ion_model * ion_model;
    if ((error = gmcmc_ion_model_create(&ion_model, 4, 3, calculate_Q_matrix, precondition_covariance)) != 0) {
        // Clean up
        for (unsigned int i = 0; i < num_params; i++)
            gmcmc_distribution_destroy(priors[i]);
        free(priors);
        free(temperatures);
        gmcmc_burst_dataset_destroy(dataset);
        gmcmc_model_destroy(model);
        fputs("Unable to create Ion Channel specific model\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -4;
    }
    
    gmcmc_model_set_modelspecific(model, ion_model);
   

    if (rank == 0) {
      if (optind + 1 < argc) {
        if ((error = gmcmc_filewriter_create_hdf5(&mcmc_options.burn_in_writer,
                                                argv[optind++], num_params, 1, 
                                                mcmc_options.num_temperatures,
                                                mcmc_options.num_burn_in_samples)) != 0) {
          free(temperatures);
          gmcmc_burst_dataset_destroy(dataset);
          gmcmc_model_destroy(model);
          gmcmc_ion_model_destroy(ion_model);
          fputs("Unable to create hdf5 burn-in writer\n", stderr);
          MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
          return -5;
        }
      }
      if ((error = gmcmc_filewriter_create_hdf5(&mcmc_options.posterior_writer,
                                              argv[optind], num_params, 1, 
                                              mcmc_options.num_temperatures,
                                              mcmc_options.num_posterior_samples)) != 0) {

        free(temperatures);
        gmcmc_burst_dataset_destroy(dataset);
        gmcmc_model_destroy(model);
        gmcmc_ion_model_destroy(ion_model);
        gmcmc_filewriter_destroy(mcmc_options.burn_in_writer);
        fputs("Unable to create hdf5 posterior writer\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -5;
      }
    }
    /*
     * Create a parallel random number generator to use
     */
    gmcmc_prng64 * rng;
    if ((error = gmcmc_prng64_create(&rng, gmcmc_prng64_dcmt607, rank)) != 0) {
        // Clean up
        free(temperatures);
        gmcmc_burst_dataset_destroy(dataset);
        gmcmc_model_destroy(model);
        gmcmc_ion_model_destroy(ion_model);
        gmcmc_filewriter_destroy(mcmc_options.burn_in_writer);
        gmcmc_filewriter_destroy(mcmc_options.posterior_writer);
        fputs("Unable to create parallel RNG\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -5;
    }
    
    // Seed the RNG
    time_t seed = time(NULL);
    gmcmc_prng64_seed(rng, seed);
    fprintf(stdout, "Using PRNG seed: %ld\n", seed);
    // Start timer
    struct timeval start, stop;
    if (rank == 0)  {
        if (gettimeofday(&start, NULL) != 0) {
            fputs("gettimeofday failed\n", stderr);
            return -6;
        }
    }
    
    /*
     * Call main population MCMC routine using MPI
     */
    error = gmcmc_popmcmc_mpi(model, dataset, gmcmc_ion_missed_events_preconditioned_mh, gmcmc_proposal_simp_mmala, &mcmc_options, rng);
   

 
    if (rank == 0) {
        // Stop timer
        if (gettimeofday(&stop, NULL) != 0) {
            fputs("gettimeofday failed\n", stderr);
            return -7;
        }
        
        double time = ((double)(stop.tv_sec - start.tv_sec) +
                       (double)(stop.tv_usec - start.tv_usec) * 1.e-6);
        
        fprintf(stdout, "Simulation took %.3f seconds\n", time);
    }
    
    // Clean up (dataset, model, rng)
    free(temperatures);
    gmcmc_burst_dataset_destroy(dataset);
    gmcmc_model_destroy(model);
    gmcmc_ion_model_destroy(ion_model);
    gmcmc_filewriter_destroy(mcmc_options.burn_in_writer);
    gmcmc_filewriter_destroy(mcmc_options.posterior_writer);
    gmcmc_prng64_destroy(rng);
    
    MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
    
    return error;
}



static void calculate_Q_matrix(const double * params, double * Q, size_t ldq, double conc){//, double conc) {
    double Alpha_2,Beta_2,Alpha_1a,Beta_1a,Alpha_1b,Beta_1b,km_2a,km_1b,km_2b,kp_2b; 
    // Rename for clarity
    if (extra.log10space) {
        Alpha_2   = pow(10.0, params[0]);       // Alpha
        Beta_2    = pow(10.0, params[1]);       // Beta
        Alpha_1a  = pow(10.0, params[2]);
        Beta_1a   = pow(10.0, params[3]);
        Alpha_1b  = pow(10.0, params[4]);
        Beta_1b   = pow(10.0, params[5]);
        km_2a     = pow(10.0, params[6]);
        km_1b     = pow(10.0, params[7]);
        km_2b     = pow(10.0, params[8]);
        kp_2b     = pow(10.0, params[9]);
    }
    else{
        Alpha_2   = params[0];       // Alpha
        Beta_2    = params[1];       // Beta
        Alpha_1a  = params[2];
        Beta_1a   = params[3];
        Alpha_1b  = params[4];
        Beta_1b   = params[5];
        km_2a     = params[6];
        km_1b     = params[7];
        km_2b     = params[8];
        kp_2b     = params[9];
    }

    //param array defined as follows
    
    //1  alpha2
    //2  beta2
    //3  alpha1a
    //4  beta1a
    //5  alpha1b
    //6  beta1b
    //7  k_{-2}a
    //8  k_{-1}b
    //9  k_{-2}b
    //10 k_{+2}b
    
    //  k_{+2}a = 100,000,000
    // k_{-1}a = k_{-2}a (7)
    // k_{-1}b = k_{-2}b (9)
    // k_{+1}b = k_{+2}b (10)
    
    // k_{+1}a is set by mr (8*9*11*13)/(12*9*7)
    //printf ("parameters %.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",Alpha_2,Beta_2,Alpha_1a,Beta_1a,Alpha_1b,Beta_1b,km_2a,km_1b,km_2b,kp_2b);    
    Q[0 * ldq + 0] = -Alpha_2;
    Q[1 * ldq + 0] = 0;
    Q[2 * ldq + 0] = 0;
    Q[3 * ldq + 0] = Alpha_2;
    Q[4 * ldq + 0] = 0;
    Q[5 * ldq + 0] = 0;
    Q[6 * ldq + 0] = 0;
    
    Q[0*  ldq + 1] = 0;
    Q[1 * ldq + 1] = -Alpha_1a;
    Q[2 * ldq + 1] = 0;
    Q[3 * ldq + 1] = 0;
    Q[4 * ldq + 1] = Alpha_1a;
    Q[5 * ldq + 1] = 0;
    Q[6 * ldq + 1] = 0;
    
    Q[0 * ldq + 2] = 0;
    Q[1 * ldq + 2] = 0;
    Q[2 * ldq + 2] = -Alpha_1b;
    Q[3 * ldq + 2] = 0;
    Q[4 * ldq + 2] = 0;
    Q[5 * ldq + 2] = Alpha_1b;
    Q[6 * ldq + 2] = 0;

    Q[0 * ldq + 3] = Beta_2;
    Q[1 * ldq + 3] = 0;
    Q[2 * ldq + 3] = 0;
    Q[3 * ldq + 3] = -(Beta_2+km_2b+km_1b);
    Q[4 * ldq + 3] = km_2b;
    Q[5 * ldq + 3] = km_1b;
    Q[6 * ldq + 3] = 0;
    
    Q[0 * ldq + 4] = 0;
    Q[1 * ldq + 4] = Beta_1a; 
    Q[2 * ldq + 4] = 0;
    Q[3 * ldq + 4] = kp_2b * conc;
    Q[5 * ldq + 4] = 0;
    Q[6 * ldq + 4] = km_1b;
    Q[4 * ldq + 4] = -(Q[1 * ldq + 4]+ Q[3 * ldq + 4]+Q[6 * ldq + 4]);
    
    Q[0* ldq +5] = 0;
    Q[1* ldq +5] = 0;
    Q[2* ldq +5] = Beta_1b;
    Q[3* ldq +5] = km_2a * conc;
    Q[4* ldq +5] = 0;
    Q[6* ldq +5] = km_2b;
    Q[5* ldq +5] = -(Q[2* ldq +5] +  Q[3* ldq +5] + Q[6* ldq +5]);
    
    Q[0* ldq +6] = 0;
    Q[1* ldq +6] = 0;
    Q[2* ldq +6] = 0;
    Q[3* ldq +6] = 0;
    Q[4* ldq +6] = conc*((km_2a)*km_2b*(km_1b)*(kp_2b))/(km_2b*(kp_2b)*km_1b); //mr
    Q[5* ldq +6] = kp_2b * conc;
    Q[6* ldq +6] = -( Q[4* ldq +6]+Q[5* ldq +6]);

}

static void precondition_covariance(double * M, size_t ldq) {
    if (extra.precondition ){
        //return preconditioned matrix covariance^-1
        M[0 * ldq + 0] = 0.0025108903686987;
        M[0 * ldq + 1] = -0.0001001788709362;
        M[0 * ldq + 2] = -0.0000369550808347;
        M[0 * ldq + 3] = 0.0060144800092579;
        M[0 * ldq + 4] = -0.0000007772704043;
        M[0 * ldq + 5] = 0.0001731813894352;
        M[0 * ldq + 6] = -0.0000000001518952;
        M[0 * ldq + 7] = 0.0000767094805138;
        M[0 * ldq + 8] = 0.0000867888501996;
        M[0 * ldq + 9] = -0.0000000007034620;
        M[1 * ldq + 0] = -0.0001001788709362;
        M[1 * ldq + 1] = 0.0000048042038109;
        M[1 * ldq + 2] = 0.0000008150753179;
        M[1 * ldq + 3] = -0.0003041257086398;
        M[1 * ldq + 4] = 0.0000000727890661;
        M[1 * ldq + 5] = -0.0000194386486973;
        M[1 * ldq + 6] = 0.0000000000228288;
        M[1 * ldq + 7] = -0.0000027016352759;
        M[1 * ldq + 8] = -0.0000015974720156;
        M[1 * ldq + 9] = 0.0000000000327345;
        M[2 * ldq + 0] = -0.0000369550808347;
        M[2 * ldq + 1] = 0.0000008150753179;
        M[2 * ldq + 2] = 0.0000780967394652;
        M[2 * ldq + 3] = -0.0004043346871928;
        M[2 * ldq + 4] = -0.0000029353087034;
        M[2 * ldq + 5] = 0.0007809654060073;
        M[2 * ldq + 6] = -0.0000000006764287;
        M[2 * ldq + 7] = 0.0000651868471494;
        M[2 * ldq + 8] = -0.0000171695006113;
        M[2 * ldq + 9] = 0.0000000000542796;
        M[3 * ldq + 0] = 0.0060144800092579;
        M[3 * ldq + 1] = -0.0003041257086398;
        M[3 * ldq + 2] = -0.0004043346871928;
        M[3 * ldq + 3] = 0.7219989478734024;
        M[3 * ldq + 4] = -0.0000524636984310;
        M[3 * ldq + 5] = -0.0136638561285591;
        M[3 * ldq + 6] = 0.0000000146616855;
        M[3 * ldq + 7] = -0.0066557294388111;
        M[3 * ldq + 8] = 0.0019143215916367;
        M[3 * ldq + 9] = -0.0000000745715321;
        M[4 * ldq + 0] = -0.0000007772704043;
        M[4 * ldq + 1] = 0.0000000727890661;
        M[4 * ldq + 2] = -0.0000029353087034;
        M[4 * ldq + 3] = -0.0000524636984310;
        M[4 * ldq + 4] = 0.0000002973086960;
        M[4 * ldq + 5] = -0.0000775320019539;
        M[4 * ldq + 6] = 0.0000000000672769;
        M[4 * ldq + 7] = -0.0000073344280293;
        M[4 * ldq + 8] = 0.0000005870662778;
        M[4 * ldq + 9] = 0.0000000000074507;
        M[5 * ldq + 0] = 0.0001731813894352;
        M[5 * ldq + 1] = -0.0000194386486973;
        M[5 * ldq + 2] = 0.0007809654060073;
        M[5 * ldq + 3] = -0.0136638561285591;
        M[5 * ldq + 4] = -0.0000775320019539;
        M[5 * ldq + 5] = 0.0333639928891791;
        M[5 * ldq + 6] = -0.0000000290188919;
        M[5 * ldq + 7] = 0.0031544764329331;
        M[5 * ldq + 8] = -0.0003291927023133;
        M[5 * ldq + 9] = 0.0000000017002396;
        M[6 * ldq + 0] = -0.0000000001518952;
        M[6 * ldq + 1] = 0.0000000000228288;
        M[6 * ldq + 2] = -0.0000000006764287;
        M[6 * ldq + 3] = 0.0000000146616855;
        M[6 * ldq + 4] = 0.0000000000672769;
        M[6 * ldq + 5] = -0.0000000290188919;
        M[6 * ldq + 6] = 0.0000000000000404;
        M[6 * ldq + 7] = -0.0000000040349163;
        M[6 * ldq + 8] = 0.0000000002343996;
        M[6 * ldq + 9] = 0.0000000000000033;
        M[7 * ldq + 0] = 0.0000767094805138;
        M[7 * ldq + 1] = -0.0000027016352759;
        M[7 * ldq + 2] = 0.0000651868471494;
        M[7 * ldq + 3] = -0.0066557294388111;
        M[7 * ldq + 4] = -0.0000073344280293;
        M[7 * ldq + 5] = 0.0031544764329331;
        M[7 * ldq + 6] = -0.0000000040349163;
        M[7 * ldq + 7] = 0.0007862072290830;
        M[7 * ldq + 8] = 0.0000058413308315;
        M[7 * ldq + 9] = -0.0000000006448045;
        M[8 * ldq + 0] = 0.0000867888501996;
        M[8 * ldq + 1] = -0.0000015974720156;
        M[8 * ldq + 2] = -0.0000171695006113;
        M[8 * ldq + 3] = 0.0019143215916367;
        M[8 * ldq + 4] = 0.0000005870662778;
        M[8 * ldq + 5] = -0.0003291927023133;
        M[8 * ldq + 6] = 0.0000000002343996;
        M[8 * ldq + 7] = 0.0000058413308315;
        M[8 * ldq + 8] = 0.0000413824977977;
        M[8 * ldq + 9] = -0.0000000003257449;
        M[9 * ldq + 0] = -0.0000000007034620;
        M[9 * ldq + 1] = 0.0000000000327345;
        M[9 * ldq + 2] = 0.0000000000542796;
        M[9 * ldq + 3] = -0.0000000745715321;
        M[9 * ldq + 4] = 0.0000000000074507;
        M[9 * ldq + 5] = 0.0000000017002396;
        M[9 * ldq + 6] = 0.0000000000000033;
        M[9 * ldq + 7] = -0.0000000006448045;
        M[9 * ldq + 8] = -0.0000000003257449;
        M[9 * ldq + 9] = 0.0000000000000147;
    } else {
        //return identity matrix
        M[0 * ldq + 0] = 1;
        M[1 * ldq + 0] = 0;
        M[2 * ldq + 0] = 0;
        M[3 * ldq + 0] = 0;
        M[4 * ldq + 0] = 0;
        M[5 * ldq + 0] = 0;
        M[6 * ldq + 0] = 0;
        M[7 * ldq + 0] = 0;
        M[8 * ldq + 0] = 0;
        M[9 * ldq + 0] = 0;

        M[0 * ldq + 1] = 0;
        M[1 * ldq + 1] = 1;
        M[2 * ldq + 1] = 0;
        M[3 * ldq + 1] = 0;
        M[4 * ldq + 1] = 0;
        M[5 * ldq + 1] = 0;
        M[6 * ldq + 1] = 0;
        M[7 * ldq + 1] = 0;
        M[8 * ldq + 1] = 0;
        M[9 * ldq + 1] = 0;

        M[0 * ldq + 2] = 0;
        M[1 * ldq + 2] = 0;
        M[2 * ldq + 2] = 1;
        M[3 * ldq + 2] = 0;
        M[4 * ldq + 2] = 0;
        M[5 * ldq + 2] = 0;
        M[6 * ldq + 2] = 0;
        M[7 * ldq + 2] = 0;
        M[8 * ldq + 2] = 0;
        M[9 * ldq + 2] = 0;

        M[0 * ldq + 3] = 0;
        M[1 * ldq + 3] = 0;
        M[2 * ldq + 3] = 0;
        M[3 * ldq + 3] = 1;
        M[4 * ldq + 3] = 0;
        M[5 * ldq + 3] = 0;
        M[6 * ldq + 3] = 0;
        M[7 * ldq + 3] = 0;
        M[8 * ldq + 3] = 0;
        M[9 * ldq + 3] = 0;

        M[0 * ldq + 4] = 0;
        M[1 * ldq + 4] = 0;
        M[2 * ldq + 4] = 0;
        M[3 * ldq + 4] = 0;
        M[4 * ldq + 4] = 1;
        M[5 * ldq + 4] = 0;
        M[6 * ldq + 4] = 0;
        M[7 * ldq + 4] = 0;
        M[8 * ldq + 4] = 0;
        M[9 * ldq + 4] = 0;

        M[0 * ldq + 5] = 0;
        M[1 * ldq + 5] = 0;
        M[2 * ldq + 5] = 0;
        M[3 * ldq + 5] = 0;
        M[4 * ldq + 5] = 0;
        M[5 * ldq + 5] = 1;
        M[6 * ldq + 5] = 0;
        M[7 * ldq + 5] = 0;
        M[8 * ldq + 5] = 0;
        M[9 * ldq + 5] = 0;

        M[0 * ldq + 6] = 0;
        M[1 * ldq + 6] = 0;
        M[2 * ldq + 6] = 0;
        M[3 * ldq + 6] = 0;
        M[4 * ldq + 6] = 0;
        M[5 * ldq + 6] = 0;
        M[6 * ldq + 6] = 1;
        M[7 * ldq + 6] = 0;
        M[8 * ldq + 6] = 0;
        M[9 * ldq + 6] = 0;

        M[0 * ldq + 7] = 0;
        M[1 * ldq + 7] = 0;
        M[2 * ldq + 7] = 0;
        M[3 * ldq + 7] = 0;
        M[4 * ldq + 7] = 0;
        M[5 * ldq + 7] = 0;
        M[6 * ldq + 7] = 0;
        M[7 * ldq + 7] = 1;
        M[8 * ldq + 7] = 0;
        M[9 * ldq + 7] = 0;

        M[0 * ldq + 8] = 0;
        M[1 * ldq + 8] = 0;
        M[2 * ldq + 8] = 0;
        M[3 * ldq + 8] = 0;
        M[4 * ldq + 8] = 0;
        M[5 * ldq + 8] = 0;
        M[6 * ldq + 8] = 0;
        M[7 * ldq + 8] = 0;
        M[8 * ldq + 8] = 1;
        M[9 * ldq + 8] = 0;

        M[0 * ldq + 9] = 0;
        M[1 * ldq + 9] = 0;
        M[2 * ldq + 9] = 0;
        M[3 * ldq + 9] = 0;
        M[4 * ldq + 9] = 0;
        M[5 * ldq + 9] = 0;
        M[6 * ldq + 9] = 0;
        M[7 * ldq + 9] = 0;
        M[8 * ldq + 9] = 0;
        M[9 * ldq + 9] = 1;
    }
}



