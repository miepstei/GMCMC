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


/**
 * Calculates the Q matrix from the current parameter values.
 *
 * @param [in]  params  the parameter values
 * @param [out] Q       the Q matrix, initialised to zero
 * @param [in]  ldq     the leading dimension of the Q matrix
 */
static void calculate_Q_matrix(const double *, double *, size_t, double);
static void precondition_covariance(double *, size_t );

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
    const unsigned int num_params = 4;
    
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
        fputs("Unable to create priors (unimplemented)\n", stderr);
        MPI_ERROR_CHECK(MPI_Finalize(), "Failed to shut down MPI");
        return -3;
    } else {
        // Set up priors for standard space
        double max_priors[] = {log((1.0e+6)/4),log((1.0e+6)/4),log((1.0e+6)/4), log((1.0e+10)/4)};
        for (unsigned int i = 0; i < num_params; i++) {
            if ((error = gmcmc_distribution_create_lognormal(&priors[i], max_priors[i] , 1)) != 0) { 
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
        const double params[] = { 3.32565993, 4.63287556, 4.27906498, 7.74135215 };
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
        const double params[] = {2116.70301082,42941.33642377,19013.62750450,55125450.53199392 };
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
    if ((error = gmcmc_ion_model_create(&ion_model, 2, 1, calculate_Q_matrix , precondition_covariance)) != 0) {
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
    double Alpha,Beta,k_off,k_on; 
    // Rename for clarity
    if (extra.log10space) {
        Alpha    = pow(10.0, params[0]);       // Alpha
        Beta     = pow(10.0, params[1]);       // Beta
        k_off    = pow(10.0, params[2]);       // disassociation rate constant
        k_on     = pow(10.0, params[3]);       // association rate constant

    }
    else{
        Alpha    = params[0];  // Alpha
        Beta     = params[1];  // Beta
        k_off    = params[2];
        k_on     = params[3]; 
    }
    
    //printf("params: %.8f,%.8f\n",params[0],params[1]);    
    // Construct Q matrix - [1] is open state, [2] is closed, as per convention
    // Q_AA - 1 by 1
    Q[0 * ldq + 0] = -Alpha;
    Q[1 * ldq + 0] = Alpha;
    Q[2 * ldq + 0] = 0;

    Q[0 * ldq + 1] = Beta;
    Q[1 * ldq + 1] = -(Beta+k_off);
    Q[2 * ldq + 1] = k_off;

    Q[0 * ldq + 2] = 0;
    Q[1 * ldq + 2] = k_on * conc;
    Q[2 * ldq + 2] = -(k_on * conc);
}

static void precondition_covariance(double * M, size_t ldq) {
    if (extra.precondition ){
        //return preconditioned matrix covariance^-1

        M[0 * ldq + 0] = 0.0032143140361692;
        M[0 * ldq + 1] = -0.0001288943074150;
        M[0 * ldq + 2] = 0.0000852542882848;
        M[0 * ldq + 3] = -0.0000000001755091;
        M[1 * ldq + 0] = -0.0001288943074150;
        M[1 * ldq + 1] = 0.0000059861886513;
        M[1 * ldq + 2] = -0.0000027292535204;
        M[1 * ldq + 3] = 0.0000000000467321;
        M[2 * ldq + 0] = 0.0000852542882848;
        M[2 * ldq + 1] = -0.0000027292535204;
        M[2 * ldq + 2] = 0.0000231105413411;
        M[2 * ldq + 3] = -0.0000000001988117;
        M[3 * ldq + 0] = -0.0000000001755091;
        M[3 * ldq + 1] = 0.0000000000467321;
        M[3 * ldq + 2] = -0.0000000001988117;
        M[3 * ldq + 3] = 0.0000000000000292;

    } else {
        //return identity matrix
        M[0 * ldq + 0] = 1;
        M[1 * ldq + 0] = 0;
        M[2 * ldq + 0] = 0;
        M[3 * ldq + 0] = 0;

        M[0 * ldq + 1] = 0;
        M[1 * ldq + 1] = 1;
        M[2 * ldq + 1] = 0;
        M[3 * ldq + 1] = 0;

        M[0 * ldq + 2] = 0;
        M[1 * ldq + 2] = 0;
        M[2 * ldq + 2] = 1;
        M[3 * ldq + 2] = 0;

        M[0 * ldq + 3] = 0;
        M[1 * ldq + 3] = 0;
        M[2 * ldq + 3] = 0;
        M[3 * ldq + 3] = 1;
    }
}


