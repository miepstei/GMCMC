#include <gmcmc/gmcmc_ion_missed_events.h>
#include <gmcmc/gmcmc_errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>


// Forward declarations of utility functions
extern int dcprogs_likelihood(const void *, const double *, int, size_t,int, const double , const double , const size_t,int, double *);

/**
 * Ion Channel model-specific data.
 *
 * In addition to data common to all models, ion channel models have a Q matrix
 * which details the transitions between open and closed states.
 */
struct gmcmc_ion_model {
  /**
   * Number of closed and open states in the model.
   */
  unsigned int closed, open;

  /**
   * Function to update the Q matrix based on the current parameter values.
   */
  gmcmc_ion_calculate_Q_matrix calculate_Q_matrix;
};


/**
 * Ion Channel model likelihood function (incorporating missed events) using Metropolis-Hastings.
 * Calculates p(D|M,params) (i.e. likelihood of seeing the data D given the
 * model M and parameters params)
 *
 * @param [in]  dataset     the data
 * @param [in]  model       the model
 * @param [in]  params      the parameter vector
 * @param [out] likelihood  the likelihood object to create and populate
 * @param [out] serdata     serialised data to be passed to the proposal function
 * @param [out] size        size of serialised data object, in bytes
 *
 * @return 0 on success,
 *         GMCMC_ENOMEM if there is not enough memory to allocate temporary
 *                        variables,
 *         GMCMC_ELINAL if there was an unrecoverable error in an external
 *                        linear algebra routine.
 */


static int ion_likelihood_missed_events_mh(const void * dataset, const gmcmc_model * model,
                             const double * params, size_t n, const size_t * block,
                             double * likelihood, void ** geometry){

  // Metropolis-Hastings likelihood functions don't calculate geometry
  (void)geometry;
  (void)n;
  (void)block;
  
  //Initialise error status
  int error = 0;
  
  // Initialise log likelihood to negative infinity so that it is set on
  // non-fatal errors
  *likelihood = -INFINITY;
  double  set_likelihood, running_likelihood;
  set_likelihood = 0;
  running_likelihood=0;

  // Get the model specific data
  gmcmc_ion_model * ion_model = (gmcmc_ion_model *)gmcmc_model_get_modelspecific(model);
  const unsigned int num_states = ion_model->closed + ion_model->open;

  //get the experimental conditions

  double ** exp_params = (double **)gmcmc_dataset_get_auxdata(dataset);

  double * tres = exp_params[0];
  double * tcrit = exp_params[1];
  double * conc = exp_params[2];
  double * useChs = exp_params[3];
  size_t sets = (size_t)*exp_params[4]; 

  // Allocate the Q matrix

  for (size_t i =0; i < sets; i++) {
      // Calculate the Q matrix
      double * Q;
      size_t ldq = (num_states + 1u) & ~1u;
      if ((Q = calloc(num_states * ldq, sizeof(double))) == NULL)
         GMCMC_ERROR("Unable to allocate Q matrix", GMCMC_ENOMEM);
      //printf("i,conc,tres,tcrit,useChs = %i,%.16f,%.16f,%.16f, %.1f\n",i,conc[i],tres[i],tcrit[i],useChs[i]);
      gmcmc_ion_model_calculate_Q_matrix(ion_model, params, Q, ldq ,conc[i]);

      //the plan is to send the dataset over to the c++ library to be parsed into objects that 
      //dcprogs understands
      if((error = dcprogs_likelihood(dataset,Q, num_states, ldq, ion_model->open, tres[i], tcrit[i],i, useChs[i] ,&set_likelihood) !=0)){
         free(Q);
         //printf("lik = %.16f\n",*likelihood); 
         GMCMC_WARNING("Unable to calculate missed_events likelihood", GMCMC_DCPROGS); 
      }
       //printf("tot lik = %.16f, set lik=%.16f p1=%.16f  p2=%.16f\n",*likelihood, set_likelihood, params[0], params[1]);
      running_likelihood += set_likelihood;
      free(Q);
  //cleanup...
  //
  }
  *likelihood = running_likelihood;
  //printf("lik = %.16f\n",*likelihood);
  free(exp_params);
  return error;

}
const gmcmc_likelihood_function gmcmc_ion_missed_events_likelihood_mh = &ion_likelihood_missed_events_mh;

