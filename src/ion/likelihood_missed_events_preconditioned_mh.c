#include <gmcmc/gmcmc_ion_missed_events.h>
#include <gmcmc/gmcmc_errno.h>
#include <gmcmc/gmcmc_geometry.h>
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
  gmcmc_ion_precondition_covariance precondition_covariance;
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


static int ion_likelihood_missed_events_preconditioned_mh(const void * dataset, const gmcmc_model * model,
                             const double * params, size_t n, const size_t * block,
                             double * likelihood, void ** geometry){

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
      
      gmcmc_ion_model_calculate_Q_matrix(ion_model, params, Q, ldq ,conc[i]);

      if((error = dcprogs_likelihood(dataset,Q, num_states, ldq, ion_model->open, tres[i], tcrit[i],i, useChs[i] ,&set_likelihood) !=0)){
         free(Q);
         GMCMC_WARNING("Unable to calculate missed_events likelihood", GMCMC_DCPROGS); 
      }
      running_likelihood += set_likelihood;
      free(Q);
  }
  *likelihood = running_likelihood;

  if (geometry != NULL) {
    size_t ldfi = (n + 1u) & ~1u;
    gmcmc_geometry_simp_mmala * g;
    if ((*geometry = g = calloc(1, sizeof(gmcmc_geometry_simp_mmala))) == NULL) {
      free(exp_params);
      GMCMC_ERROR("Failed to allocate gradient structure", GMCMC_ENOMEM);
    }    

    if((g->gradient_log_prior = malloc(n * sizeof(double))) == NULL ||
      (g->gradient_log_likelihood = calloc(n, sizeof(double))) == NULL ||
      (g->hessian_log_prior = calloc(n, sizeof(double))) == NULL ||
      (g->FI = calloc(n, (g->ldfi = (n + 1u) & ~1u) * sizeof(double))) == NULL){

      free(exp_params);
      free(g->gradient_log_prior);
      free(g->gradient_log_likelihood);
      free(g->hessian_log_prior);
      free(g->FI);
      free(g);
      GMCMC_ERROR("Failed to allocate gradient vectors", GMCMC_ENOMEM);
    }
    
    for (size_t j = 0; j < n; j++) {

      // Gradient of the log-likelihood is set to 0
      g->gradient_log_likelihood[j] = 0;

      // Calculate gradient of the log prior
      const gmcmc_distribution * prior = gmcmc_model_get_prior(model, j);
      g->gradient_log_prior[j] = gmcmc_distribution_log_pdf_1st_order(prior, params[j]);

      // Calculate hessian of the log prior
      g->hessian_log_prior[j] = gmcmc_distribution_log_pdf_2nd_order(prior, params[j]);

    }    

    //Metric tensor is a fixed metric supplied with the model

    gmcmc_ion_model_precondition_covariance(ion_model,g->FI,g->ldfi);
  }

  free(exp_params);
  return error;

}
const gmcmc_likelihood_function gmcmc_ion_missed_events_preconditioned_mh = &ion_likelihood_missed_events_preconditioned_mh;

