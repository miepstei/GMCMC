#include <gmcmc/gmcmc_ion_missed_events.h>
#include <gmcmc/gmcmc_errno.h>
#include <gmcmc/gmcmc_geometry.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>


// Forward declarations of utility functions
extern int dcprogs_likelihood(const void *, const double *, int, size_t, int, double, double, double *);
static int first_order_deriv(const void *, const gmcmc_model * , const double *, double **, const int, double * );
static int second_order_deriv(const void *, const gmcmc_model * , const double *, double **, const size_t, const size_t, const double, double * );
/**
 *  Ion Channel model-specific data.
 *   
 *  In addition to data common to all models, ion channel models have a Q matrix
 *  which details the transitions between open and closed states.
 */

struct gmcmc_ion_model {
 /**
 *     Number of closed and open states in the model.
 */       
  unsigned int closed, open;

 /**
 *   Function to update the Q matrix based on the current parameter values.
 */

  gmcmc_ion_calculate_Q_matrix calculate_Q_matrix;
};



/**
 *  Ion channel model likelihood function using Simplified M-MALA
 *   
 *  @param [in]  dataset     dataset  
 *  @param [in]  model       model to evaluate  
 *  @param [in]  params      current parameter values to evaluate the model
 *  @param [in]  n           number of parameters in the current block
 *  @param [in]  block       indices of the parameters in the current block (may
 *                            be NULL if there is no blocking)
 *  @param [out] likelihood  likelihood value
 *  @param [out] geometry    geometry for the current parameter block (may be
 *                            NULL if no geometry is required by the current
 *                            stage of the algorithm)
 *
 *
 *  @return 0 on success,
 *      GMCMC_ENOMEM if there is not enough memory to allocate temporary
 *                       variables,
 *      GMCMC_ELINAL if there was an unrecoverable error in an external
 *                      linear algebra routine
 *      GMCMC_DCPROGS if there was a problem calculating likelihood numbers 
 *                      within DCPROGS.
 */


static int ion_missed_events_likelihood_simp_mmala(const void * dataset, const gmcmc_model * model,
                             const double * params, size_t n, const size_t * block,
                             double * likelihood, void ** geometry){

  //disable blick updating of parameters
  (void)block; 

  const size_t num_params = gmcmc_model_get_num_params(model);
  *likelihood = -INFINITY;
  // Get the model specific data
  gmcmc_ion_model * ion_model = (gmcmc_ion_model *)gmcmc_model_get_modelspecific(model);
  const unsigned int num_states = ion_model->closed + ion_model->open;
  
  //get the experimental conditions
  double ** exp_params = (double **)gmcmc_dataset_get_auxdata(dataset);
  double * tres = exp_params[0];
  double * tcrit = exp_params[1];
  //double * conc = exp_params[2];
  
  //allocate the Q-matrix
  double * Q;
  size_t ldq = (num_states + 1u) & ~1u;

  if ((Q = calloc(num_states * ldq, sizeof(double))) == NULL )
    GMCMC_ERROR("Unable to allocate Q matrix", GMCMC_ENOMEM);
  
  //calculate Q matrix
  ion_model->calculate_Q_matrix(params, Q, ldq, 1);
  
  /*
  for (int i=0; i<num_params; i++){
    for (int j=0;j<num_params;j++){
      printf("q(%i,%i) -> %.16f\n",i,j,Q[(j*num_states)+i]);
      }
  }
  */

  //calculate the log-likelihood
  int error = 0;
  if((error = dcprogs_likelihood(dataset,Q, num_states, ldq, ion_model->open, *tres, *tcrit, likelihood) !=0)){
    free(Q);
    free(exp_params);
    GMCMC_WARNING("Unable to calculate missed_events likelihood", GMCMC_DCPROGS);
  }

  //printf("Likelihood = %.16f\n",*likelihood);
  
  //now calculate the gradient information
  // Calculate the gradient of the log-likelihood
  if (geometry != NULL){
    size_t ldfi = (num_params + 1u) & ~1u;
    //size_t *size = (num_params + 3) * ldfi;
    gmcmc_geometry_simp_mmala * g;
    if ((*geometry = g = calloc(1, sizeof(gmcmc_geometry_simp_mmala))) == NULL) {
      free(Q);
      free(exp_params);
      GMCMC_ERROR("Failed to allocate gradient structure", GMCMC_ENOMEM);
    }
   //*size *= sizeof(double);

   // Unpack serialised data structure
   //double * gradient_ll = *geometry;            // Length num_params
   //double * gradient_log_prior = &gradient_ll[ldfi];     // Length num_params (start on aligned offset)
   //double * FI = &gradient_log_prior[ldfi];              // Observed Fisher information Length num_params * ldfi
   //double * hessian_log_prior = &FI[num_params * ldfi];
  
    if((g->gradient_log_prior = malloc(num_params * sizeof(double))) == NULL ||
      (g->gradient_log_likelihood = calloc(num_params, sizeof(double))) == NULL ||
      (g->hessian_log_prior = calloc(num_params, sizeof(double))) == NULL ||
      (g->FI = calloc(num_params, (g->ldfi = (num_params + 1u) & ~1u) * sizeof(double))) == NULL){
   
      free(Q);
      free(exp_params);
      free(g->gradient_log_prior);
      free(g->gradient_log_likelihood);
      free(g->hessian_log_prior);
      free(g->FI);
      free(g);
      GMCMC_ERROR("Failed to allocate gradient vectors", GMCMC_ENOMEM);
    } 

    for (size_t j = 0; j < num_params; j++) {
    
      // Calculate the gradient of the log-likelihood
      double sensitivity = 0;
      int sens_error = 0;
      if ((sens_error = first_order_deriv(dataset,model,params,exp_params,j,&sensitivity))!=0){
        free(Q);
        free(exp_params);
        free(g->gradient_log_prior);
        free(g->gradient_log_likelihood);
        free(g->hessian_log_prior);
        free(g->FI);
        free(g);
        GMCMC_WARNING("Failed to calculate 1st order gradients", GMCMC_DCPROGS);
      }
    
      g->gradient_log_likelihood[j] = sensitivity;

      // Calculate gradient of the log prior
      const gmcmc_distribution * prior = gmcmc_model_get_prior(model, j);
      g->gradient_log_prior[j] = gmcmc_distribution_log_pdf_1st_order(prior, params[j]);

      // Calculate hessian of the log prior
      g->hessian_log_prior[j] = gmcmc_distribution_log_pdf_2nd_order(prior, params[j]);

    }
    int sens_error = 0;
    //Metric tensor / Fisher information
    for (size_t i = 0 ;i < num_params; i++) {
      for (size_t j = i; j < num_params; j++) {
        double sec_ord_sens;
        if ((sens_error=second_order_deriv(dataset,model,params,exp_params,i,j,*likelihood,&sec_ord_sens)) !=0){
          free(Q);
          free(g->gradient_log_prior);
          free(g->gradient_log_likelihood);
          free(g->hessian_log_prior);
          free(g->FI);
          free(g);
          free(exp_params);
          GMCMC_WARNING("Failed to calculate 2nd order gradients", GMCMC_DCPROGS);
        }
        g->FI[i * ldfi + j] = sec_ord_sens;
        //printf("%.16f\n",FI[j * ldfi + i]);       
        g->FI[j * ldfi + i] = g->FI[i * ldfi + j];
      }
    }
    for (size_t i = 0 ;i < num_params; i++) {
      for (size_t j = 0; j < num_params; j++) {
        printf("%.16f\n",g->FI[i * ldfi + j]);
      }
    }
  } 
    
    //cleanup...
    //printf("\n\n");
  free(Q);
  free(exp_params);
  //printf("Returning\n");
  return error;
}

static int second_order_deriv(const void * dataset, const gmcmc_model * model, const double * params, double ** exp_params,
                             const size_t ps1, const size_t ps2, const double curr_likelihood,  double * sensitivity){
  //setup
  int error = 0;

  const size_t num_params = gmcmc_model_get_num_params(model);
  gmcmc_ion_model * ion_model = (gmcmc_ion_model *)gmcmc_model_get_modelspecific(model);

  const unsigned int num_states = ion_model->closed + ion_model->open;
  size_t ldq = (num_states + 1u) & ~1u;

  //unpack experimental params
  double * tres = exp_params[0];
  double * tcrit = exp_params[1];
  //double * conc = exp_params[2];
  
  //create a mutable copy of the current params
  double * param_copy = calloc(num_params, sizeof(double));
  for (size_t j = 0; j < num_params; j++){
    param_copy[j] = params[j];
    //printf("%.16f\n", param_copy[j]);
  }

  //printf("%zu,%zu\n",ps1,ps2);
                   
  double const param1 = param_copy[ps1]; 
  double const param2 = param_copy[ps2];
  double const h1 = 0.01;//sqrt(DBL_EPSILON) * param1;
  double const h2 = 0.01;//sqrt(DBL_EPSILON) * param2;

  //printf("Hessian %i,%i\n",ps1,ps2);
  //allocate memory for Q matrix
  double * Qj;  
  if ((Qj = calloc(num_states * ldq, sizeof(double))) == NULL )
    GMCMC_ERROR("Unable to allocate Qj matrix for 2nd order derivatives", GMCMC_ENOMEM);
    
  if (ps1 != ps2){
    //we need four likelihood calcs
    param_copy[ps1] = param1+h1;
    param_copy[ps2] = param2+h2;    

    //f(p1+h,p2+h)
    ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
    double * uu_likelihood;
    
    if((uu_likelihood = malloc(sizeof(double))) ==NULL){
      free(Qj);
      GMCMC_ERROR("Unable to allocate uu likelihood for 2nd order derivatives", GMCMC_ENOMEM);
    }
    *uu_likelihood = -INFINITY;
    if((error = dcprogs_likelihood(dataset,Qj, num_states, ldq, ion_model->open,  *tres, *tcrit, uu_likelihood) !=0)){
      free(Qj);
      free(uu_likelihood);
      free(param_copy);
      GMCMC_WARNING("Unable to calculate missed_events likelihood for f(x+h, x+h)", GMCMC_DCPROGS);
    }
   // printf("uu_likelihood = %.16f\n",*uu_likelihood);
    //printf("(%zu,%zu) = p1+h,p2+h=%.16f  ,  %.16f\n",ps1,ps2,param1+h1,param2+h2);
 
    //f(p1+h,p2-h)
    param_copy[ps2] = param2-h2;
    ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
    double * um_likelihood;

    if ((um_likelihood = malloc(sizeof(double))) == NULL){
      free(Qj);
      free(uu_likelihood);
      GMCMC_ERROR("Unable to allocate um_likelihood for 2nd order derivatives", GMCMC_ENOMEM);;
    }

    *um_likelihood = -INFINITY;
    if((error = dcprogs_likelihood(dataset,Qj, num_states, ldq, ion_model->open, *tres, *tcrit, um_likelihood) != 0)){
      free(Qj);
      free(uu_likelihood);
      free(um_likelihood);
      free(param_copy);
      GMCMC_WARNING("Unable to calculate missed_events likelihood for f(x+h, x-h)", GMCMC_DCPROGS);
    }                                        
   //
   //printf("um_likelihood = %.16f\n",*um_likelihood);

    //f(p1-h,p2+h)
    param_copy[ps1] = param1-h1;
    param_copy[ps2] = param2+h2;
    ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
    double * mu_likelihood;

    if ((mu_likelihood = malloc(sizeof(double))) == NULL){
      free(Qj);
      free(uu_likelihood);
      free(um_likelihood);
      GMCMC_ERROR("Unable to allocate mu_likelihood for 2nd order derivatives", GMCMC_ENOMEM);;
    }
    *mu_likelihood = -INFINITY;
    if((error = dcprogs_likelihood(dataset,Qj,num_states, ldq, ion_model->open, *tres, *tcrit, mu_likelihood) != 0)){
      free(Qj);
      free(uu_likelihood);
      free(um_likelihood);
      free(mu_likelihood);
      free(param_copy);
      GMCMC_WARNING("Unable to calculate missed_events likelihood for f(x-h, x+h)", GMCMC_DCPROGS);
    }
   // printf("mu_likelihood = %.16f\n",*mu_likelihood);

    //f(p1-h,p2-h)
    param_copy[ps2] = param2-h2;
    ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
    double * mm_likelihood;
    
    if ((mm_likelihood = malloc(sizeof(double))) == NULL){
      free(Qj);
      free(uu_likelihood);
      free(um_likelihood);
      free(mu_likelihood);
      GMCMC_ERROR("Unable to allocate mm_likelihood for 2nd order derivatives", GMCMC_ENOMEM);
    }
    *mm_likelihood=-INFINITY;

    if((error =dcprogs_likelihood(dataset,Qj,num_states, ldq, ion_model->open, *tres, *tcrit, mm_likelihood) !=0 )){
      free(Qj);
      free(uu_likelihood);
      free(um_likelihood);
      free(mu_likelihood);
      free(mm_likelihood);
      free(param_copy);
      GMCMC_WARNING("Unable to calculate missed_events likelihood for f(x+h)", GMCMC_DCPROGS);
    }
   // printf("mm_likelihood = %.16f\n",*mm_likelihood);

    *sensitivity = (*mm_likelihood+*uu_likelihood-*um_likelihood-*mu_likelihood)/(4*h1*h2);
    /*printf("denominator = %.16f\n",(*mm_likelihood+*uu_likelihood-*um_likelihood-*mu_likelihood));
    printf("sensitivity = %.16f\n\n",*sensitivity);
    //cleanup allocations relevant to this if statement
    free(uu_likelihood);
    free(um_likelihood);
    free(mu_likelihood);
    free(mm_likelihood);
    */
  }

  else {
    //we need two likelihood calcs and we reuse the current likelihood calculation
    //f(p+h)
    param_copy[ps1] = param1+h1;
    ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
    double * u_likelihood;
    if ((u_likelihood = malloc(sizeof(double))) == NULL){
      free(Qj);
      free(param_copy);
      GMCMC_ERROR("Unable to allocate u_likelihood for diagonal 2nd order derivatives", GMCMC_ENOMEM);
    }
    *u_likelihood = -INFINITY; 
    if ((error=dcprogs_likelihood(dataset,Qj,num_states, ldq, ion_model->open, *tres, *tcrit, u_likelihood) !=0)){
      free(Qj);
      free(u_likelihood);
      free(param_copy);
      GMCMC_WARNING("Unable to calculate missed_events likelihood for diagonal element f(x+h)", GMCMC_DCPROGS);
    }

    //f(p-h)
    param_copy[ps1] = param1-h1;
    ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
    double * l_likelihood;

    if((l_likelihood = malloc(sizeof(double))) == NULL){
      free(Qj);
      free(u_likelihood);
      free(param_copy);
      GMCMC_ERROR("Unable to allocate l_likelihood for 2nd order derivatives", GMCMC_ENOMEM);;
    }    

    *l_likelihood = -INFINITY;
    if ((error=dcprogs_likelihood(dataset,Qj,num_states, ldq, ion_model->open, *tres, *tcrit, l_likelihood) !=0)) {
      free(Qj);
      free(u_likelihood);
      free(l_likelihood);
      free(param_copy);
      GMCMC_WARNING("Unable to calculate missed_events likelihood for diagonal element f(x-h)", GMCMC_DCPROGS);
    }

    *sensitivity = (*l_likelihood - (2*curr_likelihood) + *u_likelihood)/(pow(h1,2));

    
    //printf("Hessian diagonal -> %i,%i\n",ps1,ps1);
    //printf("p , l-likeihood -> %.16f, %.16f\n",param1-h1,*l_likelihood);
    //printf("p, u-likelihood -> %.16f, %.16f\n",param1+h1,*u_likelihood);
    //printf("p, curr_likelihood -> %.16f, %.16f\n",param1,curr_likelihood);
    //printf("h^2 -> %.16f\n",pow(h1,2));
    //printf("sensitivity -> %.16f\n",*sensitivity);
    
    //cleanup
    free(u_likelihood);
    free(l_likelihood);
  } 

  //cleanup
  free(Qj);
  free(param_copy);
  //printf("Hessian structure for params %i,%i is %.8f\n",ps1,ps2,*sensitivity);
  return error;
}

static int first_order_deriv(const void * dataset, const gmcmc_model * model, const double * params, double ** exp_params,
                             const int ps, double * sensitivity){


  const size_t num_params = gmcmc_model_get_num_params(model);
  gmcmc_ion_model * ion_model = (gmcmc_ion_model *)gmcmc_model_get_modelspecific(model);
  
  const unsigned int num_states = ion_model->closed + ion_model->open;
  size_t ldq = (num_states + 1u) & ~1u; 

  //unpack experimental params
  double * tres = exp_params[0];
  double * tcrit = exp_params[1];
  //double * conc = exp_params[2];

  //create a copy of the current params
  double * param_copy = calloc(num_params, sizeof(double));
  for (size_t j = 0; j < num_params; j++){
    param_copy[j] = params[j];
  }


  //by central differe
  double t_param = param_copy[ps];
  double h = 0.01;//params[ps]*sqrt(DBL_EPSILON);
    
  //f(x+h)
  param_copy[ps] = t_param+h;
  double * Qj;

  if ((Qj = calloc(num_states * ldq, sizeof(double))) == NULL )
    GMCMC_ERROR("Unable to allocate Qj matrix for 1st order derivaives", GMCMC_ENOMEM);
  
  ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
    
  int error = 0;
  double * u_likelihood = malloc(sizeof(double));
  *u_likelihood = -INFINITY;

  //printf("HREHREH%i\n",ps);
  if((error = dcprogs_likelihood(dataset,Qj, num_states, ldq, ion_model->open, *tres, *tcrit, u_likelihood) !=0)){
    free(Qj);
    free(u_likelihood);
    free(param_copy);
    GMCMC_WARNING("Unable to calculate missed_events likelihood for f(x+h)", GMCMC_DCPROGS);
  }
  
  //printf("HERHEH%i\n",ps);
  //f(x-h)
  param_copy[ps] = t_param-h;
  ion_model->calculate_Q_matrix(param_copy, Qj, ldq, 1);
  double *l_likelihood = malloc(sizeof(double)); 
  *l_likelihood = -INFINITY;
  if((error = dcprogs_likelihood(dataset,Qj, num_states, ldq, ion_model->open, *tres, *tcrit, l_likelihood) !=0)){
    free(Qj);
    free(u_likelihood);
    free(l_likelihood);
    free(param_copy);
    GMCMC_WARNING("Unable to calculate missed_events likelihood for f(x-h)", GMCMC_DCPROGS);
  }

  *sensitivity = (*u_likelihood - *l_likelihood)/(2*h);
  //printf("Gradient structure for param %i is %.16fi. x+h = %.16f, x-h = %.16f\n",ps,t_param+h,t_param-h,*sensitivity); 
  //printf("f(x+h) = %.16f\n", *u_likelihood);
  //printf("f(x-h) = %.16f\n", *l_likelihood);

  //cleanup
  free(Qj);
  free(u_likelihood);
  free(l_likelihood);
  free(param_copy);

  return error;
}


const gmcmc_likelihood_function gmcmc_ion_missed_events_likelihood_simp_mmala = &ion_missed_events_likelihood_simp_mmala;

