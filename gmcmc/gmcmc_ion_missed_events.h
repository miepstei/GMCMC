/*
 * gmcmc_ion_missed_events.h
 *
 *  Created on: 11 Sept 2013
 *      Author: Michael Epstein
 */

#ifndef GMCMC_ION_MISSED_EVENTS_H
#define GMCMC_ION_MISSED_EVENTS_H

#include <gmcmc/gmcmc_likelihood.h>
#include <gmcmc/gmcmc_ion.h>


/**
 * Dataset type.
 */
typedef struct {
  void (*destroy)(void *);                          /**< Destroy dataset-specific */
  size_t * (*burst_length)(const void *, size_t, size_t);   /**< Get number of burst elements in set i at position n */
  const double * (*burst)(const void *, size_t, size_t);    /**< Get burst in set i at position n*/
  const void * (*get_params)(const void *);         /**< Get auxiliary data vector of experimental conditions*/
  size_t (*burst_count)(const void *, size_t);                        /**< Get number of bursts in set i*/
} gmcmc_burst_dataset_type;

/**
 * A Geometric MCMC dataset has an array of timepoints t and one or more arrays
 * of data points y(t).  There are m timepoints and n arrays of m data points.
 *
 * For some datasets there is also an auxiliary data vector.
 */
typedef struct {
  const gmcmc_burst_dataset_type * type;      /**< Dataset type */
  void * data;                          /**< Dataset-specific data structures */
} gmcmc_burst_dataset;


//extern const gmcmc_likelihood_function ion_likelihood_missed_events_mh;

extern const gmcmc_likelihood_function gmcmc_ion_missed_events_likelihood_mh;
extern const gmcmc_likelihood_function gmcmc_ion_missed_events_likelihood_simp_mmala;
extern const gmcmc_likelihood_function gmcmc_ion_missed_events_preconditioned_mh;

/**
 * Destroys a dataset, freeing any memory used to store the data.
 *
 * @param [in] dataset  the dataset to destroy.
 */
void gmcmc_burst_dataset_destroy(gmcmc_burst_dataset *);

/**
 * Gets a pointer to the timepoints t.
 *
 * @param [in] dataset  the dataset
 *
 * @return the number of elements in the burst at position i in set n.
 */
static inline size_t * gmcmc_dataset_get_burst_number(const gmcmc_burst_dataset * dataset, size_t i, size_t n) {
  return dataset->type->burst_length(dataset->data , i , n);
}

/**
 * Gets a pointer to a data vector y_i(t).
 *
 * @param [in] dataset  the dataset
 * @param [in] i        the index of the burst
 *
 * @return a pointer to burst(t), or NULL if i is out of range.
 */
static inline const double * gmcmc_dataset_get_burst(const gmcmc_burst_dataset * dataset, size_t i, size_t n) {
  return dataset->type->burst(dataset->data, i , n);
}


/**
 * Gets the number of data vectors y.
 *
 * @param [in] dataset  the dataset
 *
 * @return the number bursts in the dataset.
 */
static inline size_t gmcmc_dataset_get_num_bursts(const gmcmc_burst_dataset * dataset , size_t n) {
  return dataset->type->burst_count(dataset->data, n);
}

/**
 * Gets a pointer to any auxiliary data in the dataset.
 *
 * @param [in] dataset  the dataset.
 *
 * @return a pointer to the auxiliary data.
 */
static inline const void * gmcmc_dataset_get_auxdata(const gmcmc_burst_dataset * dataset) {
  return (dataset->type->get_params == NULL) ? NULL : dataset->type->get_params(dataset->data);
}

/**
 *  Loads an ion channel dataset from a Matlab file.  
 *  The dimensionality of the timepoints must match that of the data.
 *  
 *  @param [out] dataset     the dataset object to load data into
 *  @param [in]  filename    the name of the Matlab .mat file containing the data
 *  
 *  @return 0 on success,
 *  
 *  GMCMC_ENOMEM if there is not enough memory to allocate the dataset or
 *               data vectors,
 *  
 *  GMCMC_EINVAL if the Matlab file does not contain valid ion channel
 *               data,
 *  
 *  GMCMC_EIO    if there is an input/output error.
 */
int gmcmc_dataset_create_matlab_ion_burst(gmcmc_burst_dataset **, const char *);


#endif  /* GMCMC_ION_MISSED_EVENTS_H */
