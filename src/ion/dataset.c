#include <gmcmc/gmcmc_ion.h>
#include <gmcmc/gmcmc_ion_missed_events.h>
#include <stdlib.h>

/**
 * Destroys an ion channel dataset.
 *
 * @param [in] dataset  the dataset to destroy.
 */
void gmcmc_ion_dataset_destroy(gmcmc_ion_dataset * dataset) {
  dataset->type->destroy(dataset->data);
  free(dataset);
}

void gmcmc_burst_dataset_destroy(gmcmc_burst_dataset * dataset) {
  dataset->type->destroy(dataset->data);
  free(dataset);
}
