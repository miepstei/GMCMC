#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gmcmc/gmcmc_errno.h>
#include <gmcmc/gmcmc_ion_missed_events.h>

#include <mat.h>
#include <matrix.h>

/**
 * An ion channel with bursts loaded from Matlab.
 */
typedef struct {
    MATFile * file;       /** Matlab file pointer */
    mxArray * bursts;     /** Matlab cell array of bursts */
    mxArray * tcrit;      /** Matlab array of tcrits */
    mxArray * tres;       /** Matlab array of tres*/
    mxArray * conc;       /** Matlab array of concentrations */
    mxArray * n;          /** Matlab scalar of experiment sets */
    mxArray * useChs;     /** Matlab array of chs flags */
} burst_matlab_dataset;

/**
 * Destructor for ion channel Matlab datasets.  Frees the Matlab mxArrays and
 * closes the MATFile pointer.
 *
 * @param [in] data  the Matlab ion channel dataset.
 */
static void destroy(void * data) {
    burst_matlab_dataset * m = (burst_matlab_dataset *)data;
    mxDestroyArray(m->bursts);
    mxDestroyArray(m->tcrit);
    mxDestroyArray(m->tres);
    mxDestroyArray(m->conc);
    mxDestroyArray(m->n);
    mxDestroyArray(m->useChs);
    matClose(m->file);
    free(m);
}

/**
 * Get a pointer to the bursts from a Matlab ion channel dataset.
 *
 * @param [in] data  the Matlab ion channel dataset
 *
 * @return a pointer to the timepoints or NULL if the mxArray contains no real
 *           data.
 */

/** Obtain the actual burst i in a burst set */

static const double * burst(const void * data, size_t i, size_t set_no) {
    burst_matlab_dataset * m = (burst_matlab_dataset *)data;

    if (set_no >= mxGetScalar(m->n) )
        GMCMC_ERROR_VAL("index is out of range for burst set", GMCMC_EINVAL, NULL);   
    
    if (i >= mxGetN(mxGetCell(m->bursts,set_no))){
       //we have moved passed the end of the array
       GMCMC_ERROR_VAL("index is out of range", GMCMC_EINVAL, NULL);
    } 
    
    if (mxGetCell(m->bursts,set_no) == NULL)
        GMCMC_ERROR_VAL("bursts contains no real data", GMCMC_EIO, NULL);

    return mxGetPr(mxGetCell(mxGetCell(m->bursts,set_no),i));
}

/** obtain length of individual burst in a burst set*/

static size_t * burst_length(const void * data, size_t i, size_t set_no){
    burst_matlab_dataset * m = (burst_matlab_dataset *)data;

    if (set_no >= mxGetScalar(m->n) )
        GMCMC_ERROR_VAL("index is out of range for burst set", GMCMC_EINVAL, NULL);

    if (i >= mxGetN(mxGetCell(m->bursts,set_no)))
        GMCMC_ERROR_VAL("index is out of range for burst", GMCMC_EINVAL, NULL);
    
    if (mxGetCell(m->bursts,set_no) == NULL)
        GMCMC_ERROR_VAL("bursts contains no real data", GMCMC_EIO, NULL);

    size_t * burst_length = malloc(sizeof *burst_length);
    *burst_length = mxGetN(mxGetCell(mxGetCell(m->bursts,set_no),i));    
    
    return burst_length;

}

/**
 * Gets the number of bursts in the burst set.
 *
 * @return the number of bursts in the set.
 */
static size_t m(const void * data, size_t set_no) {
    burst_matlab_dataset * m = (burst_matlab_dataset *)data;

    if (set_no >= mxGetScalar(m->n) )
        GMCMC_ERROR("index is out of range for burst set", GMCMC_EINVAL);

    return mxGetN(mxGetCell(m->bursts,set_no));
}

/**
 * Get the experimental parameters (tres,tcrit,conc
 *
 *@return an array of pointers to experimental conditions
 */

static const void * get_params(const void * data){
    burst_matlab_dataset * m = (burst_matlab_dataset *)data;

    double ** exp_params = malloc(sizeof(double *) * 5);
    exp_params[0] = mxGetPr(m->tres);
    exp_params[1] = mxGetPr(m->tcrit);
    exp_params[2] = mxGetPr(m->conc);
    exp_params[3] = mxGetPr(m->useChs);
    exp_params[4] = mxGetPr(m->n);
    void * params = exp_params;
    return params;
}


/**
 * Ion channel Matlab dataset type.
 */

//burst_length destroy, number of elements at position i, get burst at i, experimental data, burst_count
static const gmcmc_burst_dataset_type my_type = { destroy, burst_length, burst,
    get_params, m};

/**
 * Loads an ion channel dataset from a Matlab file.  The file must contain a
 * column cell array named "bursts" with each cell containing arrays of double
 * vectors 
 *  *
 * @param [out] dataset     the dataset object to load data into
 * @param [in]  filename    the name of the Matlab .mat file containing the data
 *
 * @return 0 on success,
 *         GMCMC_ENOMEM if there is not enough memory to allocate the dataset or
 *                        data vectors,
 *         GMCMC_EINVAL if the Matlab file does not contain valid ion channel
 *                        data,
 *         GMCMC_EIO    if there is an input/output error.
 */
int gmcmc_dataset_create_matlab_ion_burst(gmcmc_burst_dataset ** dataset, const char * filename) {
    // Allocate the Matlab dataset structure
    burst_matlab_dataset * m;
    if ((m = malloc(sizeof(burst_matlab_dataset))) == NULL)
        GMCMC_ERROR("Failed to allocate Matlab-specific dataset structure", GMCMC_ENOMEM);
    
    // Open the Matlab data file
    m->file = matOpen(filename, "r");
    if (m->file == NULL) {
        free(m);
        GMCMC_ERROR("Unable to open Matlab data file", GMCMC_EIO);
    }
   
    // Obtain the number of concentrations in the file
    if ((m->n  = matGetVariable(m->file,"n")) == NULL) {
        matClose(m->file);
        free(m);
        GMCMC_ERROR("Unable to find experiment number", GMCMC_EINVAL);
    }
 
    // Get pointers to the bursts
    if ((m->bursts = matGetVariable(m->file, "bursts")) == NULL) {
        matClose(m->file);
        free(m);
        GMCMC_ERROR("Unable to find bursts", GMCMC_EINVAL);
    }

    // Check the bursts are stored as cell array and check number of burst sets
    if (!mxIsCell(m->bursts) && mxGetNumberOfElements(m->bursts) != mxGetScalar(m->n) ) {
        mxDestroyArray(m->bursts);
        matClose(m->file);
        free(m);
        GMCMC_ERROR("Matlab file bursts are not of format cell", GMCMC_EINVAL);
    }

    //get tcrit
    if ((m->tcrit = matGetVariable(m->file,"tcrit"))==NULL){
        matClose(m->file); 
        free(m);
        GMCMC_ERROR("Unable to find tcrit", GMCMC_EINVAL);
    }

    //check it is a scalar array and has the correct number of elements
    if (!mxIsDouble(m->tcrit) || mxGetNumberOfElements(m->tcrit) != mxGetScalar(m->n) ) {
        mxDestroyArray(m->tcrit);
        matClose(m->file);
        free(m);
        GMCMC_ERROR("tcrit is not of format scalar double", GMCMC_EINVAL);
    }

    //get tres
    if ((m->tres = matGetVariable(m->file,"tres"))==NULL){
        matClose(m->file);
        free(m);
        GMCMC_ERROR("Unable to find tres", GMCMC_EINVAL);
    }

    //check it is a double and has the correct number of elements

    if (!mxIsDouble(m->tres) || mxGetNumberOfElements(m->tres) != mxGetScalar(m->n) ) {
        mxDestroyArray(m->tres);
        matClose(m->file);
        free(m);
        GMCMC_ERROR("tres is not of format scalar double", GMCMC_EINVAL);
    }

    //get CHS flags...
    if ((m->useChs = matGetVariable(m->file,"useChs"))==NULL){
        matClose(m->file);
        free(m);
        GMCMC_ERROR("Unable to find chs flags", GMCMC_EINVAL);
    }

    if (!mxIsDouble(m->useChs) || mxGetNumberOfElements(m->useChs) != mxGetScalar(m->n) ) {
        mxDestroyArray(m->useChs);
        matClose(m->file);
        free(m);
        GMCMC_ERROR("chs is not of format scalar double", GMCMC_EINVAL);
    }

    //get the concentration to apply to the mechanism and check there are the correct number of concs
    if ((m->conc = matGetVariable(m->file,"conc"))==NULL){
        matClose(m->file);
        free(m);
        GMCMC_ERROR("Unable to find conc", GMCMC_EINVAL);
    }

    if (!mxIsDouble(m->conc) || mxGetNumberOfElements(m->conc) != mxGetScalar(m->n) ) {
        mxDestroyArray(m->conc);
        matClose(m->file);
        free(m);
        GMCMC_ERROR("conc is not of format scalar double", GMCMC_EINVAL);
    }


    // Allocate the dataset structure
    if ((*dataset = malloc(sizeof(gmcmc_burst_dataset))) == NULL) {
        mxDestroyArray(m->bursts);
        mxDestroyArray(m->tcrit);
        mxDestroyArray(m->tres);
        mxDestroyArray(m->conc);
        mxDestroyArray(m->useChs);
        mxDestroyArray(m->n);
        matClose(m->file);
        free(m);
        GMCMC_ERROR("Unable to allocate dataset", GMCMC_ENOMEM);
    }
    
    // Set the dataset type and dataset-specific structure

    (*dataset)->type = &my_type;
    (*dataset)->data = m;
    
    return 0;
}
