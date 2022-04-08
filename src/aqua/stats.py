"""Statistical analysis of AQuA events

For details of AQuA for Matlab, see https://github.com/yu-lab-vt/AQuA
"""

## Imports

import warnings
import copy

import numpy as np
import pandas as pd

import scipy.stats
import scipy.integrate

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.decomposition import PCA

from tqdm import tqdm

import multiprocessing as mp
import functools


## General statistics

def perm_stat( xs, ys, f,
               n = 1000,
               paired = False ):
    """Statistic `f` (a function taking the two datasets as arguments) quantifying
    the difference between data in `xs` and data in `ys`, computed in an imposed
    permuted null hypothesis
    
    Keyword arguments:
    n - the number of permutations
    paired - whether to treat the observations in `xs` and `ys` as paired when permuting
        (In the paired case, data points are randomly "flipped" between the categories x and y while
        maintaining the pairings of individual data points.)
    """
    
    if paired:
        if len( xs ) != len( ys ):
            raise ValueError( 'Must have equal groups for paired observations' )
    
    ret = np.zeros( (n,) )
    
    for i in range( n ):
        if paired:
            flip = np.random.randint( 2, xs.shape )
            xs_cur = np.zeros( xs.shape )
            ys_cur = np.zeros( ys.shape )
            xs_cur[flip == 0] = xs[flip == 0]
            xs_cur[flip == 1] = ys[flip == 1]
            ys_cur[flip == 0] = ys[flip == 0]
            ys_cur[flip == 1] = xs[flip == 1]
        else:
            combined_data = np.r_[xs, ys]
            combined_data = np.random.permutation( combined_data )
            xs_cur = combined_data[:len( xs )]
            ys_cur = combined_data[len( xs ):]
        
        ret[i] = f( xs_cur, ys_cur )
    
    return ret

def boot_stat( xs, f,
               n = 1000 ):
    """Statistic `f` (a function taking the data as an argument) computed by resampling
    the data `xs` with replacement
    
    Keyword arguments:
    n - the number of bootstrap samples to compute
    """
    
    n_x = len( xs )
    
    boot_values = np.zeros( (n,) )
    for i_boot in range( n ):
        sample = np.random.randint( n_x, size = (n_x,) )
        boot_values[i_boot] = f( xs[sample] )
    
    return boot_values


## General analyses

def ramp_effects( df, window,
                  window_key = 'center_time',
                  outcome_key = 'event_count',
                  group_key = 'cell_global',
                  p_threshold = 0.05,
                  verbose = False,
                  **kwargs ):
    """Determine the effect of time in `window` on event rate in the DataFrame raster `df`
    using statsmodels' GLM implementation
    
    Keyword arguments:
    outcome_key - key in `df` to check if there's a ramp effect on
        (Default: 'event_count')
    group_key - ramp effects are determined for each unique value of `group_key`
        (Default: 'cell_global')
    window_key - key to use as the independent variable (i.e. "time";
        default: 'center_time')
    p_threshold - p-value threshold for significant ramp effect
    verbose - if True, show a progress bar
    
    The rest of the kwargs are passed to `glm`
    """
    
    it = df.groupby( group_key )
    if verbose:
        it = tqdm( it )
    
    ret_list = []
    for group_idx, group in it:
        cur_ret = dict()
        cur_ret[group_key] = group_idx
        
        filter_window = ( (group[window_key] >= window[0])
                          & (group[window_key] < window[1]) )
        group_filtered = group.loc[filter_window]
        
        if np.sum( group_filtered[outcome_key] ) == 0:
            # No data in the window; cannot fit model
            
            cur_ret['slope'] = None
            cur_ret['p'] = None
            cur_ret['slope_low'] = None
            cur_ret['slope_high'] = None

            cur_ret['effect'] = 'No data'
            
            ret_list.append( cur_ret )
            continue
        
        ramp_model = smf.glm( formula = f'{outcome_key} ~ {window_key}',
                              data = group_filtered,
                              **kwargs )
        ramp_results = ramp_model.fit()
        
        cur_ret['slope'] = ramp_results.params[window_key]
        cur_ret['p'] = ramp_results.pvalues[window_key]
        cur_ret['slope_low'] = ramp_results.conf_int()[0][window_key]
        cur_ret['slope_high'] = ramp_results.conf_int()[1][window_key]
        
        cur_ret['effect'] = ( '0' if cur_ret['p'] >= p_threshold else
                              ( '+' if cur_ret['slope'] >= 0. else '-' ) )
        
        ret_list.append( cur_ret )
    
    return pd.DataFrame( ret_list )

# TODO Way to properly include propagation?
_default_pca_keys = [
    'mark_area_log',
    'mark_circMetric',
    'mark_decayTau_log',
    'mark_dffMax_log',
    'mark_dffMax2_log',
    'mark_fall91_log',
    'mark_nOccurSameLoc_log',
    'mark_nOccurSameLocSize_log',
    'mark_nOccurSameTime',
    'mark_peri_log',
    'mark_rise19_log',
    'mark_width11_log',
    'mark_width55_log'
]

def event_pca( events,
               keys = _default_pca_keys,
               return_object = False,
               **kwargs ):
    """Perform PCA on `events` using the features in `keys`
    
    If `return_object` is true, returns the PCA object from sklearn.decomposition
    
    Otherwise (default), returns (loadings, frac_variance, scores) where
        loadings - DataFrame where each row is a PC and each column is an input feature in `keys`
        frac_variance - Series where each entry is the fraction of variance explained by a PC
        scores - DataFrame where each row is an event from `events` and each column is the score
            for each PC
    
    kwargs are passed to PCA
    """
    
    ## Preprocess
    
    # Check that the keys we want are available
    good_keys = [ k for k in keys if k in events.keys() ]
    if len( good_keys ) < 1:
        raise Exception( 'None of the keys in `keys` are in `events`' )
    
    # Remove bad data points
    good_idx = ~np.any( np.isnan( events[good_keys] ), axis = 1 )
    
    ## Fit our model
    pca = PCA( **kwargs )
    pca.fit( events[good_keys].loc[good_idx] )
    
    if return_object:
        return pca
    
    ## Reformat outputs as DataFrames for convenience
    pc_index = [ f'mark_pc_{i + 1}' for i in range( pca.n_components_ ) ]
    
    # Loadings (loadings)
    loadings = pd.DataFrame( pca.components_,
                             index = pc_index,
                             columns = good_keys )
    
    # Fraction of variance explained
    frac_variance = pd.Series( pca.explained_variance_ratio_,
                               index = pc_index )
    
    # Scores
    # TODO This is slow
    scores = pd.DataFrame( np.zeros( (events.shape[0], pca.n_components_) ),
                           index = events.index,
                           columns = pc_index )
    for i_row, row in events[good_keys].iterrows():
        try:
            row_score = pca.transform( np.array( row ).reshape( 1, -1 ) )[0, :]
        except ValueError as e:
            # Row could not be transformed
            scores.iloc[i_row] = np.nan
        scores.iloc[i_row] = row_score
        
    return loadings, frac_variance, scores

def compare_rates( rf1, rf2, t, n ):
    """Compare two rate functions `rf1` and `rf2` using `n` iterations of the
    parametric bootstrap evaluated at `t`.
    
    Returns a one-sided pointwise p-value for whether `rf2` is greater than `rf1`
    """
    
    n_t = t.shape[0]
    
    # Determine the rate functions at t
    rt1 = rf1.predict( t )
    rt2 = rf2.predict( t )
    
    rt1_boot = np.zeros( (n, n_t) )
    rt2_boot = np.zeros( (n, n_t) )
    is_2_greater_1 = np.zeros( (n, n_t) )
    for i_boot in range( n ):
        # Get a sample of the corresponding NHPPs with the given rate functions
        ts1_boot = _pois_inhom( rt1, t )
        ts2_boot = _pois_inhom( rt2, t )
        
        # Determine the kernel fit for the bootstrapped data
        rf1_boot = rf1.copy()
        rf1_boot.fit( ts1_boot )
        rt1_boot[i_boot, :] = rf1_boot.predict( t )
        rf2_boot = rf2.copy()
        rf2_boot.fit( ts2_boot )
        rt2_boot[i_boot, :] = rf2_boot.predict( t )
        
        # Compare the newly fitted data between the two functions
        is_2_greater_1[i_boot, :] = rt2_boot[i_boot, :] > rt1_boot[i_boot, :]
        
    return np.sum( is_2_greater_1, axis = 0 ) / n


## RateFunctionKernel helpers

def _pois_inhom( rt, t ):
    """Simulate an inhomogeneous Poisson process with rates `rt` at times `t`"""
    rt_int = scipy.integrate.cumtrapz( rt, x = t )
    t_int = t[:-1] + 0.5 * np.diff( t )
    n_int = t_int.shape[0]
    
    ret = []
    acc = 0.
    i_cur = 0
    
    while True:
        acc += np.random.exponential( 1. )
        
        # We're outside of the time bounds; return condition
        if acc > rt_int[-1]:
            return np.array( ret )
        
        # Find zero crossing of the integral
        delta = rt_int - acc
        i_cur = np.where( delta >= 0 )[0][0]
        
        # Some interpolation magic for time steps
        delta_cur = delta[i_cur]
        delta_prev = delta[i_cur - 1]
        frac = 0. if delta_cur == delta_prev else (delta_prev / (delta_prev - delta_cur))
        
        t_cur = t_int[i_cur]
        t_prev = t_int[i_cur - 1]
        ret.append( frac * t_prev + (1. - frac) * t_cur )
        
    return ret

def _rate_kernel( ts, t_eval, kernel ):
    """Computes an estimate event rate by convolving with the given kernel
    
    Arguments:
    ts - the event times
    t_eval - the time points to evaluate the rate function at
    kernel - a symmetric, normalized function in time to place on each event
    """
    ret = np.zeros( t_eval.shape )
    for t in ts:
        if np.isnan( t ):
            continue
        cur_kernel = kernel( t_eval - t )
        ret = ret + cur_kernel
    return ret

def _rate_kernel_error_analytic( rt, t, error_kind ):
    """Compute dispersion of the `_rate_kernel` estimator using analytic Poisson results"""
    
    if len( error_kind ) < 1:
        raise Exception( '`error_kind` has no strategy specified' )
    error_kind_strategy = error_kind[0]
    
    warnings.warn( 'NHPP analytic results are not correct; use `bootstrap` instead' )
    
    n_t = t.shape[0]
    rt_err = np.sqrt( rt )
    
    if error_kind_strategy == 'sd':
        return rt_err
    
    if error_kind_strategy == 'se':
        # Actually kind of not sure how to do this
        raise NotImplementedError( 'Standard error not implemented' )
    
    if error_kind_strategy == 'ci':
        if len( error_kind ) < 2:
            raise Exception( 'No confidence level specified in `error_kind`' )
        ci_alpha = error_kind[1]
        
        raise NotImplementedError( 'Confidence interval not implemented' )
    
    raise Exception( f"Unknown `error_kind` strategy: '{error_kind_strategy}'" )

def _rate_kernel_error_boot_parametric( rt, t, kernel, n_boot, error_kind ):
    """Compute dispersion of the `_rate_kernel` estimator using the parametric bootstrap"""
    
    if len( error_kind ) < 1:
        raise Exception( '`error_kind` has no strategy specified' )
    error_kind_strategy = error_kind[0]
    
    n_t = t.shape[0]
    
    rt_boot = np.zeros( (n_boot, n_t) )
    for i_boot in range( n_boot ):
        # Get a sample of a NHPP with rate function `rt` at times `t`
        ts_boot = _pois_inhom( rt, t )
        # Determine the kernel fit for the bootstrapped data with `_rate_kernel`
        rt_boot[i_boot, :] = _rate_kernel( ts_boot, t, kernel )
    
    if error_kind_strategy == 'ci':
        if len( error_kind ) < 2:
            raise Exception( 'No confidence level specified in `error_kind`' )
        ci_alpha = error_kind[1]
        
        if n_boot < (1. / ci_alpha):
            warnings.warn( f'Insufficient bootstrap samples ({n_boot}) for desired confidence level ({ci_alpha:0.4f})' )
        
        rt_low = np.quantile( rt_boot, ci_alpha / 2., axis = 0 )
        rt_high = np.quantile( rt_boot, 1. - (ci_alpha / 2.), axis = 0 )
        return (rt_low, rt_high)
    
    raise Exception( f"Unknown `error_kind` strategy: '{error_kind_strategy}'" )

## Kernels
    
def _standard_kernel_rect( x ):
    return (np.abs(x) <= 1.) * (1 / 2.)

def get_kernel_rect( scale = 1. ):
    """Uniform kernel on [-scale, scale]"""
    return lambda x: (1 / scale) * _standard_kernel_rect(x / scale)

def _standard_kernel_tri( x ):
    return (np.abs(x) <= 1.) * (1 - np.abs(x))

def get_kernel_tri( scale = 1. ):
    """Triangular kernel on [-scale, scale]"""
    return lambda x: (1 / scale) * _standard_kernel_tri(x / scale)

def _standard_kernel_epanechnikov( x ):
    return (np.abs(x) <= 1.) * (3 / 4.) * (1. - np.power(x, 2.))

def get_kernel_epanechnikov( scale = 1. ):
    """Epanechnikov (mean-square-error optimal) kernel"""
    return lambda x: (1 / scale) * _standard_kernel_epanechnikov(x / scale)

def get_kernel_gauss( scale = 1. ):
    """Gaussian kernel with s.d. `scale`"""
    return scipy.stats.norm( scale = scale ).pdf

def get_kernel( *args ):
    """Get a kernel with the given specifications"""
    if len( args ) < 1:
        raise Exception( 'Kernel type unspecified' )
        
    kernel_type = args[0]
    
    if kernel_type == 'rect':
        if len( args ) > 1:
            return get_kernel_gauss( scale = args[1] )
        else:
            return get_kernel_gauss()
    if kernel_type == 'tri':
        if len( args ) > 1:
            return get_kernel_tri( scale = args[1] )
        else:
            return get_kernel_tri()
    if kernel_type == 'epanechnikov':
        if len( args ) > 1:
            return get_kernel_epanechnikov( scale = args[1] )
        else:
            return get_kernel_epanechnikov()
    if kernel_type == 'gauss' or kernel_type == 'gaussian' or kernel_type == 'normal':
        if len( args ) > 1:
            return get_kernel_gauss( scale = args[1] )
        else:
            return get_kernel_gauss()
    
    raise Exception( f"Unknoen kernel type: '{kernel_type}'" )

def get_kernel_family( kernel_type ):
    """Get a kernel family for a given `kernel_type`"""
    if kernel_type == 'rect':
        return get_kernel_rect
    if kernel_type == 'tri':
        return get_kernel_tri
    if kernel_type == 'epanechnikov':
        return get_kernel_epanechnikov
    if kernel_type == 'gauss' or kernel_type == 'gaussian' or kernel_type == 'normal':
        return get_kernel_gauss
    raise Exception( f"Unknoen kernel type: '{kernel_type}'" )
    
## RateFunction classes

class RateFunctionConstant:
    """Estimates a rate function for a 1-D homogeneous Poisson process by using the average rate"""
    
    def __init__( self, window ):
        """Initializes a new rate function for the given time window"""
        self._window = window
        self._rate = None
        
    def copy( self ):
        return copy.copy( self )
    
    def fit( self, X, y = None ):
        """Fit a constant rate to the data
        
        Arguments:
        X - 1-D array of event locations
        y - (ignored)
        """
        
        dt = self._window[1] - self._window[0]
        
        X_window = X[(X >= self._window[0]) & (X < self._window[1])]
        self._rate = X_window.shape[0] / dt
        
    def predict( self, X,
                 error = None,
                 error_kind = ('ci', 0.05),
                 error_kernel = ('epanechnikov', 1.) ):
        """Predict rate function at the given points
        
        Arguments:
        X - 1-D array of locations to predict at
        """
        
        if self._rate is None:
            raise Exception( 'Model not fit (no known rate)' )
        
        r_hat = self._rate * np.ones( X.shape )
        
        # TODO Include in some common base class
        if error is not None:
            
            # Determine what error strategy we're using
            if type( error ) is str:
                # Reformat string error strategies
                error = (error,)    
            if type( error ) is not tuple:
                raise Exception( f'Unsupported type for `error`: {type( error )}' )
                
            if len( error ) < 1:
                raise Exception( '`error` has no strategy specified' )
            error_strategy = error[0]
            
            if type( error_kind ) is str:
                # Reformat string error kind
                error_kind = (error_kind,)
            
            if error_strategy.lower() == 'none':
                # Don't return errors
                return r_hat
            
            if error_strategy.lower() == 'analytic':
                r_error = _rate_kernel_error_analytic( r_hat, X, error_kind )
                return r_hat, r_error
            
            if error_strategy.lower() == 'bootstrap':
                if len( error ) < 2:
                    raise Exception( "`error` strategy is 'bootstrap' but no `n` specified" )
                
                n_boot = error[1]
                
                if type( error_kernel ) is tuple:
                    # Passed in a specification rather than a kernel function; decode
                    error_kernel = get_kernel( *error_kernel )
                
                r_error = _rate_kernel_error_boot_parametric( r_hat, X, error_kernel, n_boot, error_kind )
                return r_hat, r_error
            
            raise Exception( f"Unknown error strategy {error_strategy}" )
        
        return r_hat
        

class RateFunctionKernel:
    """Estimates a rate function for a 1-D non-homogeneous Poisson process by convolving with a kernel
    """
    
    def __init__( self,
                  kernel = None,
                  kernel_family = None ):
        """Initializes a new rate function
        
        Default behavior is to use an Epanechnikov kernel with `scale` of 1
        
        Keyword arguments:
        kernel - the specific kernel function to use for fitting, or a tuple specifying a kernel;
            see `get_kernel`
        kernel_family - if set, uses this function (with parameter `scale`) to perform cross-validation
            over the `scale` parameter; alternatively, a string that determines the kernel family
            (see `get_kernel_family`)
        """
        
        if type( kernel_family ) == str:
            kernel_family = get_kernel_family( kernel_family )
        self._kernel_family = kernel_family
        
        # Placeholder for validated scale; only used if `kernel_family` is set
        self._scale_cv = None
        # Kernel will be set if specified, or set when cross-validated in `fit`
        self._kernel = None
        
        if self._kernel_family is None:
            # Kernel family isn't set; assume we want a specific kernel
            if kernel is None:
                # Default kernel is a default Epanechnikov kernel
                kernel = ('epanechnikov', 1.)
            if type( kernel ) is tuple:
                # Passed in a specification rather than a kernel function; decode
                kernel = get_kernel( *kernel )
            self._kernel = kernel
        
        # Placeholder for the data; needed for `predict` later
        self._data = None
    
    def copy( self ):
        """Make a shallow copy of this object"""
        return copy.copy( self )
    
    def fit( self, X, y = None ):
        """Fit a rate function to the data
        
        Arguments:
        X - 1-D array of event locations
        y - (ignored)
        """
        
        # Cache the data for `predict`
        self._data = X
        
        if self._kernel_family is not None:
            # TODO Perform cross-validation to determine scale
            raise NotImplementedError( 'Cross-validation not implemented' )
    
    def predict( self, X,
                 error = None,
                 error_kind = ('ci', 0.05) ):
        """Predict rate function at the given points
        
        Arguments:
        X - 1-D array of locations to predict at
        
        Keyword arguments:
        error - strategy for determining dispersion of predictions; valid entries are
            None or 'none' - (default) do not compute dispersion
            'analytic' - use analytic results (supports 'sd', 'se', and 'ci')
            ('bootstrap', n) - use parametric bootstrap with `n` realizations (supports 'ci')
        error_kind - what kind of dispersion measure to return; valid entries are
            ('ci', alpha) - alpha-level confidence interval
            'sd' - standard deviation (noise of underlying process)
            'se' - standard error (noise of estimation)
        """
        
        if self._kernel_family is not None and self._scale_cv is None:
            raise Exception( 'Kernel scale not yet determined' ) 
        if self._kernel is None:
            raise Exception( 'No kernel set' )
        if self._data is None:
            raise Exception( 'Model not fit (data unspecified)' )
        
        r_hat = _rate_kernel( self._data, X, self._kernel )
        
        if error is not None:
            
            # Determine what error strategy we're using
            if type( error ) is str:
                # Reformat string error strategies
                error = (error,)    
            if type( error ) is not tuple:
                raise Exception( f'Unsupported type for `error`: {type( error )}' )
                
            if len( error ) < 1:
                raise Exception( '`error` has no strategy specified' )
            error_strategy = error[0]
            
            if type( error_kind ) is str:
                # Reformat string error kind
                error_kind = (error_kind,)
            
            if error_strategy.lower() == 'none':
                # Don't return errors
                return r_hat
            
            if error_strategy.lower() == 'analytic':
                r_error = _rate_kernel_error_analytic( r_hat, X, error_kind )
                return r_hat, r_error
            
            if error_strategy.lower() == 'bootstrap':
                if len( error ) < 2:
                    raise Exception( "`error` strategy is 'bootstrap' but no `n` specified" )
                
                n_boot = error[1]
                
                r_error = _rate_kernel_error_boot_parametric( r_hat, X, self._kernel, n_boot, error_kind )
                return r_hat, r_error
            
            raise Exception( f"Unknown error strategy {error_strategy}" )
        
        return r_hat

## KernelRegression helpers

# ...

## KernelRegression class

class KernelRegression:
    
    def __init__( self,
                  method = 'nw',
                  kernel = None,
                  kernel_family = None ):
        """Initializes a new kernel regression fit
        
        Default behavior is to use an Epanechnikov kernel with `scale` of 1
        
        Keyword arguments:
        method - the method used to fit the regression model. Options are:
            'nw' - (Default) Nadaraya-Watson ("locally constant") estimator
            'local' - Local linear estimator
        kernel - the specific kernel function to use for fitting, or a tuple specifying a kernel;
            see `get_kernel`
        kernel_family - if set, uses this function (with parameter `scale`) to perform cross-validation
            over the `scale` parameter; alternatively, a string that determines the kernel family
            (see `get_kernel_family`)
        """
        
        self._method = method
        
        if type( kernel_family ) == str:
            kernel_family = get_kernel_family( kernel_family )
        self._kernel_family = kernel_family
        
        # Placeholder for validated scale; only used if `kernel_family` is set
        self._scale_cv = None
        # Kernel will be set if specified, or set when cross-validated in `fit`
        self._kernel = None
        
        if self._kernel_family is None:
            # Kernel family isn't set; assume we want a specific kernel
            if kernel is None:
                # Default kernel is a default Epanechnikov kernel
                kernel = ('epanechnikov', 1.)
                
            if type( kernel ) is tuple:
                # Passed in a specification rather than a kernel function; decode
                self._kernel_family = kernel[0]
                kernel = get_kernel( *kernel )
                
            self._kernel = kernel
        
        # Placeholder for the data; needed for `predict` later
        self._data_input = None
        self._data_output = None
        
        # Placeholder for fit statistics
        self._fit_stats = None
    
    def __stats_linear_smoother( self, r_hat, weights,
                                 verbose = False ):
        """Uses the linear smoother weights to determine fit statistics
        
        ...
        """
        
        # TODO Need to figure out better way of filtering out NaNs
        filter_good = ~np.any( np.isnan( weights), axis = 1 )
        
        r_hat_effective = r_hat[filter_good]
        weights_effective = weights[filter_good, :][:, filter_good]
        
        n_data = weights_effective.shape[0]
        
        # Temporary object to hold fit stats
        ret = dict()
            
        # Determine effective number of parameters
        if verbose:
            print( 'Determining dof...' )
        
        # TODO This is EXTREMELY expensive
        nu = np.trace( weights_effective )
        
        nu_twiddle = 0.
        for i in range( n_data ):
            nu_twiddle += np.sum( np.power( weights_effective[i, :], 2. ) )
        p = nu - nu_twiddle
        
        if verbose:
            print( f'    Effective dof: {nu:0.2f}' )
            print( f'    Effective p:   {p:0.2f}' )
            
        ret['dof'] = nu
        # TODO Find out proper names for these
        ret['nu_twiddle'] = nu_twiddle
        ret['p'] = p

        # Determine estimate of error variance
        # (Don't need to log, this is fast)
#         if verbose:
#             print( 'Determining error variance...' )
            
        residuals = self._data_output[filter_good] - r_hat_effective
        sq_error = np.sum( np.power( residuals, 2. ) )
        var_hat = sq_error / (n_data - p)
        
        ret['sse'] = sq_error
        ret['error_var_hat'] = var_hat
        
        # Compute LOO cross-validation error
        if verbose:
            print( 'Computing LOO CV error...' )
        
        cv_loo = (1. / n_data) * np.sum( np.power( residuals / (1. - np.diag( weights_effective )), 2. ) )
        ret['cv_loo'] = cv_loo
        
        return pd.Series( ret )
    
    def __fit_local( self,
                   equal_var = True,
                   verbose = False,
                   return_fit = False ):
        """..."""
        
        # Check for immediate errors
        
        if equal_var == False:
            raise NotImplementedError( 'Local fit: Heteroskedastic case not yet implemented' )
        
        if self._data_input is None or self._data_output is None:
            raise Exception( 'Local fit: No fit data provided' )
            
        # Use __predict_local on the fit data
        # NOTE We can't get variance yet because we haven't fit the nosie variance
        r_hat_data, _, weights_data = self.__predict_local( self._data_input,
                                                            return_weights = True,
                                                            var = None,
                                                            verbose = verbose )
        
        self._fit_stats = self.__stats_linear_smoother( r_hat_data, weights_data,
                                                        verbose = verbose )
        
        if return_fit:
            # NOW we can determine estimate of variance in regression function estimator
            var_hat = self._fit_stats['error_var_hat']
            var_r_hat_data = self.__var_linear_smoother( weights_data,
                                                         var_hat = var_hat,
                                                         equal_var = equal_var,
                                                         verbose = verbose )
            
            return r_hat_data, var_r_hat_data
    
    def __fit_nw( self,
                  equal_var = True,
                  verbose = False,
                  return_fit = False,
                  workers = None ):
        """Fit kernel regression model using Nadaraya-Watson ("locally constant") estimator
        
        ...
        """
        
        # Check for immediate errors
        
        if equal_var == False:
            raise NotImplementedError( 'NW fit: Heteroskedastic case not yet implemented' )
        
        if self._data_input is None or self._data_output is None:
            raise Exception( 'NW fit: No fit data provided' )
        
        # Use __predict_nw on the fit data
        # NOTE We can't get variance yet because we haven't fit the noise variance
        r_hat_data, _, weights_data = self.__predict_nw( self._data_input,
                                                         return_weights = True,
                                                         var = None,
                                                         verbose = verbose,
                                                         workers = workers )
        
        self._fit_stats = self.__stats_linear_smoother( r_hat_data, weights_data,
                                                        verbose = verbose )
        
        if return_fit:
            # NOW we can determine estimate of variance in regression function estimator
            var_hat = self._fit_stats['error_var_hat']
            var_r_hat_data = self.__var_linear_smoother( weights_data,
                                                         var_hat = var_hat,
                                                         equal_var = equal_var,
                                                         verbose = verbose )
            
            return r_hat_data, var_r_hat_data
    
    def fit( self, X, y, **kwargs ):
        """[n_samples, n_features]"""
        
        # Check for immediate errors
        
        if self._kernel is None:
            raise Exception( 'No kernel set or determined' )
        
        if X.shape[0] != y.shape[0]:
            raise Exception( 'Input and output must have the same number of samples' )
        
        # Cache the data for prediction later
        
        self._data_input = X
        self._data_output = y
        
        # Choose the correct method
        
        if self._method.lower() == 'nw':
            return self.__fit_nw( **kwargs )
        
        if self._method.lower() == 'local':
            return self.__fit_local( **kwargs )
        
        raise Exception( f"Unknown fit method: {self._method}" )
    
    def __var_linear_smoother( self, weights,
                               var_hat = None,
                               equal_var = True,
                               verbose = False ):
        """..."""
        
        # Catch early errors
        
        if equal_var == False:
            raise NotImplementedError( 'NW variance: Heteroskedastic case not yet implemented' )
        
        if var_hat is None:
            # Determine error variance from fitted statistics
            
            if self._fit_stats is None:
                raise Exception( 'NW predict: no fit stats for error variance' )

            if 'error_var_hat' not in self._fit_stats:
                raise Exception( 'NW predict: error variance not in fit stats' )

            var_hat = self._fit_stats['error_var_hat']
            
        var_r_hat = var_hat * np.sum( np.power( weights, 2. ), axis = 1 )
        
        return var_r_hat
    
    def __predict_local( self, X,
                         return_weights = False,
                         var = 'equal',
                         verbose = False ):
        """..."""
        
        # Catch early errors
        
        if var is not None and var.lower() != 'equal':
            raise NotImplementedError( 'Local predict: Heteroskedastic case not yet implemented' )
        
        if self._data_input is None or self._data_output is None:
            raise Exception( 'Local predict: model not fit' )
        
        if len( X.shape ) > 1:
            if X.shape[1] > 1:
                warnings.warn( 'Local predict: Input is multi-dimensional; ignoring all but first' )
                X = X[:, 0]
        
        # Determine predictions
        
        n_data = self._data_input.shape[0]
        n_predict = X.shape[0]
        
        # TODO This uses the actual matrix equations to determine weights
        weights = np.zeros( (n_predict, n_data) )
        
        it = range( n_predict )
        if verbose:
            it = tqdm( it, total = n_predict )
            
        # TODO We're not handling nan values correctly here!
            
        for i in it:
            
            xi = X[i]
            
            # i is indexing over *prediction points*
            # j is denoting *data points*
            
            # Determine kernel weighting of data centered at x
            dxj = self._data_input - xi
            Kj = self._kernel( dxj )
            
            # Determine hat matrix weights
            s2 = np.sum( Kj * np.power( dxj, 2. ) )
            s1 = np.sum( Kj * dxj )
            bj = Kj * (s2 - dxj * s1)
            
            weights[i, :] = bj / np.sum( bj )
        
        # Weight matrix is a linear smoother on the data
        if verbose:
            print( 'Making linear prediction' )
        r_hat = weights @ self._data_output
            
        # Determine variance
        if var is None:
            var_r_hat = None
        else:
            # Determine estimate of variance in regression function estimator
            var_r_hat = self.__var_linear_smoother( weights,
                                                    equal_var = True if var.lower() == 'equal' else False,
                                                    verbose = verbose )
            
        if return_weights:
            return r_hat, var_r_hat, weights
        
        return r_hat, var_r_hat
    
    def __predict_nw( self, X,
                      return_weights = False,
                      var = 'equal',
                      verbose = False,
                      workers = None ):
        """..."""
        
        use_1d = True
        
        # Catch early errors
        
        if var is not None and var.lower() != 'equal':
            raise NotImplementedError( 'NW predict: Heteroskedastic case not yet implemented' )
        
        if self._data_input is None or self._data_output is None:
            raise Exception( 'Nw predict: model not fit' )
        
        if len( X.shape ) > 1:
            if X.shape[1] == 1:
                # Make life easier by flattening the data
                X = X[:, 0]
            if X.shape[1] > 1:
                # Don't assume we can use the 1d speedup
                # TODO Vectorize higher-dimensional data
                use_1d = False
                
        # TODO Check number of dimensions for fit and predict data
        
        # Determine predictions
        
        n_data = self._data_input.shape[0]
        n_predict = X.shape[0]
        
        # Determine estimate of regression function
        denom = np.zeros( (n_predict,) )
        num = np.zeros( (n_predict,) )
        weights = np.zeros( (n_predict, n_data) )
        
#         Original method
#         # We iterate over all of the *fit data* for doing the kernel estimates
#         it = enumerate( zip( self._data_input, self._data_output ) )
#         if verbose:
#             it = tqdm( it, total = n_data )
        
#         for i, (xi, yi) in it:
#             if np.sum( np.isnan( xi ) ) > 0 or np.sum( np.isnan( yi ) ) > 0:
#                 # TODO Make ignoring NaNs a parameter
#                 weights[:, i] = np.nan

#             # Add influence of kernel centered at fit point i across all predict points
#             if use_1d:
#                 Ki = self._kernel( X - xi )
#             else:
#                 Ki = np.array( [self._kernel( Xi - xi ) for Xi in X] )
#             denom += Ki
#             num += Ki * yi
#             weights[:, i] = Ki
    
        if workers is None:
            # Non-parallel version
            
            it = enumerate( zip( self._data_input, self._data_output ) )
            if verbose:
                it = tqdm( it, total = n_data )
                
            for i, datum in it:
                cur_num, cur_denom = _predict_nw_single_step( self._kernel, X, datum,
                                                              use_1d = use_1d )
                
                num += cur_num
                denom += cur_denom
                weights[:, i] = cur_denom
                
        else:
            # Parallel version
            
            data = zip( self._data_input, self._data_output )
            # We partially apply _predict_nw_single_step with everything except the datum
            map_func = functools.partial( _predict_nw_single_step,
                                          self._kernel,
                                          X,
                                          use_1d = use_1d )
            
            # Tack on tqdm for verbose version
            it = enumerate( _xmap( map_func,
                                   data,
                                   chunksize = 100,
                                   processes = workers ) )
#             if verbose:
#                 it = tqdm( it, total = n_data )
            
            print( 'Before next' )
            print( next( it ) )
            print( 'After next' )
            
#             for i, (cur_num, cur_denom) in list( it ):
#                 num += cur_num
#                 denom += cur_denom
#                 weights[:, i] = cur_denom
            
        r_hat = num / denom
        
        # Divide by denominator in weight matrix
        # TODO Use np reshape magic to speed up
        if verbose:
            print( 'Normalizing hat matrix...' )
        for i in range( n_data ):
            weights[:, i] = weights[:, i] / denom
        
        # Determine variance
        if var is None:
            var_r_hat = None
        else:
            # Determine estimate of variance in regression function estimator
            var_r_hat = self.__var_linear_smoother( weights,
                                                    equal_var = True if var.lower() == 'equal' else False,
                                                    verbose = verbose )
            
        if return_weights:
            return r_hat, var_r_hat, weights
        
        return r_hat, var_r_hat
    
    def predict( self, X, **kwargs ):
        """..."""
        
        # TODO Error checking
        
        # Choose the correct method
        
        if self._method.lower() == 'nw':
            return self.__predict_nw( X, **kwargs )
        
        if self._method.lower() == 'local':
            return self.__predict_local( X, **kwargs )
        
        raise Exception( f"Unknown fit method: ''{self._method}'" )
        
    
#     def fit_predict( self, X, y, **kwargs ):
#         # TODO Might already predict on X as part of fit
#         # TODO Different kwargs for fit and predict?
#         self.fit( X, y, **kwargs )
#         return self.predict( X )

# TODO This is an EXTREME HACK for this to work
# From https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55

# TODO This is hella broken

_xmap_func = None

def _xmap_worker_init( func ):
    print( 'Initializing worker' )
    global _xmap_func
    _xmap_func = func
    print( _xmap_func( (np.array( [0, 0] ), 4.) ) )

def _xmap_worker( x ):
    print( 'Running worker' )
    return _xmap_func( x )

def _xmap( func, iterable,
           chunksize = 1,
           processes = None ):
    with mp.Pool( processes,
                  initializer = _xmap_worker_init,
                  initargs = (func,) ) as pool:
        return pool.imap( _xmap_worker, iterable, chunksize )

def _predict_nw_single_step( kernel, domain, datum,
                             use_1d = True ):
    
#     print( 'Predicting...' )
    
    # Unpack tuple
    xi, yi = datum
    
#     print( xi )
#     print( yi )
    
    if np.sum( np.isnan( xi ) ) > 0 or np.sum( np.isnan( yi ) ) > 0:
        return (np.nan * np.ones( domain.shape[0] ),
                np.nan * np.ones( domain.shape[0] ))
    
    # Add influence of kernel centered at fit point i across all predict points
    if use_1d:
        Ki = kernel( domain - xi )
    else:
        Ki = np.array( [kernel( Xi - xi ) for Xi in domain] )
    
#     print( Ki )
    
    denom = Ki
    num = Ki * yi
#     weights[:, i] = Ki

#     print( 'Done predicting' )

    return num, denom