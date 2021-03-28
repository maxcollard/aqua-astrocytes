"""Statistical analysis of AQuA events

For Matlab AQuA library, see https://github.com/yu-lab-vt/AQuA
"""

## Imports

import warnings

import numpy as np
import pandas as pd

import scipy.stats
import scipy.integrate

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
    
## RateFunctionKernel class

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
                kernel = get_kernel_epanechnikov()
            if type( kernel ) is tuple:
                # Passed in a specification rather than a kernel function; decode
                kernel = get_kernel( *kernel )
            self._kernel = kernel
        
        # Placeholder for the data; needed for `predict` later
        self._data = None
            
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

## LocalRegression class

class LocalRegression:
    
    def __init__( self,
                  method = 'nw' ):
        
        # Instance variables
        self.method = method
    
    def __fit_nw( X, y ):
        pass
    
    def fit( X, y ):
        """[n_samples, n_features]"""
        pass
    
    def predict( X ):
        pass
    
    def fit_predict( X, y ):
        # TODO Might already predict on X as part of fit
        self.fit( X, y )
        return self.predict( X )