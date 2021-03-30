"""A Python interface for astrocyte quantification and analysis

For details of AQuA for Matlab, see https://github.com/yu-lab-vt/AQuA
"""

## Helper functions

_default_asterisk_thresholds = [
    (0.0001, '#'),
    (0.001, '***'),
    (0.01, '**'),
    (0.05, '*')
]

def asterisk( p,
              correction = None,
              thresholds = _default_asterisk_thresholds ):
    """Returns a familiar marker for a given p-value
    
    Keyword arguments:
    correction - specify to correct asterisk thresholds for multiple
        hypotheses; values are:
        None or 'none' - raw p-values
        ('bonferroni', n) - modify threshold for FWER with `n` observations
    thresholds - list of (p_thresh, ast) pairs where
    """
    
    # Normalize formatting of `correction`
    if correction is None:
        correction = (None,)
    if type( correction ) == str:
        correction = (correction,)
    
    if len( correction ) < 1:
        raise Exception( 'No correction kind specified' )
    correction_kind = correction[0]
    
    if correction_kind.lower() == 'bonferroni':
        if len( correction ) < 2:
            raise Exception( "No `n` specified for 'bonferroni' correction" )
        n = correction[1]
        thresholds_use = [ (p_thresh / n, ast)
                           for p_thresh, ast in thresholds ]
        
    elif correction_kind is None or correction_kind.lower() == 'none':
        thresholds_use = thresholds
        
    else:
        raise Exception( f"Unknown correction kind '{correction_kind}'" )
    
    # Must be in ascending order by p-value threshold
    thresholds_use = sorted( thresholds_use,
                             key = lambda t: t[0] )
    
    for p_thresh, ast in thresholds_use:
        if p < p_thresh:
            return ast
        
    # p is >= the greatest threshold; no asterisk
    return ''