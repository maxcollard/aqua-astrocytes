"""Management tools for data exported from AQuA

For Matlab AQuA library, see https://github.com/yu-lab-vt/AQuA
"""

## Imports

import os
from datetime import datetime
import itertools
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

import h5py
import yaml

from tqdm import tqdm

## HiveManager helpers

def default_filename_parser( fn ):
    """Parses a filename to a dict that only contains the filename under the key 'filename'"""
    ret = dict()
    ret['filename'] = fn
    return ret

def fov_id_postprocessor( df ):
    """Labels each unique FOV for each mouse / slice with its own id"""
    
    # TODO This is such a terrible way to do this lol
    
    mice = list( sorted( set( df['mouse'] ) ) )
    mouse_slices = { mouse: list( sorted( set( df[df['mouse'].str.match( mouse )]['slice'] ) ) )
                     for mouse in mice }
    mouse_slice_fovs = { mouse: { sl: list( sorted( set( df[df['mouse'].str.match( mouse )
                                                            & (df['slice'] == sl)]['fov'] ) ) )
                                  for sl in slices }
                         for mouse, slices in mouse_slices.items() }

    fov_combinations = [ { 'mouse': mouse, 'slice': sl, 'fov': fov }
                         for mouse, slice_fovs in mouse_slice_fovs.items()
                         for sl, fovs in slice_fovs.items()
                         for fov in fovs ]
    fov_id_combination = list( zip( itertools.count( start = 1 ), fov_combinations ) )

    # TODO Would be slick to do some nifty DataFrame dict indexing

    fov_ids = np.zeros( (df.shape[0],) )
    for fov_id, combo in fov_id_combination:
        query = ( (df['mouse'] == combo['mouse'])
                  & (df['slice'] == combo['slice'])
                  & (df['fov'] == combo['fov']) )
        fov_ids[query] = fov_id
        
    return fov_ids.astype( int )

def _postprocess_label( df, labels ):
    # TODO This is pretty slow
    
    df_labels = []
    for i_row, row in df.iterrows():
        
#         print( f'{row}' )
        
        # Determine which label matches
        cur_label = None
        for i_label, label in labels.iterrows():
#             print( f'Label {i_label}' )
            does_match = True
            for k in label.keys():
                if k == 'label':
                    # We don't try to match the 'label' key because this is special
                    continue
                if k not in row.keys():
#                     print( f'{k} Label key not in datum' )
                    does_match = False
                    break
                if row[k] != label[k]:
#                     print( f'{k}: Label value does not match datum value' )
                    does_match = False
                    break
            
            if does_match:
#                 print( 'Match found!' )
                # This label matches; set current label to this value
                cur_label = label['label']
                break
        
        df_labels.append( cur_label )
    
    return df_labels

def get_label_postprocessor( labels ):
    """Returns a postprocessor df -> df_labels that assigns the first value of the 'label'
    field in `labels` that matches a query for the rest of the keys in `labels` when applied
    to `df`.
    """
    return lambda df: _postprocess_label( df, labels )

def load_labels( fn, key ):
    """Loads a YAML label file to be used with `get_label_postprocessor`
    
    Arguments:
    fn - Filename for the label YAML file; see example
    key - Key in the 'labels' section to be used for this particular labeler
    """
    # Load the raw data for the YAML labels
    with open( fn, 'r' ) as file:
        ret_raw = yaml.safe_load( file )
    
    # Pull out the key we want from the labels
    for entry in ret_raw:
        entry['label'] = entry['labels'][key]
        del entry['labels']
    
    # Turn into a DataFrame
    return pd.DataFrame( ret_raw )

def _postprocess_conditional( df, key, spec ):
    # TODO This is *EXTREMELY* slow
    
    if key not in df.keys():
        # The key we're supposed to go off of isn't in the data
        return None
    
    ret = np.zeros( (df.shape[0],) )
    for i, row in df.iterrows():
        cur_condition = row[key]
        
        if cur_condition not in spec:
            ret[i] = None
            continue
            
        condition_key = spec[cur_condition]
        if condition_key not in row.keys():
            ret[i] = None
            continue
        
        ret[i] = row[condition_key]
    
    return ret

def get_conditional_postprocessor( key, spec ):
    """The value in the `key` column determines which column in `spec` is used
    
    Arguments:
    key - the key being used as the "switch" for conditional labeling
    spec - a dict of form `value` -> `column`, where `value` is a value of the `key` column and
        `column` is the corresponding column to use for that row
    """
    return lambda df: _postprocess_conditional( df, key, spec )

def _postprocess_transform( df, key,
                            invert = False,
                            binarize = False,
                            log = False ):
    
    if key not in df.keys():
        return None
    
    cur_data = df[key]
    
    if invert:
        cur_data = -cur_data
    
    if binarize:
        return (cur_data > 0).astype( float )
    
    if log:
        # TODO Slow
        ret = np.zeros( cur_data.shape )
        for i, x in enumerate( cur_data ):
            if x <= 0:
                ret[i] = None
            else:
                ret[i] = np.log10( x )
        return ret
    
    # Do not binarize or log-transform
    return cur_data

def get_transform_postprocessor( key,
                                 invert = False,
                                 binarize = False,
                                 log = False ):
    """Transform the values in the `key` column
    
    NOTE That if no keyword flags are set the `key` column will be returned unaltered
    
    Arguments:
    key - the column in the DataFrame to transform
    
    Keyword flags:
    invert - take the negative of the data
    binarize - return 1.0 if the data is > 0 and 0.0 otherwise; short-circuits log
    log - take the base-10 logarithm of the data
    """
    return lambda df: _postprocess_transform( df, key,
                                              invert = invert,
                                              binarize = binarize,
                                              log = log )

def _postprocess( df, postprocessors ):
    """Add columns (in place) defined by postprocessor functions"""
    for key, postprocessor in postprocessors:
        result = postprocessor( df )
        
        if result is None:
            # Result is unobtainable for some reason; ignore this postprocessor
            continue
        
        df[key] = result

## HiveManager class

class HiveManager:
    """Manages a Pythonized AQuA data hive"""
    
    def __init__( self, path,
                  parser = None,
                  extension = '.mat' ):
        """Initializes a hive for the given path
        
        Arguments:
        path - The base directory of the hive (relative or absolute)
        
        Keyword arguments:
        parser - A function of signature str -> dict that takes in a filename from the hive
            and returns the features that are included when constructing the `datasets` property.
            Default: `default_filename_parser`, which just puts the entire filename into a key
            called 'filename'
        extension - The file extension of the datasets in the hive, including the dot.
            Default: '.mat'
        """
        
        # Set instance variables
        self.path = path
        self.extension = extension
        
        self.parser = default_filename_parser if parser is None else parser
        
        self.__dataset_postprocessors = []
        self.__event_postprocessors = []
    
#     @property
#     def dataset_postprocessors( self ):
#         """Each value constructs a new column for `datasets` based on the pre-existing values
#         in the DataFrame; each key is the name of the newly created column"""
#         return self.__dataset_postprocessors
    
    def add_dataset_postprocessor( self, key, postprocessor ):
        """Add a new dataset postprocessor to the chain / update an existing postprocessor entry
        
        Arguments:
        key - The name of the column created by the postprocessor
        postprocessor - A function df -> iterable that constructs a new column based on the
            existing data in the `datasets` DataFrame
        """
        self.__dataset_postprocessors.append( (key, postprocessor) )
    
    # TODO Include some way to remove postprocessors
#     def remove_dataset_postprocessor( self, key ):
#         """Remove the dataset postprocessor at the given key, if it exists"""
#         if key in self.__dataset_postprocessors:
#             del self.__dataset_postprocessors[key]
    
    def add_event_postprocessor( self, key, postprocessor ):
        """Add a new event postprocessor to the chain / update an existing postprocessor entry
        
        Arguments:
        key - The name of the column created by the postprocessor
        postprocessor - A function df -> iterable that constructs a new column based on the
            existing data in a loaded events DataFrame
        """
        self.__event_postprocessors.append( (key, postprocessor) )
        
    # TODO Include some way to remove postprocessors
#     def remove_event_postprocessor( self, key ):
#         """Remove the event postprocessor at the given key, if it exists"""
#         if key in self.__event_postprocessors:
#             del self.__event_postprocessors[key]
    
    def add_event_laterality_postprocessors( self ):
        """Convenience method to properly align lateral / medial
        
        Assumes that the event data have the 'left_right' column with two possible values:
        - 'LM': lateral is on the left, medial on the right
        - 'ML': medial on the left, lateral on the right
        """
        grow_lateral_spec = {
            'LM': 'mark_propGrowLeft',
            'ML': 'mark_propGrowRight'
        }
        grow_medial_spec = {
            'LM': 'mark_propGrowRight',
            'ML': 'mark_propGrowLeft'
        }
        shrink_lateral_spec = {
            'LM': 'mark_propShrinkLeft',
            'ML': 'mark_propShrinkRight'
        }
        shrink_medial_spec = {
            'LM': 'mark_propShrinkRight',
            'ML': 'mark_propShrinkLeft'
        }
        
        self.add_event_postprocessor( 'mark_propGrowLateral',
                                      get_conditional_postprocessor( 'left_right', grow_lateral_spec) )
        self.add_event_postprocessor( 'mark_propGrowMedial',
                                      get_conditional_postprocessor( 'left_right', grow_medial_spec) )
        self.add_event_postprocessor( 'mark_propShrinkLateral',
                                      get_conditional_postprocessor( 'left_right', shrink_lateral_spec) )
        self.add_event_postprocessor( 'mark_propShrinkMedial',
                                      get_conditional_postprocessor( 'left_right', shrink_medial_spec) )
    
    def add_standard_event_postprocessors( self ):
        """Add a standard postprocessing pipeline"""
        
        # Overwrite the "Shrink" marks with their negation (to make them positive)
        neg_keys = ['mark_propShrinkAnterior',
                    'mark_propShrinkLeft',
                    'mark_propShrinkPosterior',
                    'mark_propShrinkRight',
                    'mark_propShrinkLateral',
                    'mark_propShrinkMedial']
        for k in neg_keys:
            self.add_event_postprocessor( k, get_transform_postprocessor( k, invert = True ) )

        # Add binary versions of propagation / shrink
        bin_keys = ['mark_propGrowAnterior',
                    'mark_propGrowLeft',
                    'mark_propGrowPosterior',
                    'mark_propGrowRight',
                    'mark_propShrinkAnterior',
                    'mark_propShrinkLeft',
                    'mark_propShrinkPosterior',
                    'mark_propShrinkRight',
                    'mark_propGrowLateral',
                    'mark_propGrowMedial',
                    'mark_propShrinkLateral',
                    'mark_propShrinkMedial']
        for k in bin_keys:
            self.add_event_postprocessor( f'{k}_bin', get_transform_postprocessor( k, binarize = True ) )

        # Add long transforms
        log_keys = ['mark_area',
                    'mark_decayTau',
                    'mark_dffMax',
                    'mark_dffMax2',
                    'mark_fall91',
                    'mark_nOccurSameLoc',
                    'mark_nOccurSameLocSize',
                    'mark_peri',
                    'mark_propGrowAnterior',
                    'mark_propGrowLeft',
                    'mark_propGrowPosterior',
                    'mark_propGrowRight',
                    'mark_propShrinkAnterior',
                    'mark_propShrinkLeft',
                    'mark_propShrinkPosterior',
                    'mark_propShrinkRight',
                    'mark_propGrowLateral',
                    'mark_propGrowMedial',
                    'mark_propShrinkLateral',
                    'mark_propShrinkMedial',
                    'mark_rise19',
                    'mark_width11',
                    'mark_width55']
        for k in log_keys:
            self.add_event_postprocessor( f'{k}_log', get_transform_postprocessor( k, log = True ) )
    
    @property
    def datasets( self ):
        """Returns a DataFrame containing information about all the datasets in the hive"""
        
        dataset_specs = []
        
        for i_dataset, candidate_filename in enumerate( sorted( os.listdir( self.path ) ) ):
            # Check for correct extension
            name, ext = os.path.splitext( candidate_filename )
            if ext != self.extension:
                continue
                
            # Get parser results
            cur_spec = self.parser( candidate_filename )
            if cur_spec is None:
                # Parser filtered out this file
                continue
            
            # Add sequential 1-indexed dataset_id to record
            cur_spec.update( {'dataset_id': i_dataset + 1} )
            
            dataset_specs.append( cur_spec )
        
        ret = pd.DataFrame( dataset_specs )
        
        # Add columns defined by postprocessor functions
#         for key, postprocessor in self.postprocessors.items():
#             ret[key] = postprocessor( ret )
        _postprocess( ret, self.__dataset_postprocessors )
    
        return ret
    
    def __get_dataset_keys( self, dataset, dataset_keys ):
        ret_dataset = None
        ret_keys = []
        
        if type( dataset ) == str:
            # String filename provided; determine dataset using parser
            ret_dataset = self.parser( dataset )
        else:
            ret_dataset = dataset
        
        if dataset_keys is None:
            # Default behavior is to merge all parts of the dataset spec except the filename
            ret_keys = [k for k in dataset.keys() if k != 'filename']
        else:
            # Check to make sure the desired keys are in the dataset
            for k in dataset_keys:
                if k not in dataset.keys():
                    raise Exception( f'Merge key "{k}" not in dataset' )
                ret_keys.append( k )
        
        return ret_dataset, ret_keys
    
    def load_events( self, dataset,
                     dataset_keys = None,
                     header_keys = None,
                     postprocess = True ):
        """Load the events from a given dataset
        
        Returns:
        header - overall information about the dataset
        events - DataFrame, each row is an astrocyte event
        
        Arguments:
        dataset - a string denoting the filename in the hive to load, or a Series containing
            a 'filename' key
        
        Keyword arguments:
        dataset_keys - a list of keys from the `dataset` Series to merge into all entries
            of the returned data. Default behavior is to include all elements of `dataset`
            except for 'filename'
        header_keys - a list of keys from the dataset header to merge into all entries of
            the returned data. Default behavior is to include none
        postprocess - set to False to disable event postprocessing. Default: True
        """
        
#         if type( dataset ) == str:
#             # String filename provided; determine dataset using parser
#             dataset = self.parser( dataset )
        
#         if dataset_keys is None:
#             # Default behavior is to merge all parts of the dataset spec except the filename
#             dataset_keys = [k for k in dataset.keys() if k != 'filename']
#         else:
#             # Check to make sure the desired keys are in the dataset
#             for k in dataset_keys:
#                 if k not in dataset.keys():
#                     raise Exception( f'Merge key "{k}" not in dataset' )
        
        dataset, dataset_keys = self.__get_dataset_keys( dataset, dataset_keys )
    
        if header_keys is None:
            header_keys = []
    
        if 'filename' not in dataset.keys():
            # Can't load file if filename not specified
            raise Exception( 'No dataset filename specified' )
        
        header = dict()
        marks = dict()
        ret_dict = dict()
        
        # TODO More intelligent way to do this?
        # These keys are specially handled
        system_keys = ['fs', 'Ts', 'eventFrames', 'eventCells']
        
        # Load data from file
        file_path = os.path.join( self.path, dataset['filename'] )
        with h5py.File( file_path, 'r' ) as file:
            
            # Determine sampling rate from data in file
            if 'fs' in file:
                header['fs'] = file['fs'][0, 0]
                header['Ts'] = 1. / header['fs']
            elif 'Ts' in file:
                header['Ts'] = 1. / file['Ts'][0, 0]
                header['fs'] = 1. / header['Ts']
            else:
                header['fs'] = None
                header['Ts'] = None

            event_frames = file['eventFrames'][0, :].astype( int )
            event_cells = file['eventCells'][0, :].astype( int )
            
            # Determine marks and headers
            for k in file.keys():
                # If key starts with 'mark_', it's a mark
                if 'mark_' in k:
                    marks[k] = file[k][:, :].flatten()
                # Any non-mark key that isn't a "special" key is a header
                elif k not in system_keys:
                    header[k] = file[k][:, :]
        
        n_cells = np.max( event_cells ) + 1
        n_events = len( event_frames )
        
        event_times = None if header['Ts'] is None else header['Ts'] * event_frames
        
        ret_dict['cell'] = []
        ret_dict['start_frame'] = []
        if event_times is not None:
            ret_dict['start_time'] = []
        for k in marks.keys():
            ret_dict[k] = []
        
        for i_event in range( n_events ):
            ret_dict['start_frame'].append( event_frames[i_event] )
            ret_dict['cell'].append( event_cells[i_event] )
            if event_times is not None:
                ret_dict['start_time'].append( event_times[i_event] )
            for k in marks.keys():
                ret_dict[k].append( marks[k][i_event] )
        
        ret = pd.DataFrame.from_dict( ret_dict )
        
        for k in dataset_keys:
            ret[k] = dataset[k]
            
        for k in header_keys:
            ret[k] = header[k][0, 0]
        
        if postprocess:
            _postprocess( ret, self.__event_postprocessors )
        
        return header, ret
    
    # TODO Break raster computation out of HiveManager; it should be its own thing
    def load_raster( self, dataset,
                     dataset_keys = None,
                     header_keys = None,
                     bin_width = None):
        """Construct a raster from the events in a dataset
        
        Returns:
        header - overall information about the dataset
        raster - DataFrame, each row is one time bin
        
        Arguments:
        dataset - a string denoting the filename in the hive to load, or a Series containing
            a 'filename' key
        
        Keyword arguments:
        dataset_keys - a list of keys from the `dataset` Series to merge into all entries
            of the returned data. Default behavior is to include all elements of `dataset`
            except for 'filename'
        header_keys - a list of keys from the dataset header to merge into all entries of
            the returned data. Default behavior is to include none
        bin_width - the width of each time bin in the raster (in seconds). Default: one frame.
        """
        
        dataset, dataset_keys = self.__get_dataset_keys( dataset, dataset_keys )
        
        if header_keys is None:
            header_keys = []
        
        if 'filename' not in dataset.keys():
            # Can't load file if filename not specified
            raise Exception( 'No dataset filename specified' )
        
        # First step is to extract events
        # TODO Find a way to speed this up when extracting both at the same time
        event_header, events = self.load_events( dataset, dataset_keys = [], postprocess = False )
        
        # Copy over the event headers
        header = dict()
        header.update( event_header )
        # Note down the bin width
        header['bin_width'] = bin_width
        
        n_cells = np.max( events['cell'] )
        
        bin_centers = None
        if bin_width is None:
            frame_max = np.max( events['start_frame'] ) + 1
            bin_edges_frames = np.arange( 0, frame_max + 1 ) - 0.5
            bin_centers_frames = bin_edges_frames[:-1] + 0.5 * np.diff( bin_edges_frames )
            
            if 'Ts' in event_header.keys():
                bin_centers = event_header['Ts'] * bin_centers_frames
            
            cell_event_frames = [ events[events['cell'] == cell]['start_frame']
                                  for cell in range( 1, n_cells + 1 ) ]
            
            cell_raster = np.zeros( (n_cells, bin_centers_frames.shape[0]) )
            for i_cell, fs in enumerate( cell_event_frames ):
                cell_raster[i_cell, :], _ = np.histogram( fs, bin_edges_frames )
        else:
            if 'start_time' not in events.keys():
                raise Exception( 'Cannot make raster with no event times' )
            t_max = np.max( events['start_time'] ) + bin_width
            bin_edges = np.arange( 0, t_max, bin_width )
            bin_centers = bin_edges[:-1] + 0.5 * np.diff( bin_edges )
        
            cell_event_times = [ events[events['cell'] == cell]['start_time']
                                 for cell in range( 1, n_cells + 1 ) ]
            
            cell_raster = np.zeros( (n_cells, bin_centers.shape[0]) )
            for i_cell, ts in enumerate( cell_event_times ):
                cell_raster[i_cell, :], _ = np.histogram( ts, bin_edges )
                
        ret_dict = dict()
        ret_dict['cell'] = []
        ret_dict['event_count'] = []
        if bin_centers is not None:
            ret_dict['center_time'] = []
        if bin_width is None:
            ret_dict['center_frame'] = []
        
        for i_cell in range( cell_raster.shape[0] ):
            for i_bin in range( cell_raster.shape[1] ):
                ret_dict['cell'].append( i_cell + 1 )
                ret_dict['event_count'].append( int( cell_raster[i_cell, i_bin] ) )
                if bin_centers is not None:
                    ret_dict['center_time'].append( bin_centers[i_bin] )
                if bin_width is None:
                    ret_dict['center_frame'].append( int( bin_centers_frames[i_bin] ) )
        
        ret = pd.DataFrame.from_dict( ret_dict )
        
        for k in dataset_keys:
            ret[k] = dataset[k]
            
        for k in header_keys:
            ret[k] = header[k][0, 0]
        
#         if postprocess:
#             _postprocess( ret, self.__event_postprocessors )
        
        return header, ret
    
    def iter_dataset_events( self, dataset_keys = None ):
        """Generates an iterator of the form (dataset, header, events) where
        
        dataset - the parsed specification of the current dataset's filename
        header - overall information about the dataset
        events - DataFrame, each row is an astrocyte event
        
        Keyword arguments:
        dataset_keys - keys from output of `datasets` to merge into the `events`
            output (see `load_events`)
        """
        
        return ( (dataset,) + self.load_events( dataset, dataset_keys = dataset_keys )
                 for _, dataset in self.datasets.iterrows() )
    
    def all_events( self, dataset_keys = None, verbose = False ):
        """Returns all events for all datasets in the hive (from `iter_dataset_events`)
        concatenated vertically
        
        Returns: (headers, events)
        headers - list of header data from each dataset
        events - DataFrame, each row is an astrocyte event. A column 'cell_id' is added, which
            generates a unique identifier for each cell.
        
        Keyword arguments:
        dataset_keys - keys from output of `datasets` to merge into the `events`
            output (see `load_events`)
        verbose - if True, prints out the name of each processed file
        """
        
        headers = []
        ret = None
        
        # We're going to create a column 'cell_id' that is going to hold the "overall" cell counter
        cur_cell_overall = 0
        
        it = self.iter_dataset_events( dataset_keys = dataset_keys )
        if verbose:
            it = tqdm( it, total = len( self.datasets ) )
        
        for dataset, header, events in it:
            if verbose:
                it.set_description( f"Loading {dataset['filename']}..." )
            
            headers.append( header )
            
            events['cell_id'] = events['cell'] + cur_cell_overall
            
            if ret is None:
                ret = events
            else:
                ret = ret.append( events, ignore_index = True )
            
            cur_cell_overall += np.max( events['cell'] )
        
        return headers, ret
    
#     @property
#     def iter_rasters( self ):
#         return (i, load_raster( row ) for i, row in self.)

## Additional parsers

# Example from file format used in Michelle's uncaging data

def _parse_standard_part( part, spec ):
    ret = dict()
    
    spec_kind = spec[0]
    
    if spec_kind == 'suffix':
        # Match against the given suffix
        suffix = spec[1]
        if part != suffix:
            # Suffix does not match; exclude this file
            return None

    elif spec_kind == 'date':
        # Extract the date using the given format
        key = spec[1]
        date_format = spec[2]
        ret['date'] = datetime.strptime( part, date_format )

    elif spec_kind == 'raw':
        # Pull out exactly the string in the part
        key = spec[1]
        ret[key] = part

    elif spec_kind == 'number':
        # Find the desired integer in the part
        key = spec[1]
        # Default to first integer
        index = 0 if len( spec ) < 3 else spec[2]
        
        numbers = re.findall( '\d+', part )
        if len( numbers ) == 0:
            # No number found
            ret[key] = None
        else:
            # Hold onto the desired number
            ret[key] = int( numbers[index] )

    elif spec_kind == 'switch':
        # Look for matches against a set of possibilities
        key = spec[1]
        labels = spec[2]
        
        match = None
        for part_label, data_label in labels.items():
            if part_label in part:
                match = data_label
                break
        
        if match is None:
            ret[key] = None
        else:
            ret[key] = match
        
    return ret

def _parse_standard( fn, spec, split ):
    ret = dict()
    ret['filename'] = fn
    
    # Split name into its parts
    name, ext = os.path.splitext( fn )
    name_split = name.split( split )
    
    # Label the split parts
    name_spec = zip( name_split, spec )
    
    # Parse each part
    for cur_part, cur_specs in name_spec:
        if cur_specs is None:
            # Ignore this part
            continue
        
        if type( cur_specs ) != list:
            # Normalize specs by turning all specs into a list
            cur_specs = [cur_specs]
        
        for cur_spec in cur_specs:
            try:
                cur_parsed = _parse_standard_part( cur_part, cur_spec )
            except Exception as e:
                # Failed to parse this part; keep going
                print( e )
                continue
            
            if cur_parsed is None:
                # Signal to short-circuit (filter)
                return None
            
            # Add the new information into the record we're building
            ret.update( cur_parsed )
    
    return ret

def standard_filename_parser( spec, split = '_' ):
    """Parser that takes individual components by splitting each filename in the hive using
    a specified separator string
    
    Arguments:
    spec - a list specifying how each part of the filename should be parsed; see examples
    
    Keyword arguments:
    split - string to separate parts of the filename to be processed. Default: '_'
    """
    return lambda fn: _parse_standard( fn, spec, split )