# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:29:38 2016

@author: rzchlab
"""

import re
import os
from os.path import join, basename, dirname
import glob
from functools import partial

class Targeter:
    def __init__(self, verbose=False, **kwargs):
        self.props = kwargs
        self.verbose = verbose
        
    def acquire(self, rootpath, pattern='.*', depth=1, 
                conditions_dict=None):
        """Look in rootpath and return files that match pattern, as Target 
        objects.

        Args:
            rootpath: The directory that is (nth-)parent of all directories 
                that have data in them.
            pattern: The regular expression used to filter files in rootpath 
                and its children.
            props: A dictionary whose keys are patterns that will be 
                matched against every filepath in rootpath (that matches
                'pattern').  When a match is found a target object is made
                using the value from this dict as the props parameter. If the
                value is iterable in three dimensions then a Target will be
                made for each element of the top level list. 
                If props isn't a dict, the Target objects will just get this
                exact value assigned to their props attribute.
            depth: The number of subdirectories to be searched. A depth of 1 
                means just rootpath, 2 means rootpath and subdirectories of 
                rootpath, etc. If less than 1, the value 1 will be used.

        Returns:
            list: A list of paths (strings).

        """
        # Don't allow depth < 1
        depth = 1 if depth < 1 else depth            

        msg = 'Acquiring in regex mode with pattern: "{}"'.format(pattern)
        self._p(msg)
        ismatch = lambda x: re.match(pattern, x)
        
        # Find the matching files
        potential_matches = self._walk(rootpath, depth)
        match_basenames = [basename(x) for x in potential_matches]
        self._p('Files found: {}'.format(match_basenames))
        [self._p(x) for x in match_basenames]
        filepaths = list(filter(ismatch, potential_matches))
        
        # Return the matched files, but package them as target objects first
        return filepaths

    def _p(self, x):
        """Print if in verbose mode"""
        if self.verbose:
            print(x)

    def _walk(self, rootpath, depth):
        """Return all files in rootpath and it's children down to a level
        specified by depth.
        """
        owd = os.getcwd()
        os.chdir(rootpath)
        files = []
        for i in range(depth):
            glob_pat = os.path.sep.join('*' * (i + 1))
            fs = [join(rootpath, x) for x in glob.glob(glob_pat)]
            try:
                dirs = set([os.path.relpath(dirname(x), rootpath) for x in fs])
            except AttributeError:
                dirs = set([dirname(x) for x in fs])
            self._p('Targeter looking in (d={}): {}'.format(i, dirs))
            files += [x for x in fs if os.path.isfile(x)]
        os.chdir(owd)
        return files