#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A basic data profiling and exploration toolkit.

@author: Tim Arney
@src: https://github.com/tim-arney/data-profiler
"""

import sys
import string
import math
import re
import contextlib

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from prettytable import PrettyTable


class DataProfile:
    """
    A class to provide basic data exploration and profiling tools.
    
    Methods
    -------
    categorical_validation(self, attribute: str, category_list: list, 
                           invalid_rows: int = 5) -> None:
        Assess the validity of a categorical attribute given a list of valid
        categorical values.
        
    categoricals_validation(self, categorical_dict: dict,
                            invalid_rows: int = 5) -> None:
        A convenience wrapper function around 'categorical_validation' that
        takes a dictionary of attibute - category list pairs and iteratively
        validates each attribute.  The dictionary should take the form:
            { attribute_i: [value_1, ..., value_n] }
            
    datetime_validation(self, attribute: str, dt_format: str = '%d/%m/%Y', 
                        from_dt: Optional[str] = None, to_dt: Optional[str] = None, 
                        invalid_rows: int = 5) -> None
        Assess the validity of an attribute based on its ability to be parsed
        from a specified datetime format into a datetime object, and if specified, 
        whether it falls within a particular datetime range.  A list of invalid 
        values, if found, will also be output.
    
    describe(self, attribute: Optional[str] = None, freq_rows: int = 5) -> None
        Prints a description for each attribute in terms of its data quality,
        basic statistics for numeric attributes, frequency distribution of 
        values, and frequency distribution of string lengths for non-numeric
        attributes.
        
    email_validation(self, attribute: str, invalid_rows: int = 5) -> None
        A convenience wrapper function around 'regex_validation' that specifically
        validates email addresses.
        
    float_validation(self, attribute: str, minimum: Optional[int] = None, 
                     maximum: Optional[int] = None, invalid_rows: int = 5) -> None
        Assess the validity of an attribute based on its ability to be represented
        as a floating point type, and if specified, whether it falls within a particular
        range.  A list of invalid values, if found, will also be output.
        
    get_dataframe(self) -> pandas.core.frame.DataFrame
        A convenience function to extract the pandas DataFrame from the
        DataProfile object.
        
    head(self, rows: int = 5, attribute: Optional[str] = None, verbose: bool = True) -> None
        Wrapper funtion for pd.DataFrame.tail, to show the first "x" rows of the
        DataFrame.
        
    int_validation(self, attribute: str, minimum: Optional[int] = None, 
                   maximum: Optional[int] = None, invalid_rows: int = 5) -> None
        Assess the validity of an attribute based on its ability to be represented
        as an integer type, and if specified, whether it falls within a particular
        range.  A list of invalid values, if found, will also be output.
        
    ip_validation(self, attribute: str, invalid_rows: int = 5) -> None
        A convenience wrapper function around 'regex_validation' that specifically
        validates IPv4 addresses.
        
    preamble(self) -> None
        Prints basic information such as when the data profile was built, the 
        source of the data, and the size of the data including its dimensions,
        size on disk, and size in memory.
        
    regex_validation(self, attribute: str, regex: str, invalid_rows: int = 5) -> None
        A general validation function that takes an attribute and a regular
        expression pattern and validates the attribute based on whether it 
        matches the pattern.
        
    sample(self, rows: int = 5, attribute: Optional[str] = None, verbose: bool = True) -> None
        Wrapper funtion for pd.DataFrame.sample, to show a random selection of 
        "x" rows of the DataFrame.
        
    save(self, file: Optional[str] = None) -> None
        Saves the profile to disk.  Validity measures as noted in the summary
        and attribute description will be based on the final validation for 
        that attribute.
    
    string_validation(self, attribute: str, letters: bool = True, whitespace: bool = False, 
                      digits: bool = False, punctuation: bool = False, 
                      min_length: Optional[int] = None, max_length: Optional[int] = None, 
                      invalid_rows: int = 5) -> None
        Assess the validity of a string attribute based on character set and
        string length constraints.  Will also output a list of invalid values,
        if found.  
        
    summary(self) -> None
        Prints a summary of each attribute including its inferred type, the 
        number of observations, the number of distinct observations, the number
        of valid observations if any data validation has been applied, and the
        number of missing values.
        
    tail(self, rows: int = 5, attribute: Optional[str] = None, verbose: bool = True) -> None
        Wrapper funtion for pd.DataFrame.tail, to show the last "x" rows of the
        DataFrame.
    """


    ###########################################################################
    #                          DUNDER/MAGIC METHODS                           #
    ###########################################################################

    def __init__(self, file: str, header: bool = True) -> None:
        """
        Parameters
        ----------
        file : str
            The data file to profile.
        header : bool, optional
            Whether the data file includes a header line. The default is True.

        Returns
        -------
        None

        """
        self._instantiated = datetime.now()
        self._file_path = Path(file)
        
        if not self._file_path.is_file():
            print(f"ERROR: File {self._file_path} not found, quitting...")
            sys.exit(0)
        
        self._file_name = self._file_path.name
        self._file_ext = self._file_path.suffix.lower()
        
        if self._file_ext == '.csv':
            self._df = self._read_csv(self._file_path, header)
        else:
            print(f"ERROR: File type '{self._file_ext}' not supported, quitting...")
            sys.exit(0)
        
        self._num_records, self._num_attributes = self._df.shape
        self._attributes = self._df.columns
        self._profile_history = []
        self._save_in_progress = False
        
        self._calc_summary_stats()
        
        self.preamble()
        self.summary()
     
        
    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            The string representation of the object.

        """
        return f'DataProfile({self._file_name})'
        
 
    ###########################################################################
    #                             PUBLIC METHODS                              #
    ###########################################################################   
 
    def preamble(self) -> None:
        """
        Prints basic information such as when the data profile was built, the 
        source of the data, and the size of the data including its dimensions,
        size on disk, and size in memory.

        Returns
        -------
        None

        """
        # now = datetime.now()
        # current_time = now.strftime("%c")
        size = self._file_path.stat().st_size
        mem = sys.getsizeof(self._df)
        
        DataProfile._heading('Data Profile')
        print(f"Profiled       {self._instantiated.strftime('%c')}")
        print(f'File name      {self._file_name}')
        print(f'File path      {self._file_path.parent.resolve()}')
        print(f'File size      {DataProfile._convert_size(size)}')
        print(f'Memory usage   {DataProfile._convert_size(mem)}')
        print(f"Attributes     {self._num_attributes}")
        print(f"Observations   {self._num_records:,}")
        print(f"Duplicate rows {self._num_duplicates:,}")
        
        if not self._save_in_progress:
            cmd = {'cmd': 'preamble'}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
        
        
    def summary(self) -> None:
        """
        Prints a summary of each attribute including its inferred type, the 
        number of observations, the number of distinct observations, the number
        of valid observations, if any data validation has been applied, and the
        number of missing values.

        Returns
        -------
        None

        """
        DataProfile._heading('Summary')
        table = PrettyTable()
        
        if len(self._num_valid) == 0:
            headers = ["#", "Attribute", "Type", "Observations", "Distinct", "Missing"]
        else:
            headers = ["#", "Attribute", "Type", "Observations", "Distinct", "Valid", "Missing"]
            
        table.field_names = headers
        table.align["#"] = "r"
        table.align["Attribute"] = "l"
        table.align["Type"] = "c"
        table.align["Observations"] = "r"
        table.align["Distinct"] = "r"
        table.align["Missing"] = "r"
        
        if len(self._num_valid) == 0:
            table.align["Valid"] = "r"
            
        for i, attribute in enumerate(self._attributes):
            if len(self._num_valid) == 0:
                row = [i+1,
                       attribute, 
                       self._dtype[attribute],
                       f'{self._num_observations[attribute]:,}',
                       f'{self._num_distinct[attribute]:,}',
                       f'{self._num_missing[attribute]:,}'
                       ]
            elif attribute in self._num_valid:
                row = [i+1,
                       attribute, 
                       self._dtype[attribute],
                       f'{self._num_observations[attribute]:,}',
                       f'{self._num_distinct[attribute]:,}',
                       f'{self._num_valid[attribute]:,}',
                       f'{self._num_missing[attribute]:,}'
                       ]
            else:
                row = [i+1,
                       attribute, 
                       self._dtype[attribute],
                       f'{self._num_observations[attribute]:,}',
                       f'{self._num_distinct[attribute]:,}',
                       '',
                       f'{self._num_missing[attribute]:,}'
                       ]
                
            table.add_row(row)
                          
        print(table)
        
        if not self._save_in_progress:
            cmd = {'cmd': 'summary'}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
        
        
    def describe(self, attribute: Optional[str] = None, freq_rows: int = 5) -> None:
        """
        Prints a description for each attribute in terms of its data quality,
        basic statistics for numeric attributes, frequency distribution of 
        values, and frequency distribution of string lengths for non-numeric
        attributes.

        Parameters
        ----------
        attribute : Optional[str], optional
            The attribute to describe.  If None provided, will describe all 
            attributes.  The default is None.
        freq_rows : int, optional
            The frequency distribution will be ordered from most to least
            frequent values.  The output can be restricted to only the most and 
            least frequent values by setting freq_rows > 0.  If set <= 0 all 
            rows will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        if attribute is not None:
            if self._attribute_exists(attribute):
                attributes = [attribute]
            else:
                return
        else:
            attributes = self._attributes
        

        for attr in attributes:
            self._describe(attribute=attr, freq_rows=freq_rows)
            
    
    def head(self, rows: int = 5, attribute: Optional[str] = None, verbose: bool = True) -> None:
        """
        Wrapper funtion for pd.DataFrame.tail, to show the first "x" rows of the
        DataFrame.

        Parameters
        ----------
        rows : int, optional
            How many rows to show. The default is 5.
        attribute : Optional[str], optional
            Restrict the output to a particular attribute. The default is None.
        verbose : bool, optional
            Override pandas behaviour of hiding columns to limit the display
            width. The default is True.

        Returns
        -------
        None

        """
        if attribute is None:
            DataProfile._heading('Head')
            df = self._df
        elif self._attribute_exists(attribute):
            DataProfile._heading(f'{attribute} Head')
            df = self._df[attribute]
        else:
            return
        
        if verbose:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(df.head(rows))
        else:
            print(df.head(rows))
            
        if not self._save_in_progress:
            cmd = {'cmd': 'head',
                   'rows': rows,
                   'attribute': attribute,
                   'verbose': verbose}
                   
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
        
        
    def tail(self, rows: int = 5, attribute: Optional[str] = None, verbose: bool = True) -> None:
        """
        Wrapper funtion for pd.DataFrame.tail, to show the last "x" rows of the
        DataFrame.

        Parameters
        ----------
        rows : int, optional
            How many rows to show. The default is 5.
        attribute : Optional[str], optional
            Restrict the output to a particular attribute. The default is None.
        verbose : bool, optional
            Override pandas behaviour of hiding columns to limit the display
            width. The default is True.

        Returns
        -------
        None

        """
        if attribute is None:
            DataProfile._heading('Tail')
            df = self._df
        elif self._attribute_exists(attribute):
            DataProfile._heading(f'{attribute} Tail')
            df = self._df[attribute]
        else:
            return
        
        if verbose:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(df.tail(rows))
        else:
            print(df.tail(rows))
            
        if not self._save_in_progress:
            cmd = {'cmd': 'tail',
                   'rows': rows,
                   'attribute': attribute,
                   'verbose': verbose}
                   
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
    
    
    def sample(self, rows: int = 5, attribute: Optional[str] = None, verbose: bool = True) -> None:
        """
        Wrapper funtion for pd.DataFrame.sample, to show a random selection of 
        "x" rows of the DataFrame.

        Parameters
        ----------
        rows : int, optional
            How many rows to show. The default is 5.
        attribute : Optional[str], optional
            Restrict the output to a particular attribute. The default is None.
        verbose : bool, optional
            Override pandas behaviour of hiding columns to limit the display
            width. The default is True.

        Returns
        -------
        None

        """
        if attribute is None:
            DataProfile._heading('Sample')
            df = self._df
        elif self._attribute_exists(attribute):
            DataProfile._heading(f'{attribute} Sample')
            df = self._df[attribute]
        else:
            return
        
        if rows > self._num_records:
            rows = self._num_records
        
        if verbose:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(df.sample(rows))
        else:
            print(df.sample(rows))
            
        if not self._save_in_progress:
            cmd = {'cmd': 'sample',
                   'rows': rows,
                   'attribute': attribute,
                   'verbose': verbose}
                   
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
        
        
    def string_validation(self, attribute: str, letters: bool = True, 
                          whitespace: bool = False, digits: bool = False, 
                          punctuation: bool = False, min_length: Optional[int] = None, 
                          max_length: Optional[int] = None, invalid_rows: int = 5) -> None:
        """
        Assess the validity of a string attribute based on character set and
        string length constraints.  Will also output a list of invalid values,
        if found.  For more information on the character sets see:
            
            https://docs.python.org/3/library/string.html

        Parameters
        ----------
        attribute : str
            The attribute to validate.
        letters : bool, optional
            Whether letters a-z, A-Z are legal. The default is True.
        whitespace : bool, optional
            Whether whitespace is legal, including space, tab, linefeed, return,
            formfeed, and vertical tab. The default is False.
        digits : bool, optional
            Whether digitis 0-9 are legal. The default is False.
        punctuation : bool, optional
            Whether punctuation in the C locale are considered legal. 
            The default is False.
        min_length : Optional[int], optional
            The minimum valid string length. The default is None.
        max_length : Optional[int], optional
            The maximum valid string length. The default is None.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        if not self._attribute_exists(attribute):
            return
        
        valid_chars = ''
        rule_str = ' legal chars = '
        
        if letters:
            valid_chars += string.ascii_letters
            rule_str += string.ascii_letters
        if digits:
            valid_chars += string.digits
            rule_str += string.digits
        if punctuation:
            valid_chars += string.punctuation
            rule_str += string.punctuation
        if whitespace:
            valid_chars += string.whitespace
            rule_str += ' + whitespace'
                
        if len(valid_chars) == 0:
            print("ERROR: no legal characters nominated")
            return     
        
        if min_length is not None and max_length is not None:
            min_len = min_length
            max_len = max_length
            
            if min_length == max_length:
                rule_str += f"\n num chars = {min_length}" 
            else:
                rule_str += f"\n {min_length} <= num chars <= {max_length}"
        elif min_length is not None:
            min_len = min_length
            max_len = math.inf
            
            rule_str += f"\n num chars >= {min_length}"
        elif max_length is not None:
            min_len = -math.inf
            max_len = max_length
            
            rule_str += f"\n num chars <= {max_length}"
        else:
            min_len = -math.inf
            max_len = math.inf
        
        DataProfile._heading(f'{attribute} Validation')
        DataProfile._subheading('Rule', all_caps=False, leading_line=False)
        print(rule_str)
        
        self._valid_character_set = set(valid_chars)
        
        value_df = self._df[attribute].dropna()
        
        if is_numeric_dtype(value_df):
            value_df = value_df.astype(str)
            
        valid_df = value_df.apply(self._check_string, args=(min_len, max_len))
        
        self._validity(attribute, value_df, valid_df, invalid_rows)
        
        if not self._save_in_progress:
            cmd = {'cmd': 'string_validation',
                   'attribute': attribute,
                   'letters': letters,
                   'whitespace': whitespace,
                   'digits': digits,
                   'punctuation': punctuation,
                   'min_length': min_length,
                   'max_length': max_length,
                   'invalid_rows': invalid_rows}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
            
            
    def int_validation(self, attribute: str, minimum: Optional[int] = None, 
                       maximum: Optional[int] = None, invalid_rows: int = 5) -> None:
        """
        Assess the validity of an attribute based on its ability to be represented
        as an integer type, and if specified, whether it falls within a particular
        range.  A list of invalid values, if found, will also be output.
        
        The attribute itself may not have an inferred integer type.  It may be
        that the attribute was encoded as a string, for example to retain leading
        zeroes.  It also may have an inferred floating point type if there are 
        any missing values, as pandas uses np.nan to represent missing values
        which is a float64.

        Parameters
        ----------
        attribute : str
            The attribute to validate.
        minimum : Optional[int], optional
            The minimum valid value. The default is None.
        maximum : Optional[int], optional
            The maximum valid value. The default is None.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        if not self._attribute_exists(attribute):
            return
        
        rule_str = ' integer'
        
        if minimum is not None:
            rule_str += f", {minimum} <= {attribute}"
            min_val = minimum
        else:
            min_val = -math.inf
            
        if maximum is not None:
            if minimum is None:
                rule_str += f", {attribute}"
                
            rule_str += f" <= {maximum}"
            max_val = maximum
        else:
            max_val = math.inf

             
        DataProfile._heading(f'{attribute} Validation')
        DataProfile._subheading('Rule', all_caps=False, leading_line=False)
        print(rule_str)
        
        value_df = self._df[attribute].dropna()
        valid_df = value_df.apply(self._check_int , args=(min_val,  max_val))
        self._validity(attribute, value_df, valid_df, invalid_rows)
        
        if not self._save_in_progress:
            cmd = {'cmd': 'int_validation',
                   'attribute': attribute,
                   'minimum': minimum,
                   'maximum': maximum,
                   'invalid_rows': invalid_rows}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
            
            
    def float_validation(self, attribute: str, minimum: Optional[int] = None, 
                         maximum: Optional[int] = None, invalid_rows: int = 5) -> None:
        """
        Assess the validity of an attribute based on its ability to be represented
        as a floating point type, and if specified, whether it falls within a particular
        range.  A list of invalid values, if found, will also be output.
        
        The attribute itself may not have an inferred float type, for example
        if it's encoded as a string.

        Parameters
        ----------
        attribute : str
            The attribute to validate.
        minimum : Optional[int], optional
            The minimum valid value. The default is None.
        maximum : Optional[int], optional
            The maximum valid value. The default is None.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        if not self._attribute_exists(attribute):
            return
        
        rule_str = ' float'
        
        if minimum is not None:
            rule_str += f", {minimum} <= {attribute}"
            min_val = minimum
        else:
            min_val = -math.inf
            
        if maximum is not None:
            if minimum is None:
                rule_str += f", {attribute}"
                
            rule_str += f" <= {maximum}"
            max_val = maximum
        else:
            max_val = math.inf

             
        DataProfile._heading(f'{attribute} Validation')
        DataProfile._subheading('Rule', all_caps=False, leading_line=False)
        print(rule_str)
        
        value_df = self._df[attribute].dropna()
        valid_df = value_df.apply(self._check_float , args=(min_val,  max_val))
        self._validity(attribute, value_df, valid_df, invalid_rows)
        
        if not self._save_in_progress:
            cmd = {'cmd': 'float_validation',
                   'attribute': attribute,
                   'minimum': minimum,
                   'maximum': maximum,
                   'invalid_rows': invalid_rows}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
            
            
    def datetime_validation(self, attribute: str, dt_format: str = '%d/%m/%Y', 
                            from_dt: Optional[str] = None, to_dt: Optional[str] = None, 
                            invalid_rows: int = 5) -> None:
        """
        Assess the validity of an attribute based on its ability to be parsed
        from a specified datetime format into a datetime object, and if specified, 
        whether it falls within a particular datetime range.  A list of invalid 
        values, if found, will also be output.
        
        For format string codes see:
            
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

        Parameters
        ----------
        attribute : str
            The attribute to validate.
        dt_format : str, optional
            The datetime format string. The default is '%d/%m/%Y'.
        from_dt : Optional[str], optional
            The earliest valid datetime. The default is None.
        to_dt : Optional[str], optional
            The latest valid datetime. The default is None.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        if not self._attribute_exists(attribute):
            return
        
        rule_str = f" format = '{dt_format}'"
        
        if from_dt is not None:
            try:
                start = datetime.strptime(from_dt, dt_format)
                
                rule_str += f", {from_dt} <= {attribute}"
            except ValueError:
                print(f"ERROR: from_dt='{from_dt}' does not conform to dt_format='{dt_format}'")
                return
        else:
            start = None
            
        if to_dt is not None:
            try:
                end = datetime.strptime(to_dt, dt_format)
                
                if from_dt is None:
                    rule_str += f", {attribute}"
                    
                rule_str += f" <= {to_dt}"
            except ValueError:
                print(f"ERROR: to_dt='{to_dt}' does not conform to dt_format='{dt_format}'")
                return
        else:
            end = None
             
        DataProfile._heading(f'{attribute} Validation')
        DataProfile._subheading('Rule', all_caps=False, leading_line=False)
        print(rule_str)
        
        value_df = self._df[attribute].dropna()
        valid_df = value_df.apply(self._check_datetime , args=(dt_format, start, end))
        self._validity(attribute, value_df, valid_df, invalid_rows)
        
        if not self._save_in_progress:
            cmd = {'cmd': 'datetime_validation',
                   'attribute': attribute,
                   'dt_format': dt_format,
                   'from_dt': from_dt,
                   'to_dt': to_dt,
                   'invalid_rows': invalid_rows}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
    
    
    def categorical_validation(self, attribute: str, category_list: list, invalid_rows: int = 5) -> None:
        """
        Assess the validity of a categorical attribute given a list of valid
        categorical values.

        Parameters
        ----------
        attribute : str
            The attribute to validate.
        category_list : list
            The list of valid categorical values.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed. The default is 5.

        Returns
        -------
        None

        """
        if not self._attribute_exists(attribute):
            return
        
        if len(category_list) == 0:
            return
        
        DataProfile._heading(f'{attribute} Validation')
        DataProfile._subheading('Rule', all_caps=False, leading_line=False)
        category_string = ', '.join([str(cat) for cat in category_list])
        print(f" value in [{category_string}]")
        
        value_df = self._df[attribute].dropna()
        valid_df = value_df.apply(lambda x: x in category_list)
        self._validity(attribute, value_df, valid_df, invalid_rows)
            
        if not self._save_in_progress:
            cmd = {'cmd': 'categorical_validation',
                   'attribute': attribute,
                   'category_list': category_list,
                   'invalid_rows': invalid_rows}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
       
        
    def categoricals_validation(self, categorical_dict: dict, invalid_rows: int = 5) -> None:
        """
        A convenience wrapper function around 'categorical_validation' that
        takes a dictionary of attibute - category list pairs and iteratively
        validates each attribute.  The dictionary should take the form:
            { attribute_i: [value_1, ..., value_n] }

        Parameters
        ----------
        categorical_dict : dict
            A dictionary keyed on attribute name with each value being a list
            of valid categorical values.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed. The default is 5.

        Returns
        -------
        None

        """
        if len(categorical_dict) == 0:
            return
        
        for attribute, category_list in categorical_dict.items():
            self.categorical_validation(attribute, category_list, invalid_rows)
            
            
    def email_validation(self, attribute: str, invalid_rows: int = 5) -> None:
        """
        A convenience wrapper function around 'regex_validation' that specifically
        validates email addresses.

        Parameters
        ----------
        attribute : str
            The attribute containing email addresses.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        self.regex_validation(attribute, regex, invalid_rows)
    
    
    def ip_validation(self, attribute: str, invalid_rows: int = 5) -> None:
        """
        A convenience wrapper function around 'regex_validation' that specifically
        validates IPv4 addresses.

        Parameters
        ----------
        attribute : str
            The attribute containing IP addresses.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        regex = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        
        self.regex_validation(attribute, regex, invalid_rows)
    
    
    def regex_validation(self, attribute: str, regex: str, invalid_rows: int = 5) -> None:
        """
        A general validation function that takes an attribute and a regular
        expression pattern and validates the attribute based on whether it 
        matches the pattern.
        
        For regular expression syntax see:
            
            https://docs.python.org/3/library/re.html

        Parameters
        ----------
        attribute : str
            The attribute to validate.
        regex : str
            The regular expression pattern.
        invalid_rows : int, optional
            The number of invalid values to show. If set to <= 0 then all
            invalid values will be displayed.  The default is 5.

        Returns
        -------
        None

        """
        if not self._attribute_exists(attribute):
            return
        
        try:
            self._prog = re.compile(regex)
        except (re.error, TypeError):
            print()
            print(f"ERROR: invalid regular expression '{regex}'")
            return
        
        DataProfile._heading(f'{attribute} Validation')
        DataProfile._subheading('Rule', all_caps=False, leading_line=False)
        print(f" regex = '{regex}'")
        
        value_df = self._df[attribute].dropna()
        
        if is_numeric_dtype(value_df):
            value_df = value_df.astype(str)
            
        valid_df = value_df.apply(self._check_regex) #, args=(regex,))
        
        self._validity(attribute, value_df, valid_df, invalid_rows)
            
        if not self._save_in_progress:
            cmd = {'cmd': 'regex_validation',
                   'attribute': attribute,
                   'regex': regex,
                   'invalid_rows': invalid_rows}
            
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)
    
    
    def save(self, file: Optional[str] = None) -> None:
        """
        Saves the profile to disk.  Validity measures as noted in the summary
        and attribute description will be based on the final validation for 
        that attribute.

        Parameters
        ----------
        file : Optional[str], optional
            The name of the file to write. If none is provided it will default
            to the name of the data file appended with '_profile.txt'.  
            The default is None.

        Returns
        -------
        None

        """
        if file is None:
            log_file_name = f'{self._file_path.stem}_profile.txt'
            log_file = self._file_path.parent.resolve() / log_file_name
        else:
            log_file = Path(file)
            
            if not log_file.parent.is_dir():
                log_file.parent.mkdir()
            
        self._save_in_progress = True
            
        with open(log_file, "w") as o:
            with contextlib.redirect_stdout(o):
                for cmd in self._profile_history:
                    command = cmd['cmd']
                    
                    if command == 'preamble':
                        self.preamble()
                    elif command == 'summary':
                        self.summary()
                    elif command == 'describe':
                        self.describe(attribute=cmd['attribute'], 
                                      freq_rows=cmd['freq_rows'])
                    elif command == 'head':
                        self.head(rows=cmd['rows'], 
                                  attribute=cmd['attribute'], 
                                  verbose=cmd['verbose'])
                    elif command == 'tail':
                        self.tail(rows=cmd['rows'], 
                                  attribute=cmd['attribute'], 
                                  verbose=cmd['verbose'])
                    elif command == 'sample':
                        self.sample(rows=cmd['rows'], 
                                    attribute=cmd['attribute'], 
                                    verbose=cmd['verbose'])
                    elif command == 'regex_validation':
                        self.regex_validation(attribute=cmd['attribute'], 
                                              regex=cmd['regex'], 
                                              invalid_rows=cmd['invalid_rows'])
                    elif command == 'datetime_validation':
                        self.datetime_validation(attribute=cmd['attribute'], 
                                                 dt_format=cmd['dt_format'], 
                                                 from_dt=cmd['from_dt'], 
                                                 to_dt=cmd['to_dt'], 
                                                 invalid_rows=cmd['invalid_rows'])
                    elif command == 'int_validation':
                        self.int_validation(attribute=cmd['attribute'], 
                                            minimum=cmd['minimum'], 
                                            maximum=cmd['maximum'], 
                                            invalid_rows=cmd['invalid_rows'])
                    elif command == 'float_validation':
                        self.float_validation(attribute=cmd['attribute'], 
                                              minimum=cmd['minimum'], 
                                              maximum=cmd['maximum'], 
                                              invalid_rows=cmd['invalid_rows'])
                    elif command == 'string_validation':
                        self.string_validation(attribute=cmd['attribute'], 
                                               letters=cmd['letters'], 
                                               whitespace=cmd['whitespace'],  
                                               digits=cmd['digits'], 
                                               punctuation=cmd['punctuation'],  
                                               min_length=cmd['min_length'], 
                                               max_length=cmd['max_length'], 
                                               invalid_rows=cmd['invalid_rows'])
                    elif command == 'categorical_validation':
                        self.categorical_validation(attribute=cmd['attribute'],
                                                    category_list=cmd['category_list'],
                                                    invalid_rows=cmd['invalid_rows'])
                
        self._save_in_progress = False
        
        print(f"Profile saved to:  {log_file}")
        
            
    def save_summary(self, file: Optional[str] = None) -> pd.DataFrame:
        """
        Saves the data quality summary as a CSV file.  Data quality measures 
        include completeness, uniqueness, and validity where an attribute has 
        been validated.  Returns a copy of the summary as a DataFrame.

        Parameters
        ----------
        file : Optional[str], optional
            The name of the file to write. If none is provided it will default
            to the name of the data file appended with '_summary.csv'.  
            The default is None.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing a record for each attribute along with its
            data quality measures.

        """
        if file is None:
            summary_file_name = f'{self._file_path.stem}_summary.csv'
            summary_file = self._file_path.parent.resolve() / summary_file_name
        else:
            summary_file = Path(file)
            
            if not summary_file.parent.is_dir():
                log_file.parent.mkdir()
        
        records = []
        
        if len(self._num_valid) == 0:
            dimensions = {
                'Observations': self._num_observations,
                'Distinct': self._num_distinct,
                'Missing': self._num_missing
                }
        else:
            dimensions = {
                'Observations': self._num_observations,
                'Distinct': self._num_distinct,
                'Valid': self._num_valid,
                'Missing': self._num_missing
                }
            
        for attribute in self._attributes:
            record = {
                'Attribute': attribute,
                'Type': self._dtype[attribute]
                }
            
            for label, dimension in dimensions.items():
                if attribute in dimension:
                    record[label] = dimension[attribute]
                    
            records.append(record)
        
        df = pd.DataFrame.from_records(records)
        df['Completeness'] = df['Observations'] / self._num_records
        df['Uniqueness'] = df['Distinct'] / df['Observations']
        
        if 'Valid' in df.columns:
            df.insert(4, 'Valid', df.pop('Valid'))
            df['Validity'] = df['Valid'] / df['Observations']
        
        df.to_csv(summary_file, index=False)
        print(f"Summary saved to:  {summary_file}")
        
        return df
        
            
    def get_dataframe(self) -> pd.DataFrame:
        """
        A convenience function to extract the pandas DataFrame from the
        DataProfile object.

        Returns
        -------
        pd.DataFrame
            The data.

        """
        return self._df
    
    
    ###########################################################################
    #                            INTERNAL METHODS                             #
    ###########################################################################
    
    def _read_csv(self, file_path, header):
        
        if header:
            header = 'infer'
        else:
            header = None
            
        df = pd.read_csv(file_path, header=header)
    
        if header == None:
            names = {i: f'column_{i+1}' for i in range(len(df.columns))}
            df.rename(columns=names, inplace=True)
    
        return df
    
    
    def _attribute_exists(self, attribute):
        if attribute in self._attributes:
            return True
        else:
            print(f"ERROR: attribute '{attribute}' not found")
            return False
    
    
    def _calc_summary_stats(self):
        self._dtype = defaultdict(str)
        self._num_observations = defaultdict(int)
        self._num_distinct = defaultdict(int)
        self._num_missing = defaultdict(int)
        self._num_valid = defaultdict(int)
        
        self._df = self._df.replace(r'^\s*$', np.nan, regex=True)
        
        for attribute in self._attributes:
            
            # # START REMOVE
            # tmp = self._df[attribute].dropna()
            
            # if DataProfile._is_string_series(tmp):
            #     tmp = tmp.str.lstrip("-")
            #     if tmp.str.isnumeric().all():
            #         self._df[attribute] = pd.to_numeric(self._df[attribute], errors='ignore')
            # # END REMOVE
            
            dtype = self._df[attribute].dtype
            
            if dtype == 'float64':
                tmp = self._df[attribute].dropna()
                
                if np.array_equal(tmp, tmp.astype(int)):
                    self._dtype[attribute] = 'integer'
                else:
                    self._dtype[attribute] = 'float'
            elif dtype == 'int64':
                self._dtype[attribute] = 'integer'
            else:
                self._dtype[attribute] = 'string'
    
            self._num_observations[attribute] = self._df[attribute].count()
            self._num_distinct[attribute] = self._df[attribute].nunique()
            self._num_missing[attribute] = self._num_records - self._num_observations[attribute]
            
        self._num_duplicates = self._df.duplicated().sum()
    
    
    def _describe(self, attribute=None, freq_rows=5):
        DataProfile._heading(attribute, all_caps=False)
        
        self._quality(attribute)
        
        if is_numeric_dtype(self._df[attribute]):
            self._statistics(attribute)
            
        self._frequency(attribute, rows=freq_rows)
        
        if is_string_dtype(self._df[attribute]):
            self._string_lengths(attribute, rows=freq_rows)
        
            
        if not self._save_in_progress:
            cmd = {'cmd': 'describe',
                   'attribute': attribute,
                   'freq_rows': freq_rows}
                   
            if cmd not in self._profile_history:
                self._profile_history.append(cmd)

    
    def _quality(self, attribute):
        DataProfile._subheading('Data Quality', all_caps=False, leading_line=False)
        
        # num_missing = len(self._df[self._df[attribute].isnull()])
        # present_count = len(self._df[~self._df[attribute].isnull()])
        
        # if present_count + num_missing != self._num_records:
        #     print(f'WARNING: completeness: {present_count} + {num_missing} != {self._num_records}')
        
        recs = self._num_records
        obs = self._num_observations[attribute]
        dist = self._num_distinct[attribute]
        
        miss = recs - obs
        dup = obs - dist
        
        pct_complete = round(obs / recs * 100, 2)
        pct_distinct = round(dist / obs * 100, 2)
        
        recs_str = f'{recs:,}'
        width = len(recs_str)
        
        obs_str  = f'{obs:>{width},}'
        dist_str = f'{dist:>{width},}'
        miss_str = f'{miss:>{width},}'
        dup_str  = f'{dup:>{width},}'
        pct_complete_str = f'{pct_complete:>6.2f}'
        pct_distinct_str = f'{pct_distinct:>6.2f}'
        
        print(f' Completeness: {obs_str} / {recs_str} records = {pct_complete_str}% ... {miss_str} missing') 
        print(f' Uniqueness:   {dist_str} / {obs_str} records = {pct_distinct_str}% ... {dup_str} duplicates')
        
        if attribute in self._num_valid:
            val = self._num_valid[attribute]
            inv = obs - val
            
            pct_valid = round(val / obs * 100, 2)
            
            val_str = f'{val:>{width},}'
            inv_str = f'{inv:>{width},}'
            pct_valid_str = f'{pct_valid:>6.2f}'
            
            print(f' Validity:     {val_str} / {obs_str} records = {pct_valid_str}% ... {inv_str} invalid')
            
        
    def _validity(self, attribute, value_df, valid_df, invalid_rows):
        valid = f'{attribute}_valid'
        valid_df.name = valid

        df = pd.concat([value_df, valid_df], axis=1)
        val = int(df[valid].sum())
        self._num_valid[attribute] = val
        
        DataProfile._subheading('Data Quality', all_caps=False)
        
        recs = self._num_records
        obs = self._num_observations[attribute]
        recs_str = f'{recs:,}'
        width = len(recs_str)
        
        inv = obs - val
        
        pct_valid = round(val / obs * 100, 2)
        
        val_str = f'{val:>{width},}'
        inv_str = f'{inv:>{width},}'
        obs_str = f'{obs:>{width},}'
        pct_valid_str = f'{pct_valid:>6.2f}'
        
        print(f' Validity:     {val_str} / {obs_str} records = {pct_valid_str}% ... {inv_str} invalid')
   
        if inv > 0:
            DataProfile._subheading('Invalid Values', all_caps=False)
            
            invalid_df = df[~df[valid]]
            
            if 0 < invalid_rows < inv:
                print(f'Showing a random sample of {invalid_rows} invalid values:')
                print()
                invalid_df = invalid_df.sample(invalid_rows)
                invalid_df = invalid_df.sort_index()
                
            # table = PrettyTable()
            # table.field_names = ["Index", "Value"]
            # # table.align["Value"] = "r"
            
            for index, value in invalid_df[attribute].iteritems():
                # table.add_row([index, value])
                print(f'{value}')
                
            # print(table)
        
        
    def _statistics(self, attribute):
        DataProfile._subheading('Statistics', all_caps=False)
        
        df = self._df[attribute]
        
        if df.dtype == 'float64':
            tmp = self._df[attribute].dropna()
            
            if np.array_equal(tmp, tmp.astype(int)):
                df = tmp.astype(int)
            
        desc_df = df.describe()[1:]
        
        table = PrettyTable()
        table.field_names = ["Measure", "Value"]
        table.align["Value"] = "r"
        
        for measure, value in desc_df.iteritems():
            table.add_row([measure, f'{value:,.2f}'])
            
        print(table)
        


    def _string_lengths(self, attribute, rows=5):
        DataProfile._subheading('String Lengths', all_caps=False)
        
        freq_df = self._df[attribute].dropna().astype(str).apply(len).value_counts()
        obs = self._num_observations[attribute]
        
        min_length = freq_df.index.min()
        max_length = freq_df.index.max()
        
        if min_length < max_length:
            min_str = f'{min_length:,}'
            max_str = f'{max_length:,}'
            width = len(max_str)
            
            print(f'  Min: {min_str:>{width}} characters')
            print(f'  Max: {max_str} characters')
            print()
        
        show_all = rows <= 0 or len(freq_df) <= 2 * rows
        
        table = PrettyTable()
        table.field_names = ["Length", "Freq", "%"]
        # table.align["Freq"] = "r"
        # table.align["%"] = "r"
        
        max_freq = freq_df.max()
        int_width = len(f'{max_freq:,}')
        pct_width = len(f'{max_freq // obs }') + 3
        
        if show_all:
            for value, freq in freq_df.iteritems():
                pct_freq = round(freq / obs * 100, 2)
                
                freq_str = f'{freq:,}'
                pct_str = f'{pct_freq:.2f}'
                
                table.add_row([value, f'{freq_str:>{int_width}}', f'{pct_str:>{pct_width}}'])
        else:
            print(f'Showing the {rows} most and least frequent string lengths:')
            print()
            
            for value, freq in freq_df.iloc[:rows].iteritems():
                pct_freq = round(freq / obs * 100, 2)
                
                freq_str = f'{freq:,}'
                pct_str = f'{pct_freq:.2f}'
                
                table.add_row([value, f'{freq_str:>{int_width}}', f'{pct_str:>{pct_width}}'])
        
            table.add_row(['···', '···', '···'])
            
            for value, freq in freq_df.iloc[-rows:].iteritems():
                pct_freq = round(freq / obs * 100, 2)
                
                freq_str = f'{freq:,}'
                pct_str = f'{pct_freq:.2f}'
                
                table.add_row([value, f'{freq_str:>{int_width}}', f'{pct_str:>{pct_width}}'])
        
        # table.add_row(['----', '----'])
        # table.add_row(['Total', self._num_observations[attribute]])
        
        print(table)


    def _frequency(self, attribute, rows=5):
        DataProfile._subheading('Frequency Distribution', all_caps=False)
        
        df = self._df[attribute]
        
        if df.dtype == 'float64':
            tmp = self._df[attribute].dropna()
            
            if np.array_equal(tmp, tmp.astype(int)):
                df = tmp.astype(int)
        
        freq_df = df.value_counts()
        obs = self._num_observations[attribute]
        
        show_all = rows <= 0 or len(freq_df) <= 2 * rows
        
        table = PrettyTable()
        table.field_names = ["Value", "Freq", "%"]
        # table.align["Freq"] = "r"
        # table.align["%"] = "r"
        
        max_freq = freq_df.max()
        int_width = len(f'{max_freq:,}')
        pct_width = len(f'{max_freq // obs }') + 3
        
        if show_all:
            for value, freq in freq_df.iteritems():
                pct_freq = round(freq / obs * 100, 2)
                
                freq_str = f'{freq:,}'
                pct_str = f'{pct_freq:.2f}'
                
                table.add_row([value, f'{freq_str:>{int_width}}', f'{pct_str:>{pct_width}}'])
        else:
            print(f'Showing the {rows} most and least frequent values:')
            print()
            
            for value, freq in freq_df.iloc[:rows].iteritems():
                pct_freq = round(freq / obs * 100, 2)
                
                freq_str = f'{freq:,}'
                pct_str = f'{pct_freq:.2f}'
                
                table.add_row([value, f'{freq_str:>{int_width}}', f'{pct_str:>{pct_width}}'])
        
            table.add_row(['···', '···', '···'])
            
            for value, freq in freq_df.iloc[-rows:].iteritems():
                pct_freq = round(freq / obs * 100, 2)
                
                freq_str = f'{freq:,}'
                pct_str = f'{pct_freq:.2f}'
                
                table.add_row([value, f'{freq_str:>{int_width}}', f'{pct_str:>{pct_width}}'])
        
        # table.add_row(['----', '----'])
        # table.add_row(['Total', self._num_observations[attribute]])
        
        print(table)
        
        
    def _check_regex(self, string): #, regex):
        if self._prog.fullmatch(string):
            match = True
        else:
            match = False
            
        return match
    
    
    def _check_datetime(self, dt_string, dt_format, start, end):
        try:
            dt = datetime.strptime(dt_string, dt_format)
            if start is not None:
                if dt < start:
                    return False
            if end is not None:
                if dt > end:
                    return False
            return True
        except ValueError:
            return False
       
        
    def _check_string(self, string, min_length, max_length):
        return set(string) <= self._valid_character_set and min_length <= len(string) <= max_length
    
    
    def _check_int(self, value, min_val, max_val):
        try:
            int_val = int(value)
            flt_val = float(value)
            
            if int_val != flt_val:
                return False
            
            if min_val <= int_val <= max_val:
                return True
            else:
                return False
        except ValueError:
            return False
    
    
    def _check_float(self, value, min_val, max_val):
        try:
            flt_val = float(value)
            return min_val <= flt_val <= max_val
        except ValueError:
            return False
   
     
    ###########################################################################
    #                          STATIC HELPER METHODS                          #
    ###########################################################################
    
    @staticmethod
    def _heading(header_text: str, header_width: int = 78, all_caps: bool = True) -> None:
        """
        Prints text as a 3 line header of a given width, surrounded by hash
        symbols, and a blank line before and after.

        Parameters
        ----------
        header_text : str
            Text to display.
        header_width : int, optional
            Number of characters on each line. The default is 75.
        all_caps : bool, optional
            Convert text to uppercase. The default is True.

        Returns
        -------
        None

        """
        if all_caps:
            header_text = header_text.upper()
        
        if len(header_text) > header_width - 4:
            header_text = header_text[:header_width-7] + '...'
            
        header_line = f'#{header_text:^{header_width-2}}#'    
        
        print()
        print('#' * header_width)
        print(header_line)
        print('#' * header_width)
        print()
    
    
    @staticmethod
    def _subheading(header_text: str, header_width: int = 78, all_caps: bool = True, 
                   leading_line: bool = True) -> None:
        """
        Prints text as a subheader of a given width, centred with dashes either
        side, and a blank line before (optional) and after.

        Parameters
        ----------
        header_text : str
            Text to display.
        header_width : int, optional
            Number of characters on each line. The default is 75.
        all_caps : bool, optional
            Convert text to uppercase. The default is True.
        leading_line : bool, optional
            Print a blank line before. The default is True.

        Returns
        -------
        None

        """
        if all_caps:
            header_text = header_text.upper()
        
        if len(header_text) > header_width - 6:
            header_text = header_text[:header_width-9] + '...'
            
        dash_count = (header_width - len(header_text) - 8)
        left_dash_count = dash_count // 2
        right_dash_count = dash_count - left_dash_count
            
        text_width = len(header_text) + 4
        header_line = f"  {'-'*left_dash_count}{header_text:^{text_width}}{'-'*right_dash_count}"   
        if leading_line:
            print()
        print(header_line)
        print()
    
        
    @staticmethod
    def _convert_size(size_bytes: int) -> str:
        """
        Converts a data size given in bytes to a string scaled to the
        appropriate prefix multiplier, eg. 2,000,000 bytes -> 1.91 MB.

        Parameters
        ----------
        size_bytes : int
            Number of bytes to convert.

        Returns
        -------
        str
            The prefix multiplied string with the appropriate units appended.

        """
        if size_bytes == 0:
            return "0B"
        
        units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        
        size_string = f"{s} {units[i]}"
        
        return size_string   

        
    # src: https://stackoverflow.com/a/67001213
    @staticmethod
    def _is_string_series(s : pd.Series):
        if isinstance(s.dtype, pd.StringDtype):
            # The series was explicitly created as a string series (Pandas>=1.0.0)
            return True
        elif s.dtype == 'object':
            # Object series, check each value
            return all((v is None) or isinstance(v, str) for v in s)
        else:
            return False
    

###########################################################################
#                              MAIN FUNCTION                              #
###########################################################################
    
if __name__ == '__main__':
    
    ####
    # Run using command line arguments:
    #
    # $ python3 <input_data_file> <output_profile_file>
    #
    # If no output file is given a default one will be created
    ##
    
    if len(sys.argv) > 1:
        dp = DataProfile(sys.argv[1])
        dp.describe()
        
        if len(sys.argv) > 2:
            dp.save(sys.argv[2])
        else:
            dp.save()
            
        sys.exit(0)

    ####
    # Or run directly:
    ##
    
    # file = 'data/sample_data.csv'
    # dp = DataProfile(file) 
    # dp.describe()
    # dp.save()

   
    