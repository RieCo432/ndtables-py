from unittest import TestCase
from src import ndtable
import numpy as np


# Test Cases Motivation
#
# Most of these test cases create a table structure as it would have been useful in a previous project of mine, found at
# https://github.com/RieCo432/cs4040
#
# In a nutshell, the project consisted in running different pathfinding algorithms on a variety of different maps.
# These maps varied in their size and their density of hard and soft obstacles. An n-dimensional table would have been
# useful to store the runtimes by size, hard obstacle density, soft obstacle density, algorithm used, map number and run
# number (as for every combination of size, hard obstacle density and soft obstacle density, 10 maps were generated and
# each algorithm was run on each map 25 times). Average runtimes for any combination of criteria could then be generated
# by averaging the runtimes in the corresponding slice of this table, using as few or as many headers as desired.
#
# However, to keep the complexity of these test cases low, fewer combinations of data will be used.

def construct_basic_ndtable():
    # 3 dimensions, with 3 headers each
    ndtable_shape = (3, 3, 3)

    # one dimension is map size and includes sizes "small", "medium" and "large"
    # one dimension is hard obstacle density and includes none, sparse and dense
    # one dimension is soft obstacle density and includes none, sparse and dense
    ndtable_headers = {"size": ["small", "medium", "large"],
                       "hard": ["none", "sparse", "dense"],
                       "soft": ["none", "sparse", "dense"]}

    # create nd-table
    test_ndtable = ndtable.ndtable(shape=ndtable_shape, headers=ndtable_headers, dtype=int)

    # fill nd-table with  values from 0 to 26
    test_ndtable._data = np.reshape(np.arange(0, 27), ndtable_shape)

    return test_ndtable


class Testndtable(TestCase):

    def test_initialization(self):

        # 3 dimensions, with 3 headers each
        ndtable_shape = (3, 3, 3)

        # one dimension is map size and includes sizes "small", "medium" and "large"
        # one dimension is hard obstacle density and includes none, sparse and dense
        # one dimension is soft obstacle density and includes none, sparse and dense
        ndtable_headers = {"size": ["small", "medium", "large"],
                           "hard": ["none", "sparse", "dense"],
                           "soft": ["none", "sparse", "dense"]}

        # create nd-table
        test_ndtable = ndtable.ndtable(shape=ndtable_shape, headers=ndtable_headers, dtype=int)

        # the data portion of the nd-table should be an array of all zeros
        self.assertTrue(np.all(np.equal(test_ndtable._data, np.zeros(ndtable_shape, dtype=int))))

        # the indexing of the nd-table should look like this
        #
        # size: dim    : 0
        #       headers: small : 0
        #                medium: 1
        #                large : 2
        # hard: dim    : 1
        #       headers: none  : 0
        #                sparse: 1
        #                dense : 2
        # soft: dim    : 2
        #       headers: none  : 0
        #                sparse: 1
        #                dense : 2
        #
        # This denotes that size is the first dimension of the table and its headers are, in order, small, medium, large
        # hard obstacle density is the second dimension of the table and its headers are, in order, none, sparse, dense
        # soft obstacle density is the third dimension of the table and its headers are, in order, none, sparse, dense

        expected_indexing_dict = {
            "size": {
                "dim": 0,
                "headers": {
                    "small": 0,
                    "medium": 1,
                    "large": 2
                }
            },
            "hard": {
                "dim": 1,
                "headers": {
                    "none": 0,
                    "sparse": 1,
                    "dense": 2
                }
            },
            "soft": {
                "dim": 2,
                "headers": {
                    "none": 0,
                    "sparse": 1,
                    "dense": 2
                }
            }
        }

        self.assertEqual(expected_indexing_dict, test_ndtable._indexing)

        # these are the expected unique headers: small, medium, large
        # all of them should belong to the first dimension (size) and should have the following indexes: 0, 1, 2

        expected_unique_headers = {"small": {"dim_label": "size", "index": 0},
                                   "medium": {"dim_label": "size", "index": 1},
                                   "large": {"dim_label": "size", "index": 2}}
        self.assertEqual(expected_unique_headers, test_ndtable._unique_headers)

        # these are the expected non-unique headers: none, sparse, dense
        expected_non_unique_headers = ["none", "sparse", "dense"]
        self.assertEqual(expected_non_unique_headers, test_ndtable._non_unique_headers)

    def test_error_non_unique_header_used_without_specifying_dimension(self):
        # test to see if specifying a header that cannot be uniquely attributed to any dimension will raise the
        # correct Error

        test_ndtable = construct_basic_ndtable()

        # since neither none nor sparse can be uniquely attributed to any dimension of the table, trying to retrieve
        # values using the headers without specifying a dimension should raise an IndexError with a message that
        # specifies the problematic headers.
        # However, large is uniquely attributable to the size dimension, so it will not be included in the error message
        with self.assertRaises(IndexError, msg='Header(s) ("none") cannot be uniquely attributed to a dimension.'):
            test_ndtable.get("none")

        with self.assertRaises(IndexError, msg='Header(s) ("none", "sparse") cannot be uniquely attributed to a dimension.'):
            test_ndtable.get("none", "sparse")

        with self.assertRaises(IndexError, msg='Header(s) ("none", "sparse") cannot be uniquely attributed to a dimension.'):
            test_ndtable.get("none", "sparse", "large")

    def test_error_non_exisiting_headers(self):
        # test to see if specifying non-existing headers will raise the correct Error

        test_ndtable = construct_basic_ndtable()

        # header1 does not exist
        with self.assertRaises(IndexError, msg='Header(s) ("header1") do not exist.'):
            test_ndtable.get("header1")

        # header2 and header3 do not exist either
        with self.assertRaises(IndexError, msg='Header(s) ("header2", "header3") do not exist.'):
            test_ndtable.get("header2", "header3")

        # header4 does not exist but large does
        with self.assertRaises(IndexError, msg='Header(s) ("header4") do not exist.'):
            test_ndtable.get("header4", "large")

    def test_error_non_existing_dimensions(self):
        # test to see if specifying non-existing dimensions will raise the correct Error

        test_ndtable = construct_basic_ndtable()

        # colour does not exist
        with self.assertRaises(IndexError, msg='Dimension(s) ("colour") do not exist.'):
            test_ndtable.get(colour="some_header")

        # algorithm and complexity do not exist
        with self.assertRaises(IndexError, msg='Dimension(s) ("algorithm", "complexity") do not exist.'):
            test_ndtable.get(algorithm="some_algorithm", complexity="some_complexity")

        # complexity does not exist, but size does
        with self.assertRaises(IndexError, msg='Dimension(s) ("complexity") do not exist.'):
            test_ndtable.get(complexity="some_complexity", size="large")

    def test_error_non_existing_headers_for_dimension(self):
        # test to see if specifying a dimension and header will raise the correct Error if that header does not exist
        # for that dimension

        test_ndtable = construct_basic_ndtable()

        # size:tiny does not exist
        with self.assertRaises(IndexError, msg='Dimension:Header pair(s) ("size:tiny") do not exist.'):
            test_ndtable.get(size="tiny")

        # size:huge and hard:zero do not exist
        with self.assertRaises(IndexError, msg='Dimension:Header pair(s) ("size:huge", "hard:zero") do not exist.'):
            test_ndtable.get(size="huge", hard="zero")

        # soft:all does not exist, but size:large does
        with self.assertRaises(IndexError, msg='Dimension:Header pair(s) ("soft:all") do not exist.'):
            test_ndtable.get(soft="all", size="large")

    def test_get_singular_values_all_dims_specified(self):
        # test to see if the correct value will be retrieved when specifying valid dimensions and headers

        test_ndtable = construct_basic_ndtable()

        # size=small & hard=none & soft=none should be 0
        self.assertEqual(0, test_ndtable.get(size="small", hard="none", soft="none"))

        # size=small & hard=dense & soft=sparse should be 7
        self.assertEqual(7, test_ndtable.get(hard="dense", size="small", soft="sparse"))

        # size=large & hard=sparse & soft=dense should be 23
        self.assertEqual(23, test_ndtable.get(soft="dense", size="large", hard="sparse"))

        # size=medium & hard=none & soft=sparse should be 10
        self.assertEqual(10, test_ndtable.get(soft="sparse", hard="none", size="medium"))

    def test_get_singluar_values_using_unique_headers(self):
        # test to see if the correct value will be retrieved when specifying valid dimensions and headers and uniquely
        # attributable headers

        test_ndtable = construct_basic_ndtable()

        # small (is uniquely attributable to size) & hard=none & soft=none should be 0
        self.assertEqual(0, test_ndtable.get("small", hard="none", soft="none"))

        # small (is uniquely attributable to size) & hard=dense & soft=sparse should be 7
        self.assertEqual(7, test_ndtable.get("small", hard="dense", soft="sparse"))

        # large (is uniquely attributable to size) & hard=sparse & soft=dense should be 23
        self.assertEqual(23, test_ndtable.get("large", soft="dense", hard="sparse"))

        # medium (is uniquely attributable to size) & hard=none & soft=sparse should be 10
        self.assertEqual(10, test_ndtable.get("medium", soft="sparse", hard="none"))

    def test_get_values_using_dimension_wildcards_all_specified(self):
        test_ndtable = construct_basic_ndtable()

        # size:small & hard:none & all values for soft = [0, 1, 2]
        self.assertTrue(np.all(np.array([0, 1, 2]) == test_ndtable.get(size="small", hard="none")))

        # size:medium & all values for hard & soft:sparse = [10, 13, 16]
        self.assertTrue(np.all(np.array([10, 13, 16]) == test_ndtable.get(size="medium", soft="sparse")))

        # size:large & all values for hard & all values for soft = [18, 19, 20, 21, 22, 23, 24, 25, 26]
        self.assertTrue(np.all(np.array([[18, 19, 20], [21, 22, 23], [24, 25, 26]]) == test_ndtable.get(size="large")))

        # all values for size & hard:dense & soft:sparse = [7, 16, 25]
        self.assertTrue(np.all(np.array([7, 16, 25]) == test_ndtable.get(hard="dense", soft="sparse")))
        


