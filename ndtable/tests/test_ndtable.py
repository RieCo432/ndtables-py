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

        test_ndtable = ndtable.ndtable(shape=ndtable_shape, headers=ndtable_headers, dtype=int)

        # the data portion of the nd-table should be an array of all zeros
        self.assertTrue(np.all(np.equal(test_ndtable.data, np.zeros(ndtable_shape, dtype=int))))

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

        self.assertEqual(test_ndtable.indexing, expected_indexing_dict)

        # these are the expected unique headers: small, medium, large
        # all of them should belong to the first dimension (size) and should have the following indexes: 0, 1, 2

        expected_unique_headers = {"small": {"dim": 0, "index": 0},
                                   "medium": {"dim": 0, "index": 1},
                                   "large": {"dim": 0, "index": 2}}
        self.assertEqual(test_ndtable._unique_headers, expected_unique_headers)

        # these are the expected non-unique headers: none, sparse, dense
        expected_non_unique_headers = ["none", "sparse", "dense"]
        self.assertEqual(test_ndtable._non_unique_headers, expected_non_unique_headers)


