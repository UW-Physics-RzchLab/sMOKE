# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:13:30 2016

@author: rzchlab
"""
from sMOKE import matrix_rotate_indices, get_axarr_coords, verify_directions
import numpy as np
from nose.tools import assert_true, assert_equal, raises


class TestMatrixRotation:
    @classmethod
    def setup(cls):
        cls.A2 = np.arange(4).reshape(2, 2)
        cls.A3 = np.arange(9).reshape(3, 3)

    def rotate_2x2(self, original, deg_ccw):
        res = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                ir, jr = matrix_rotate_indices(2, i, j, deg_ccw)
                res[ir][jr] = self.A2[i, j]
        return np.array(res)

    def rotate_3x3(self, original, deg_ccw):
        res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                ir, jr = matrix_rotate_indices(3, i, j, deg_ccw)
                res[ir][jr] = self.A3[i, j]
        return np.array(res)

    def test_individually_90(self):
        ij = ((0, 0), (0, 1), (1, 0), (1, 1))
        ij_rot90 = ((1, 0), (0, 0), (1, 1), (0, 1))
        for (i, j), (ii, jj) in zip(ij, ij_rot90):
            ir, jr = matrix_rotate_indices(2, i, j, 90)
            assert_equal((ii, jj), (ir, jr))

    def test_individually_m90(self):
        ij = ((0, 0), (0, 1), (1, 0), (1, 1))
        ij_rotm90 = (0, 1), (1, 1), (0, 0), (1, 0)
        for (i, j), (ii, jj) in zip(ij, ij_rotm90):
            ir, jr = matrix_rotate_indices(2, i, j, -90)
            assert_equal((ii, jj), (ir, jr))

    def test_individually_180(self):
        ij = ((0, 0), (0, 1), (1, 0), (1, 1))
        ij_rot180 = (1, 1), (1, 0), (0, 1), (0, 0)
        for (i, j), (ii, jj) in zip(ij, ij_rot180):
            ir, jr = matrix_rotate_indices(2, i, j, 180)
            assert_equal((ii, jj), (ir, jr))

    def test_rot_by_90(self):
        A2_rot90 = np.array(((1, 3), (0, 2)))
        res = self.rotate_2x2(self.A2, 90)
        assert_true((res == A2_rot90).all())

    def test_rot_by_m90(self):
        A2_rotm90 = np.array(((2, 0), (3, 1)))
        res = self.rotate_2x2(self.A2, -90)
        assert_true((res == A2_rotm90).all())

    def test_rot_by_180(self):
        A2_rot180 = np.array(((3, 2), (1, 0)))
        res = self.rotate_2x2(self.A2, 180)
        assert_true((res == A2_rot180).all())

    def test_rot3x3_by_90(self):
        A3_rot90 = np.rot90(self.A3)
        res = self.rotate_3x3(self.A3, 90)
        assert_true((res == A3_rot90).all())

    def test_rot3x3_by_m90(self):
        A3_rotm90 = np.rot90(np.rot90(np.rot90(self.A3)))
        res = self.rotate_3x3(self.A3, -90)
        assert_true((res == A3_rotm90).all())

    def test_rot3x3_by_180(self):
        A3_rot180 = np.rot90(np.rot90(self.A3))
        res = self.rotate_3x3(self.A3, 180)
        assert_true((res == A3_rot180).all())
        

class TestCardinalDirectionsSystem:
    @classmethod
    def setup(cls):
        cls.A2 = np.arange(4).reshape(2, 2)
        cls.A3 = np.arange(9).reshape(3, 3)
    
    def rot_by_cardinals(self, origin, xplus, yplus, n):
        res = [[0 for x in range(n)] for y in range(n)]
        for i in range(n):
            for j in range(n):
                ir, jr = get_axarr_coords(origin, xplus, yplus, n, i, j)
                res[ir][jr] = self.A3[i][j]
        return np.array(res)
    
    def check_one_case(self, origin, xplus, yplus, n, rot_by, trans):
        A = self.A3
        correct = A.T if trans else A
        k = {0: 0, 90: 1, 180: 2, -90: 3}[rot_by]
        correct = np.rot90(correct, k)
        res = self.rot_by_cardinals(origin, xplus, yplus, n)
        assert_true((res == correct).all())

    def test_case_generator(self):
        origin = ('NW', 'NE', 'SE', 'SW')
        xplus = ('S', 'S', 'N', 'N')
        yplus = ('E', 'W', 'W', 'E')
        rot_by = (0, -90, 180, 90)
        trans = (False, True, False, True)
        for e in zip(origin, xplus, yplus, rot_by, trans):
            o, xp, yp, r, t = e
            yield self.check_one_case, o, xp, yp, 3, r, t
    
    @raises(ValueError)            
    def test_bad_params(self):
        self.check_one_case('NS', 'E', 'W', 3, 90, True)


class TestVerifyDirections:

    def test_good_params(self):
        verify_directions('NW', 'E', 'S')
        verify_directions('NW', 'S', 'E')
        verify_directions('SE', 'N', 'W')
        verify_directions('SW', 'E', 'N')

    @raises(ValueError)
    def check_bad_params(self, o, x, y):
        verify_directions(o, x, y)

    def test_generator_bad(self):
        ps = (('NW', 'N', 'S'), ('NS', 'S', 'E'), ('SE', 'E', 'W'))
        for o, x, y in ps:
            yield self.check_bad_params, o, x, y
