# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:14:15 2017

@author: gdi
"""

ar = np.array([[1,2,3],[3,4,5],[4,5,6]])
ar = ar - np.max(ar,axis = 1).reshape((3,1))
ar = np.exp(ar)
sum1 = np.sum(ar, axis = 1)
ar = ar/ sum1