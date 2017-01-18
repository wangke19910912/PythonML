from numpy import *
#coding=utf-8
U,Sigma,VT = linalg.svd([[1,1],[7,7]])

print U
print Sigma
print VT