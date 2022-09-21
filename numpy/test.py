import numpy as np
from numpy import array
from numpy import matrix

def test_operation():
    print(np.arange(3))
    print(np.arange(3).reshape(1,3))
    a=np.array([[1,2,3,4],[5,6,7,8]])
    print(a)
    # MATLAB写法
    m1=np.matrix('1 2 3 5;5 6 7 8')
    #numpy 写法
    m2=np.matrix([[1,2,3,5],[5,6,7,8]])
    print(m1)
    print(m2)


    #get element
    print(a.item(1))
    print(a.item((0,1)))
    print(m2.item((0,3)))
    print(m1.item(1))

    #slice row
    print(a[1:])

    #slice column
    print(m1[:,1])

def lst_square():
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])
    A = np.vstack([x, np.ones(len(x))]).T
if __name__=="__main__":
    a=np.array([[1,2,3,5,6,7,8,9,10],[1,2,3,5,6,7,8,9,10],[1,2,3,5,6,7,8,9,10]])
    b=np.matrix('1 2 3 5 6 7 8 9 11')
    c=np.concatenate([a,b],axis=0)
    print(c)
