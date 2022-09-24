import numpy as np
from numpy import array
from numpy import matrix
import matplotlib.pyplot as plt
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
    x = np.array([0, 1, 2, 3,4,5])
    y = np.array([-1, 0.2, 0.9, 2.1,2.6,3.1])
    A = np.vstack([x, np.ones(len(x))]).T
    [k,b]=np.linalg.lstsq(A,y,rcond=None)[0]
    C=np.linalg.lstsq(A, y, rcond=None)[0]
    new_y=x*C[0]+C[1]
    fitting=[]
    for i in range(len(x)):
        fitting.append(x.item(i)*k+b)
    plt.scatter(x,new_y,color = 'hotpink',label="scatter")
    plt.plot(x,fitting,label='fitting line')
    plt.legend()
    plt.show()

def multiple_variable_linear_regression():
    import random
    import numpy as np
    import pandas as pd
    random.seed(88)
    data_x = pd.DataFrame(
        {"a": random.sample(range(100), 20), "b": random.sample(range(100), 20), "c": random.sample(range(100), 20)})
    data_y = pd.Series(random.sample(range(200), 20))
    data_y.name = "y"
    res = np.linalg.lstsq(data_x, data_y, rcond=None)
    fitting = np.matmul(data_x, res[0].T)
    lst = [i for i in range(20)]
    plt.scatter(lst, data_y.values, color='green', label='scatter')
    plt.plot(lst, fitting.values, label='fitting model')
    plt.legend()
    plt.show()
def fit_polynomial():
    import math
    x=np.random.rand(2,5)
    y=x+pow(x,2)+pow(x,3)
    print(x)
    print(y)
    A=np.concatenate([np.ones((x.shape[0],1)),x],axis=1)
    Weight=np.linalg.lstsq(A,y,rcond=None)[0]
    # new_C=np.matmul(A,Weight)
    # print(Weight)
    # print(y[0,:])
    # print(new_C[0,:])
    # plt.scatter(x[0,:],y[0,:],color = 'hotpink',label="scatter")
    # plt.plot(x[0,:],new_C[0,:],label='fitting line')
    # lst=[]
    # lst_x=[]
    # for i in range(len(new_C)):
    #     lst.append(new_C.item(i))
    #     lst_x.append(x.item(i))
    # plt.plot(lst_x,lst,label='fitting')
    # plt.scatter(x, y, color='hotpink', label="scatter")
    plt.legend()
    plt.show()

if __name__=="__main__":
    #lst_square()
    # multiple_variable_linear_regression()
    A=np.array([[0],[2],[3],[4],[5]])
    print(A.all())

