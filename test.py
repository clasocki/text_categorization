import numpy

class A(object):
    def __init__(self):
        self.a = numpy.asarray(xrange(10))

def f(arr):
   arr += 10

a = A()
print a.a
map(lambda d: f(d.a), [a])
print a.a
