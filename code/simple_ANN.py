# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:19:17 2014

@author: olena
"""

import numpy as np
import neurolab as nl
import math
import random

count = 1000
v1 = np.array([random.uniform(-1,1) for x in range(count)])
v2 = np.array([random.uniform(-1,1) for x in range(count)])
v3 = [3]*count

Input = np.append(v1,v2)
Input = Input.reshape(count,2)
target = 2*v1 + v2*0.5 #np.log(v3+v2)
target = target.reshape(count,1)
#print Input
#print target


net = nl.net.newff([[-1,1],[-1,1]],[3,1])
err = net.train(Input,target,show=15)
res = net.sim([[0.2,0.1]])
print "asking to simulate net.sim([[0.2,0.1]]) with 3 input nodes"
print res
print "answer should be:",2*0.2+ 0.1*.5 #math.log(3+.1)

net = nl.net.newff([[-1,1],[-1,1]],[5,1])
err = net.train(Input,target,show=15)
res2 = net.sim([[-0.3,0.5]])
print "asking to simulate net.sim([[.3,.5]]) with 5 input nodes"
print res2
print "results should be:",-0.3*2 + 0.5*0.5 #math.log(3+0.5)


# >>> import numpy as np
#         >>> import neurolab as nl
#         >>> # Create train samples
#         >>> input = np.random.uniform(-0.5, 0.5, (10, 2))
#         >>> target = (input[:, 0] + input[:, 1]).reshape(10, 1)
#         >>> # Create network with 2 inputs, 5 neurons in input layer and 1 in output layer
#         >>> net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
#         >>> # Train process
#         >>> err = net.train(input, target, show=15)
#         Epoch: 15; Error: 0.150308402918;
#         Epoch: 30; Error: 0.072265865089;
#         Epoch: 45; Error: 0.016931355131;
#         The goal of learning is reached
#         >>> # Test
#         >>> net.sim([[0.2, 0.1]]) # 0.2 + 0.1
#         array([[ 0.28757596]])
