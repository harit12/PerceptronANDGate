#API
import numpy as np

#Debugging/User
from matplotlib import pyplot as plt
ax = plt.axes(projection='3d')

inputs =np.array([[0.,1.],[1.,1.], [ 1., 0.], [ 0., 0.]])

#Helper Functions
def add_bias(data):
    """
    args: (type: numpy array, expl: the feature vector of x)
    Add bias to front of input vector
    return: (type: numpy array, expl: numpy array containing bias in front of each row of array)
    """
    data = np.insert(data, 0, 1, axis=1)
    return data

def zVal(layer, weights):
    """
    Weights multiplied by previous layer or in this case, the inputs
    """
    bias = weights[0] * layer[0]
    zVal = weights[1:].dot(layer[1:])
    zVal+=bias
    return zVal

def activation(zVal):
    """
    args: (type: float, expl: the dot product of weights and previous layer)
    Use activation function on dot product of weights and previous layer
    return: (type: float, expl: the value of the node using the activation function)
    """
    res = (1)/(1+np.exp(-zVal))
    return res

def forward(network, inputs):
    """
    args: network(type: numpy array, expl: the type of 
    Go though each layer of network and get the weights multiplied by the previous layer. Use activation function for z value for each nueron in each layer
    return: activated(type: float)
    """
    activated = inputs
    for layer in network:
        z = zVal(activated, layer)
        activated = activation(z)
    return activated, z

def cost(hypo, output):
    res = (hypo-output)**2
    return res

def dCostdA(activated, output):
    res = 2*(activated-output)
    return res

def dAdZ(zVal):
    res = activation(zVal)*(1-activation(zVal))
    return res

def dCostdW(activated,z, inputs, output):
    derivCost = dCostdA(activated, output)
    derivAct = dAdZ(z)
    zDeriv = inputs
    res = derivCost*derivAct*zDeriv
    return res

def dCostdBias(activated, output):
    derivCost = dCostdA(activated, output)
    derivAct = dAdZ(activated)
    zDeriv = 1
    res = zDeriv*derivAct*derivCost
    return res    

def backprop(network, inputed, output):
    activated, zVal = forward(network, inputed)
    costed = cost(activated, output)
    costdWeights = dCostdW(activated, zVal, inputed, output)
    return costed, costdWeights

def update(gradient, weights, lr):
    weights = weights  + lr*(-gradient)
    return weights
#API
def create_nn(n,m):
    """
    args: n(type: int, expl: the number of nodes in the biggest layer), m(type: int, expl: number of layers not including output)
    Initialize weights of nueral net using randomization
    return: (type: numpy array, expl: numpy array of random weights) 
    """
    nueral_net = np.random.random((m,n+1))
    return nueral_net

def train(network, inputs, outputs, epochs = 20000, lr=.001, graph=True):
    inputs = add_bias(inputs)
    weight1List = []
    weight2List = []
    costList = []
    for n in range(epochs):
        for count,(x,y) in enumerate(zip(inputs,outputs)):
            weight1List.append(network[0][1])
            weight2List.append(network[0][2])
            x = np.array(x)
            costed, finalDerived = backprop(network, x,y)
            network=update(finalDerived, network, lr)
            costList.append(costed)
    if graph==True:
        ax.plot3D(weight1List, weight2List, costList, 'gray')
    return network

def predict(weights, inputs):
    activated, z = forward(weights, inputs)
    #print(activated)
    if activated>.5:
        activated = 1
    else:
        activated = 0
    return activated

def main():
    outputsAND = np.array([0.,1.,0., 0.])
    outputsOR = np.array([1,1,1,0])
    weights = create_nn(2,1)
    trainedAND = train(weights, inputs, outputsAND, graph=False)
    trainedOR = train(weights, inputs, outputsOR, graph=False)
    print(predict(trainedAND, [1.,0,1.]),'0,1')
    print(predict(trainedAND, [1.,1.,1.]),'1,1')
    print(predict(trainedAND, [1.,1.,0.]),'1,0')
    print(predict(trainedAND, [1.,0.,0.]),'0,0')
    print(predict(trainedOR, [1.,0,1.]),'0,1')
    print(predict(trainedOR, [1.,1.,1.]),'1,1')
    print(predict(trainedOR, [1.,1.,0.]),'1,0')
    print(predict(trainedOR, [1.,0.,0.]),'0,0')
    plt.show()

if __name__=="__main__":
    main()    
    
