import numpy as np
from matplotlib import pyplot as plt
inputs =np.array([[1.,0.,1.],[1.,1.,1.], [1., 1., 0.], [1., 0., 0.]])
outputs = [0.,1.,0., 1.]
#inputs = np.array(inputs)
print(type(inputs))
outputs = np.array(outputs)
lr = .01
epochs = 25000
def create_nn(n,m):
    """
    Initialize weights
    N-number of nodes
    m-number oflayers
    """
    nueral_net = np.random.random((m,n+1))
    return nueral_net
def activation(zVal):
    """
    Activation function(sigmoid)
    """
    res = (1)/(1+np.exp(-zVal))
    return res

def zVal(layer, weights):
    """
    Weights multiplied by previous layer or in this case, the inputs
    """
    bias = weights[0] * layer[0]
    zVal = weights[1:].dot(layer[1:])
    zVal+=bias
    return zVal

def forward(network, inputs):
    """
    Forward propogation
    Go though each layer of network and get the weights multiplied by the previous layer. Use activation function for z value for each nueron in each layer
    """
    activated = inputs
    for layer in network:
        z = zVal(activated, layer)
        activated = activation(z)
    print(activated)
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
    #biasDeriv = dCostdBias(activated, output)
    #res = np.insert(costdWeights,0, biasDeriv, axis = 0)
    #print(zDeriv, 'zDeriv')
    #print(sigmoidDeriv, 'sDeriv')
    #print(derivCost, 'derivCost')
    return costed, costdWeights
def update(gradient, weights):
    weights = weights  + lr*(-gradient)
    return weights
def train(network, inputs, outputs):
    costList1 = []
    costList2 = []
    costList3 = []
    costList4 = []
    for n in range(epochs):
        for count,(x,y) in enumerate(zip(inputs,outputs)):
            x = np.array(x)
            cost, finalDerived = backprop(network, x,y)
            network=update(finalDerived, network)
            if count == 0:
                costList1.append(cost)
            elif count ==1:
                costList2.append(cost)
            elif count ==2:
                costList3.append(cost)
            else:
                costList4.append(cost)
        #print(finalDeriv, 'finalDeriv')
        #print(network, 'net')
    plt.plot(np.linspace(0,epochs, len(costList1)), costList1)
    plt.plot(np.linspace(0,epochs, len(costList2)), costList2)
    plt.plot(np.linspace(0,epochs, len(costList3)), costList3)
    plt.plot(np.linspace(0,epochs, len(costList4)), costList4)
    return network
def main():
    weights = create_nn(2,1)
    print(weights)
    trained = train(weights, inputs, outputs)
    print(trained)
    print(forward(trained, [1.,0,1.]),'0,1')
    print(forward(trained, [1.,1.,1.]),'1,1')
    print(forward(trained, [1.,1.,0.]),'1,0')
    print(forward(trained, [1.,0.,0.]),'0,0')
    plt.show()

if __name__=="__main__":
    main()    
    
