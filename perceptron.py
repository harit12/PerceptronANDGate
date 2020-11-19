import numpy as np
from matplotlib import pyplot as plt
inputs = [[0,0],[0,1],[1,1],[1,0]]
outputs = [1,0,1,0]
inputs = np.array(inputs)
outputs = np.array(outputs)
lr = .01
epochs = 10000
def create_nn(n,m):
    """
    Initialize weights
    """
    nueral_net = np.random.random((m,n))
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
    zVal = weights.dot(layer)
    return zVal
def forward(network, inputs):
    """
    Forward propogation
    Go though each layer of network and get the weights multiplied by the previous layer. Use activation function for z value for each nueron in each layer
    """
    z = 0
    for layer in network:
        z = zVal(inputs, layer)
    activated = activation(z)
    return activated
def cost(hypo, output):
    res = (hypo-output)**2
    return res
def backprop(network, inputed, output):
    activated = forward(network, inputed)
    costed = cost(activated, output)
    #print(costed, 'csot')
    derivCost = activated-output
    derivCost*=2.0
    sigmoidDeriv = activated*(1-activated)
    zDeriv = inputed
    finalDeriv = derivCost*sigmoidDeriv*zDeriv
    #print(zDeriv, 'zDeriv')
    #print(sigmoidDeriv, 'sDeriv')
    #print(derivCost, 'derivCost')
    return finalDeriv, costed
def train(network, inputs, outputs):
    costList1 = []
    costList2 = []
    costList3 = []
    costList4 = []
    for n in range(epochs):
        count = 0
        finalDeriv = 0
        for inputed in inputs:
            finalDerived, cost = backprop(network, inputed, outputs[count])
            finalDeriv+=finalDerived
            count+=1
            if count == 0:
                costList1.append(cost)
            elif count ==1:
                costList2.append(cost)
            elif count ==2:
                costList3.append(cost)
            elif count==3:
                costList4.append(cost)
        finalDeriv = finalDeriv/4
        #print(finalDeriv, 'finalDeriv')
        network-=finalDeriv*lr
        #print(network, 'net')
    plt.plot(np.linspace(0,100, len(costList1)), costList1)
    plt.plot(np.linspace(0,100, len(costList2)), costList2)
    plt.plot(np.linspace(0,100, len(costList3)), costList3)
    plt.plot(np.linspace(0,100, len(costList4)), costList4)
    return network
def main():
    weights = create_nn(2,1)
    print(weights, 'og')
    trained = train(weights, inputs, outputs)
    print(trained)
    print(forward(trained, [1,1]))
    plt.show()
main()    
    
