import numpy as np

inputs = [[0,0],[0,1],[1,1],[1,0]]
outputs = [1,0,1,0]
inputs = np.array(inputs)
outputs = np.array(outputs)
lr = .01
epochs = 10000
def create_nn(n,m):
    nueral_net = np.random.random((m,n))
    return nueral_net

def activation(zVal):
    res = (1)/(1+np.exp(-zVal))
    return res
def zVal(layer, weights):
    zVal = weights.dot(layer)
    return zVal
def forward(network, inputs):
    z = np.empty((2,1))
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
    print(costed, 'csot')
    derivCost = activated-output
    derivCost*=2.0
    sigmoidDeriv = activated*(1-activated)
    zDeriv = inputed
    finalDeriv = derivCost*sigmoidDeriv*zDeriv
    #print(zDeriv, 'zDeriv')
    #print(sigmoidDeriv, 'sDeriv')
    print(derivCost, 'derivCost')
    return finalDeriv
def train(network, inputs, outputs):
    for n in range(epochs):
        count = 0
        finalDeriv = 0
        for inputed in inputs:
            finalDeriv += backprop(network, inputed, outputs[count])
            count+=1
        finalDeriv = finalDeriv/4
        #print(finalDeriv, 'finalDeriv')
        network-=finalDeriv*lr
        #print(network, 'net')
    return network
def main():
    weights = create_nn(2,1)
    print(weights, 'og')
    trained = train(weights, inputs, outputs)
    print(trained)
    print(forward(trained, [1,1]))
main()    
    
