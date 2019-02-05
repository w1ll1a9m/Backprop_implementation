import numpy as np

np.random.seed(1)

inn=input("select the sample from 0 to 7 \n")
innt=int(inn)
Theta1 = np.random.randn(8,3) 
Theta2 = np.random.randn(3,8) 

bias1=np.random.randn(1,3)

bias2=np.random.randn(1,8)

bias=np.random.rand(2)
lrate=0.5

X = np.array([[0,0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1,0],
             [0,0,0,0,0,1,0,0],
             [0,0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0,0],
             [0,0,1,0,0,0,0,0],
             [0,1,0,0,0,0,0,0],
             [1,0,0,0,0,0,0,0]])



class neural_network:
    
    def __init__(self):
        #initial
        self.In = 8
        self.Out = 8
        self.hidden = 3
        self.DeltaW = np.zeros((8,3))
        self.Deltab = np.zeros((2))
        
    

    def forward_propagation(self,X,Theta1,Theta2,bias1,bias2, i):
        #forward propagation
    
        self.z2 = np.dot(X, Theta1) + bias1*bias[0] #to get Z1s
        self.a2 = self.sigmoid(self.z2) #Activations in hidden layer
        self.z3 = np.dot(self.a2, Theta2) + bias2*bias[1] #To get Z2s
        self.a3= self.sigmoid(self.z3) #output in final layer

        if i % 5000 ==0 :
            print ("epoch " + str(i) )
            print("current input: \n" + str(X))
            print ("predicted outputs: \n" + str(self.a3))
        

        return  self.a3, self.a2
    
    def sigmoid(self, x):
        #sigmoid function
        return 1/(1+np.exp(-x))

    def sigmoid_der (self, x):
        #derivative of the sigmoid
        return x * (1 - x)

    def backward_propagation(self, x, a2, a3, Theta2,Deltat2,Deltat1,Deltab2,Deltab1):
        #backward propagation
        #error terms \ deltas
        self.x1 = np.zeros((8,1))
        self.x1=self.x1.T+x

        self.error= self.x1-a3 #error on the final layer
        
        self.delta3 = - (self.error * self.sigmoid_der(a3)) 
        self.delta2 = np.dot( self.delta3 , Theta2.T) * self.sigmoid_der(a2)
        
        self.a21=np.zeros((3,1))
        self.a21=self.a21.T+a2
        
       
        #Capital deltas \ partial derivatives

        self.Delta2 = np.dot( self.delta3.T , self.a21)
        self.Delta2b = self.delta3
        self.Delta1 = np.dot( self.delta2.T, self.x1)
        self.Delta1b = self.delta2

        #gradient descent
        

        Deltat2 = Deltat2 + self.Delta2.T
        Deltab2 = Deltab2 + self.delta3
        Deltat1 = Deltat1 + self.Delta1.T
        Deltab1 = Deltab1 + self.delta2

        return Deltat2, Deltat1, Deltab1, Deltab2

       


    
    
nn=neural_network()

epochs=100000

for i in range(epochs): 
    Deltat2 = np.zeros((3,8))
    Deltat1 = np.zeros((8,3))
    Deltab2 = 0
    Deltab1 = 0
    for sample in range(len(X)):
        a3, a2=nn.forward_propagation(X[sample],Theta1,Theta2,bias1,bias2, i)
        Deltat2, Deltat1, Deltab1, Deltab2= nn.backward_propagation( X[sample], a2, a3, Theta2,Deltat2,Deltat1,Deltab2,Deltab1)
    Theta2=Theta2 - lrate*((1/len(X))*Deltat2 + 0.00001*Theta2)
    Theta1=Theta1 - lrate*((1/len(X))*Deltat1 + 0.00001*Theta1)
    bias1=bias1 - lrate*((1/len(X)) * Deltab1)
    bias2=bias2 - lrate*((1/len(X)) * Deltab2)   
 #update parameters
    
    
    
o, o2=nn.forward_propagation(X[innt],Theta1,Theta2,bias1,bias2,0)

print ("Input: \n" + str(X[innt]))
print ("Output: \n" + str(o))
print ("\n")
print("Weights 1: \n" + str(Theta1))
print("Weights 2: \n" + str(Theta2))


