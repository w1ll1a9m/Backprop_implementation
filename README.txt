

Simple neural network trained to reproduce the input vector with a dense fully conected layer of 3 activation units for a 8 input layer.

first chose from 0 to 7 a sample of the possible vectors to train the netowrk
             [0,0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1,0],
             [0,0,0,0,0,1,0,0],
             [0,0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0,0],
             [0,0,1,0,0,0,0,0],
             [0,1,0,0,0,0,0,0],
             [1,0,0,0,0,0,0,0].
             
the network is trained using backpropagation to reproduce the chosen vector of the input from all of the possible inputs.
to run the program either open it in an IDE and run it or use the command line
$ python3 backprop.py
make sure you're using python 3
