The program will run 100000 epochs using all 8 inputs as training instances.
After training a final forward propagation run is ran to test the network.
The results get printed.

You can change the 'log' variable to 1 if you would like to see the changes during the learning phase.
It prints every 5000th activations in layer 3 and their corresponding inputs.

Initially random weights are used, however by commenting in the terms behind them
they can be changed to be randomly distributed around 0.01.

Comment in the last two lines to look at the final weights

to run the program either open it in an IDE and run it or use the command line
$ python3 neuralnetwork.py
make sure you're using python 3