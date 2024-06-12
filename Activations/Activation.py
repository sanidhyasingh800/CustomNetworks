
# a superclass implementation for all Activation functions used by the layers 
# Currently, we will support 
#   Sigmoid
#   ReLU
#   Linear (No Activation)
#   SoftMax
class Activation:
    
    def __init__(self):
        pass

    # both of these are to be implemented by any activation function

    def activate(self, input_):
        pass

    def derivative(self, input_):
        pass