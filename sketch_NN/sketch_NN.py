import matplotlib.pyplot as plt
from matplotlib import patches


def sketch_simplified_nn(input_dim, layers_dim):

    """
    Plots ANN using the simplified notation
    input_dim: number of features in the input data
    layers_dim: list with number of neurons in each layer
    including Outer Layer (determined by number
    of class labels or regression targets)

    Example: sketch_simplified_nn(100, [10, 2, 1])
    will sketch a simplified ANN with 100 input features,
    10 neurons in the first Hidden Layer, 2 neurons in the
    second Hidden Layer, and 1 neuron on the Outer Layer
    """

    # Main figure
    plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.axis([0, 50, 0, 50])
    plt.axis("off")
    # Input rectangle
    rectangle = plt.Rectangle((1, 10), 2, 30, color="k")
    plt.gca().add_patch(rectangle)
    # Input vector
    plt.quiver(3, 35, 1, 0, width=0.005, scale=12.5)
    plt.annotate("p", (4.2, 37), weight="bold", fontsize=15)
    # Input vector's dimensions
    if input_dim >= 100:
        plt.annotate(input_dim, (1, 8))
        plt.annotate(str(input_dim)+"x1", (3.4, 32.5))
    else:
        plt.annotate(input_dim, (1.5, 8))
        plt.annotate(str(input_dim)+"x1", (4, 32.5))
    # Input bias
    plt.quiver(5.5, 16, 1, 0, width=0.005, scale=32)
    plt.annotate(1, (4, 15), weight="bold", fontsize=15)
    # Loop to repeat the structure
    a = 0
    for i in range(len(layers_dim)):
        # Bias box
        rectangle = plt.Rectangle((7+a, 13), 3, 6, fill=False, color="k")
        plt.gca().add_patch(rectangle)
        plt.annotate("b$^"+str(i+1)+"$", (8+a, 15), weight="bold", fontsize=15)
        plt.annotate(str(layers_dim[i])+"x1", (7.75+a, 11))
        # Weight box
        rectangle = plt.Rectangle((7+a, 32), 3, 6, fill=False, color="k")
        plt.gca().add_patch(rectangle)
        plt.annotate("W$^"+str(i+1)+"$", (7.5+a, 34), weight="bold", fontsize=15)
        if input_dim >= 100:
            plt.annotate(str(layers_dim[i])+"x"+str(input_dim), (7.1+a, 30))
        else:
            plt.annotate(str(layers_dim[i])+"x"+str(input_dim), (7.75+a, 30))
        input_dim = layers_dim[i]
        # Net input arrows
        line = plt.Line2D((10.1+a, 11+a), (35, 35), lw=2.2, color="k")
        plt.gca().add_line(line)
        line = plt.Line2D((10.1+a, 11+a), (15, 15), lw=2.2, color="k")
        plt.gca().add_line(line)
        plt.quiver(11+a, 35, 1, -1, width=0.005, scale=16)
        plt.quiver(11+a, 15, 1, 1, width=0.005, scale=16)
        # Net input sign
        ellipse = patches.Ellipse((15+a, 25), 4, 8, fc='k', fill=False)
        plt.gca().add_patch(ellipse)
        line = plt.Line2D((15+a, 15+a), (23, 27), lw=2, color="k")
        plt.gca().add_line(line)
        line = plt.Line2D((14+a, 16+a), (25, 25), lw=2, color="k")
        plt.gca().add_line(line)
        # Net input vector and dimensions
        plt.quiver(17+a, 25, 1, 0, width=0.005, scale=12)
        plt.annotate("n$^"+str(i+1)+"$", (18+a, 26), weight="bold", fontsize=15)
        plt.annotate(str(layers_dim[i])+"x1", (17.75+a, 22.5))
        # Transfer function box and dimensions
        rectangle = plt.Rectangle((21.1+a, 10), 3, 30, color="k", fill=False)
        plt.gca().add_patch(rectangle)
        plt.annotate("f$^"+str(i+1)+"$", (22+a, 25), weight="bold", fontsize=15)
        plt.annotate(layers_dim[i], (22.25+a, 8))
        # Output vector and dimensions
        plt.quiver(24.1+a, 35, 1, 0, width=0.005, scale=12.5)
        plt.annotate("a$^"+str(i+1)+"$", (25+a, 37), weight="bold", fontsize=15)
        plt.annotate(str(layers_dim[i])+"x1", (25+a, 32.5))
        # If this is not the Outer Layer
        if i != len(layers_dim)-1:
            # Translates the whole structure horizontally
            a += 21.2
            # If we have drawn two layers
            if i % 2 == 1:
                # Shows the plot
                plt.show()
                # Creates a new figure, and starts anew
                plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
                plt.axis([0, 50, 0, 50])
                plt.axis("off")
                a = 0
                # Instead of creating input rectangle and vectors
                # Shows the outputs from the last layer on the previous figure
                plt.quiver(1, 35, 1, 0, width=0.005, scale=8.25)
                plt.annotate("a$^"+str(i+1)+"$", (3, 37), weight="bold", fontsize=15)
                plt.annotate(str(layers_dim[i])+"x1", (3, 32.5))
                plt.quiver(4.5, 16, 1, 0, width=0.005, scale=20.5)
                plt.annotate(1, (3, 15), weight="bold", fontsize=15)
        # If it's the Outer Layer
        else:
            # Just shows the plot
            plt.show()
