##  Objective :- Build a Perceptron model 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.array([
     [0, 0],
     [0, 1],
     [1, 0],
     [1, 1]
])

y = np.array([0, 0, 0, 1])  # AND logic gate output

# Initialize weights to zero
weights = np.zeros(X.shape[1])  

# Initialize bias to zero
bias = 0.0  

# Hyperparameters of learning rate and number of epochs

learning_rate = 0.1  ## larger learning rate means larger steps in the direction of the gradient.
                     ## smaller learning rate means smaller steps in the direction of the gradient.
                     
epochs = 10          ## number of iterations/times model goes over the entire dataset


# Training the Perceptron model
for epoch in range(epochs):
    print("\n"+ "-" * 50)
    print(f"Epoch {epoch + 1}")
    print("\n"+ "-" * 50)
    
    errors = 0  ## to keep a track of how many predictions were incorrect in each epoch
    
    # Go through each training example
    for i in range(len(X)):
        inputs = X[i]
        actual_output = y[i]
        
        # Calculate the weighted sum (linear combination)
        weighted_sum = np.dot(inputs, weights) + bias
        
        # Apply the step activation function (Make a prediction based on the weighted sum)
        prediction = 1 if weighted_sum >= 0 else 0
        
        # Calculate the loss (error) for the current prediction
        loss = actual_output - prediction
        
        if loss != 0:
            errors = errors + 1  ## increment the error count if prediction is incorrect
            
        # Update weights and bias based on the loss
        weights = weights + learning_rate * loss * inputs
        bias    = bias + learning_rate * loss
        
        # Print the details of the current training example
        print(f"\nTraining Example {i + 1}:")
        print(f"\n  weighted_sum: {weighted_sum}")
        print(f"\n  Inputs: {inputs}")
        print(f"\n  Actual Output: {actual_output}")
        print(f"\n  Prediction: {prediction}")
        print(f"\n  Loss: {loss}")
        print(f"\n  Updated Weights: {weights}")
        print(f"\n  Updated Bias: {bias:.2f}")
        
        
    # Print errors for this epoch
    print(f"\nErrors in epoch: {errors}")       
    
    
    # Stop training if there are no errors
    if errors == 0:
        print("\nTraining is successful")
        break
    
# Test the trained perceptron
print("\n")
print("-" * 50)
print("FINAL MODEL")
print("-" * 50)

print(f"Weights: {weights}")
print(f"Bias: {bias:.2f}")

for i in range(len(X)):
    
    inputs = X[i]
    actual_outputs = y[i]

    weighted_sum = np.dot(inputs, weights) + bias

    if weighted_sum >= 0:
        prediction = 1
    else:
        prediction = 0

    print(
        f"Input: {inputs}",
        f"Expected: {actual_outputs}",
        f"Predicted: {prediction}"
    )
        