import matplotlib.pyplot as plt
neurons=range(10,130,10);
print neurons
train_accuracy=[86.2,89.92,91.7,92.32,92.71,93.37,93.76,93.97,94.29,94.03,94.49,94.9]
test_accuracy=[86.7,89.91,92.0,92.61,93.0,93.7,93.9,94.25,94.57,94.33,94.71,95.15]
plt.xlabel('Neurons in hidden layer')
plt.ylabel('Accuracy')
plt.plot(neurons, train_accuracy,neurons,test_accuracy)
plt.show()
