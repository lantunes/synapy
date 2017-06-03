Synapy
======

Synapy is a Python implementation of a synaptic matrix. This project represents an ongoing search for new neurobiologically inspired learning algorithms and computational techniques. It represents an exploration of alternatives to the current neural net/deep learning/backprop paradigm, which, despite being spectacularly successful in many aspects, fails to capture the same spectrum of capabilities of neurobiological intelligence.

This work is inspired by, and based on, Dr. Arnold Trehub's work. For more information, see his book, **The Cognitive Brain**, and visit his web page: http://people.umass.edu/trehub/

Particularly, this project is based on his concept of the synaptic matrix. For more on the synaptic matrix, see: http://people.umass.edu/trehub/thecognitivebrain/chapter3.pdf

The synaptic matrix implemented in this project supports supervised learning scenarios. During the training phase, examples consisting of a single vector are given to the synaptic matrix, along with corresponding labels. After training, the synaptic matrix can be used to classify new inputs. The result of evaluating an input is an array of numbers, the largest of which corresponds to the predicted class. These resulting numbers can be interpreted as the relative spiking rates of the classification neurons.

Usage
-----
A synaptic matrix requires a vector input of a specified length. Ideally, the vector consists of only ones and zeroes, though arbitrary floating point values are also acceptable. To use a synaptic matrix, it must first be initialized:
  ```python
  synaptic_matrix = SupervisedSynapticMatrix(9, b=1, c=2, k=10)
  ```
  In the snippet above, a synaptic matrix is initialized that accepts input vectors of length 9. Other hyperparameters are also specified.

  Next, it must be trained in a supervised fashion:
  ```python
  example1 = [0, 1, 0,
              0, 1, 0,
              0, 1, 0]

  example2 = [1, 1, 1,
              1, 0, 1,
              1, 1, 1]

  synaptic_matrix.train(example1, label="line")
  synaptic_matrix.train(example2, label="box")
  ```
  Labels can take on any value type, and are not limited to strings.

  The trained synaptic matrix can then be used to evaluate new inputs:
  ```python
  to_evaluate = [1, 1, 1,
                 1, 0, 0,
                 0, 1, 1]

  self.assertEquals("box", synaptic_matrix.evaluate(to_evaluate))
  ```

  The relative spike frequencies generated for any given input can also be obtained:
  ```python
  to_evaluate = [1, 1, 1,
                 1, 0, 0,
                 0, 1, 1]

  self.assertEquals({"line": 16, "box": 24}, synaptic_matrix.relative_spike_frequencies(to_evaluate))
  ```
  These relative spike frequencies demonstrate the degree to which one class is favored over another.

This synaptic matrix implementation was evaluated against the [MNIST data set](http://yann.lecun.com/exdb/mnist/). It achieves 93% accuracy. This is far from the state-of-the-art values of >99% accuracy, but it represents (as far as I can tell) a novel neurobiologically-inspired approach towards machine learning. Additionally, there is no pre-processing of the images in the data set, nor is the data set augmented with any distorted versions of the original images. State-of-the-art techniques often involve pre-processing of the images, and the addition of more images to the data set obtained through deformation of the original images.

The synaptic matrix distinguishes itself from neural nets (as they are currently implemented) principally by addressing two common difficulties encountered in learning problems: the problem of imbalanced data, and the problem of insufficient data. Real-world data is, more often than not, imbalanced. That is, a data set consisting of examples of various classes will often contain more examples of certain classes, and fewer of others. The synaptic matrix should, in theory, be able to classify instances of one class it has seen few examples of, just as well as it can classify instances of another class it has seen many more examples of. The synaptic matrix should also, in theory, be able to learn from far fewer examples overall.

How does it Work?
-----------------

A synaptic matrix is simply an *m x n* matrix, **_W_**. During training and evaluation, it expects an input column vector, **_x_**, with *m* rows. The number of rows in the synaptic matrix is equal to the number of rows in the input column vector. The input column vector, **_x_**, consists of only zeroes and ones.

A synaptic matrix is first initialized by setting all the values to one:

**_W_** = **_J<sub>m,1</sub>_**

Note that there is only a single column in the synaptic matrix at this point.

### Learning

To have the synaptic matrix learn a new example, we take the i<sup>th</sup> example, **_x_**<sub>i</sub>, and perform the following steps:

1. Calculate the eligibility vector, **_E_**:

   **_E_** = **_x_**<sub>i</sub> <sup>o</sup> **_W_**<sub>.,i</sub>
   <sub><sup>(where <sup>o</sup> represents the Hadamard product)</sup></sub>

2. Calculate N, the number of eligible synapses:

   N = sum {**_E_**}

3. Update the i<sup>th</sup> column of the synaptic matrix:

   **_W_**<sub>.,i</sub> = b + **_E_**(c + kN<sup>-1</sup>)

   The variables _b_, _c_, and _k_ are hyperparameter constants and whole numbers, where _b < c << k_.

4. Expand the synaptic matrix, by adding a new column, in preparation for any new examples:

   **_W_** = [**_W_** **_J<sub>m,1</sub>_**]

Each column in the synaptic matrix is, by convention, called a _class cell_. The column represents the cell's dendritic synaptic weights.

Note that once an example is learned, the index of the class cell must be associated with the label representing the class of the example. In practice, this means keeping track of an associative array of the labels of the classes that have been learned to the indices of the class cells that represent them.

### Evaluating

During evaluation against an example, we simply want to find the class cell with the highest _activation_, or _activity_. The highest activity in the synaptic matrix is given by:

a<sub>max</sub> = max {**_x_**<sup>T</sup> **_W_**}

Here, a<sub>max</sub>, the maximum activity in the synaptic matrix, is the maximum of the vector-matrix product of the transpose of **_x_** with **_W_**. The class cell with the highest activity is simply:

c<sub>a<sub>max</sub></sub> = argmax {**_x_**<sup>T</sup> **_W_**}

Where c<sub>a<sub>max</sub></sub> represents the index of the class cell with the highest activity in the synaptic matrix. To determine the predicted label, we simply look up the label associated with c<sub>a<sub>max</sub></sub> in the associative label-to-index array.

### Training

Training consists of the following very simple algorithm:

for-each training example **_x_**<sub>i</sub><br/>
&nbsp;&nbsp;&nbsp; **Evaluate** **_x_**<sub>i</sub> against **_W_**, comparing the predicted to the actual label<br/>
&nbsp;&nbsp;&nbsp; if the prediction is wrong, **Learn** **_x_**<sub>i</sub>, else continue to the next example
