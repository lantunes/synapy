Synapy
======

Synapy is a Python implementation of a synaptic matrix. To learn more about the synaptic matrix, visit the page of the original [Java implementation](https://github.com/lantunes/synaptic-matrix).

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