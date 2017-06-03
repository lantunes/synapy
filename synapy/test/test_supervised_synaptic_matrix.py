# Copyright (c) 2017 Luis M. Antunes <lantunes@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import unittest

from synapy import SupervisedSynapticMatrix


class TestSupervisedSynapticMatrix(unittest.TestCase):

    def test_one_example_per_class(self):
        synaptic_matrix = SupervisedSynapticMatrix(9, b=1, c=2, k=10)

        example1 = [0, 1, 0,
                    0, 1, 0,
                    0, 1, 0]

        example2 = [1, 1, 1,
                    1, 0, 1,
                    1, 1, 1]

        synaptic_matrix.train(example1, label=1)
        synaptic_matrix.train(example2, label=2)

        to_evaluate = [1, 1, 1,
                       1, 0, 0,
                       0, 1, 1]

        self.assertEquals(2, synaptic_matrix.evaluate(to_evaluate))
        self.assertEquals({1: 16, 2: 24}, synaptic_matrix.relative_spike_frequencies(to_evaluate))

    def test_one_example_per_class_with_string_classes(self):
        synaptic_matrix = SupervisedSynapticMatrix(9, b=1, c=2, k=10)

        example1 = [0, 1, 0,
                    0, 1, 0,
                    0, 1, 0]

        example2 = [1, 1, 1,
                    1, 0, 1,
                    1, 1, 1]

        synaptic_matrix.train(example1, label="line")
        synaptic_matrix.train(example2, label="box")

        to_evaluate = [1, 1, 1,
                       1, 0, 0,
                       0, 1, 1]

        self.assertEquals("box", synaptic_matrix.evaluate(to_evaluate))
        self.assertEquals({"line": 16, "box": 24}, synaptic_matrix.relative_spike_frequencies(to_evaluate))

    def test_multiple_examples_per_class(self):
        synaptic_matrix = SupervisedSynapticMatrix(49, b=1, c=10, k=500)

        example1_0 = [0, 1, 1, 1, 1, 1, 1,
                      0, 1, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0,
                      0, 1, 1, 1, 1, 0, 0,
                      0, 1, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0]
        synaptic_matrix.train(example1_0, label=0)

        example1_1 = [0, 0, 0, 1, 0, 0, 0,
                      0, 0, 1, 0, 1, 0, 0,
                      0, 1, 0, 0, 0, 1, 0,
                      0, 1, 1, 1, 1, 1, 0,
                      0, 1, 0, 0, 0, 1, 0,
                      0, 1, 0, 0, 0, 1, 0,
                      0, 1, 0, 0, 0, 1, 0]
        synaptic_matrix.train(example1_1, label=1)

        example2_0 = [0, 0, 0, 0, 0, 0, 0,
                      0, 1, 1, 1, 1, 0, 0,
                      0, 1, 0, 0, 0, 0, 0,
                      0, 1, 1, 1, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0]
        synaptic_matrix.train(example2_0, label=0)

        example2_1 = [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 1, 0, 1, 0, 0,
                      0, 1, 0, 0, 0, 1, 0,
                      0, 1, 1, 1, 1, 1, 0,
                      0, 1, 0, 0, 0, 1, 0,
                      0, 0, 0, 0, 0, 0, 0]
        synaptic_matrix.train(example2_1, label=1)

        to_evaluate = [0, 0, 0, 0, 0, 0, 0,
                       0, 1, 1, 1, 1, 1, 0,
                       0, 1, 0, 0, 0, 0, 0,
                       0, 1, 1, 1, 1, 0, 0,
                       0, 1, 0, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0, 0]

        self.assertEquals(0, synaptic_matrix.evaluate(to_evaluate))
        self.assertEquals({0: 613, 1: 423}, synaptic_matrix.relative_spike_frequencies(to_evaluate))

    def test_one_example_per_class_with_fractional_inputs(self):
        synaptic_matrix = SupervisedSynapticMatrix(9, b=1, c=2, k=10)

        example1 = [0.0, 0.9, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.9, 0.0]

        example2 = [1.0, 0.7, 0.9,
                    1.1, 0.0, 1.3,
                    0.8, 1.2, 0.9]

        synaptic_matrix.train(example1, label="line")
        synaptic_matrix.train(example2, label="box")

        to_evaluate = [1.1, 1.0, 0.7,
                       0.8, 0.0, 0.0,
                       0.0, 1.1, 1.0]

        self.assertEquals("box", synaptic_matrix.evaluate(to_evaluate))
        self.assertEquals({'box': 20.100000000000001, 'line': 16.199999999999999},
                          synaptic_matrix.relative_spike_frequencies(to_evaluate))
