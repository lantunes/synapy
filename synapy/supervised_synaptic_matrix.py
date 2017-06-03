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

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class SupervisedSynapticMatrix:
    def __init__(self, input_length, b, c, k):
        self._synaptic_matrix = np.ones((input_length, 1), dtype=int)
        self._b = b
        self._c = c
        self._k = k
        self._current_column = 0
        self._class_cells_to_labels = {}
        self._labels_to_class_cells = {}

    def train(self, input_example, label):
        predicted_label = self.evaluate(input_example)
        if predicted_label != label:
            class_cell = self._current_column
            self.learn(input_example)
            self._class_cells_to_labels[class_cell] = label
            if label not in self._labels_to_class_cells:
                self._labels_to_class_cells[label] = []
            self._labels_to_class_cells[label].append(class_cell)
            logging.info("prediction incorrect; trained next available class cell (%d)" % class_cell)
        else:
            logging.info("prediction correct")

    def learn(self, input_example):
        eligibility_vector = np.multiply(input_example, self._synaptic_matrix[:, self._current_column])
        number_of_eligible_synapses = np.sum(eligibility_vector)
        update_factor = self._c + (self._k / number_of_eligible_synapses)
        self._synaptic_matrix[:, self._current_column] = np.add(self._b, np.multiply(eligibility_vector, update_factor))
        self._expand_synaptic_matrix()

    def evaluate(self, input):
        most_active_class_cell = np.argmax(np.dot(input, self._synaptic_matrix))
        if most_active_class_cell in self._class_cells_to_labels :
            return self._class_cells_to_labels[most_active_class_cell]

    def relative_spike_frequencies(self, input):
        _, cols = np.shape(self._synaptic_matrix)
        if cols < 2:
            raise ValueError("cannot obtain relative spike frequencies on untrained synaptic matrix")
        all_frequencies = np.dot(input, self._synaptic_matrix[:, :-1])  #exclude the last column, which was added after learning
        relative_frequencies = dict.fromkeys(self._labels_to_class_cells.keys(), 0)
        for i, frequency in enumerate(all_frequencies):
            label = self._class_cells_to_labels[i]
            if frequency > relative_frequencies[label]:
                relative_frequencies[label] = frequency
        return relative_frequencies

    def _expand_synaptic_matrix(self):
        rows, cols = np.shape(self._synaptic_matrix)
        expanded_matrix = np.ones((rows, cols + 1), dtype=int)
        expanded_matrix[:, :-1] = self._synaptic_matrix
        self._synaptic_matrix = expanded_matrix
        self._current_column += 1
