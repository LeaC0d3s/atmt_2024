import torch

from itertools import count
from queue import PriorityQueue


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """

    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue()  # beams to be expanded
        self.final = PriorityQueue()  # beams that ended in EOS

        self._counter = count()  # for correct ordering of nodes with same score

    def add(self, score, node):
        """ Adds a new beam search node to the queue of current nodes """
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        # Adds a completed node to the queue, marking it as complete
        # Pad the sequence
        if node.length < self.max_len:
            missing = self.max_len - node.length
            node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad] * missing).long()))
        # node.is_complete = True  # Mark the node as complete
        #print(node.sequence)
        self.nodes.put((score, next(self._counter), node))  # Keep the completed node in the queue


    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        print(self.nodes.qsize())

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        node = merged.get()
        node = (node[0], node[2])

        return node

    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """

        #print("I'm currently here hahahahha")
        # Keep track of how many search paths are already finished (EOS) # unused in new approach
        finished = self.final.qsize()

        # Identify the best finished hypothesis in the current nodes
        temp_queue = PriorityQueue()

        # Determine the best score of completed hypotheses
        best_finished_score = None

        # Extract nodes and populate the temporary queue
        while not self.nodes.empty():
            node = self.nodes.get()
            temp_queue.put((node[0], next(self._counter), node[2]))
            if node[2].is_complete:
                if best_finished_score is None or node[0] < best_finished_score:
                    best_finished_score = node[0]

        #print(best_finished_score)
        nodes = PriorityQueue()
        # Retain the best nodes while respecting the beam size
        current_beam_size = 0
        while not temp_queue.empty() and current_beam_size < self.beam_size:
            node = temp_queue.get()
            #print(node[2].is_complete)
            #print(best_finished_score)
            #print(node[0])
            if node[2].is_complete or (best_finished_score is None or node[0] < best_finished_score):
                #print("got added")
                nodes.put(node)
                current_beam_size += 1


        self.nodes = nodes
        print(self.nodes.qsize())


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""

    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):
        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search
        self.is_complete = False

    def eval(self, alpha=0.0):
        """ Returns score of sequence up to this node

        params:
            :alpha float (default=0.0): hyperparameter for
            length normalization described in in
            https://arxiv.org/pdf/1609.08144.pdf (equation
            14 as lp), default setting of 0.0 has no effect

        """
        normalizer = (5 + self.length) ** alpha / (5 + 1) ** alpha
        return self.logp / normalizer
