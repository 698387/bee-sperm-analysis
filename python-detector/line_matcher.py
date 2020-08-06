"""
Author: Pablo Luesia Lahoz
Date: March 2020
Description: Implementation of the lineMatcher class, used to match different
             spermatozoids between photograms
"""

import numpy as np
from itertools import combinations, repeat
from scipy.spatial.distance import cdist

class LineMatcher(object):
    """
    Given different sets of arrays of pixels or lines, it match 
    between it set the arrays that are similar enough, due to
    an error distance. It returns the lines that has at least a given
    number of matchs
    """

    class Match(object):
        """
        The match class, used for the LineMatcher to store and compare matches
        """
        def __init__(self, position, line_indices):
            self.position = position            # Match position
            self.indices = line_indices         # Match indices for each frame
            self.match_counter = 1              # Counter of the match
                
        # Update the current match with a new match
        def update(self, position, line_indices):
            repeated = True
            for index in line_indices:
                if not( index in self.indices): # non repeated
                    repeated = False
                    self.indices.append(index)
            if repeated:
                return 
            # Calculate the new position
            non_norm_pos = self.position * self.match_counter + position
            self.match_counter += 1     # Update the counter
            # Update the position
            self.position = non_norm_pos / self.match_counter
            # Add the new indices
            for index in line_indices:
                if not( index in self.indices): # non repeated
                    self.indices.append(index)

        # Substraction overload
        def __sub__(self, other):
            return np.sum((self.position - other.position)**2)


    """
    Initialize the matcher
    @param max_distance_error is the max error to accept two lines similar
    @param init_lines are the initial set of lines to insert
    @param method the method to estimate the lines. Default is square euclidean
                  
    """
    def __init__(self, max_distance_error, matchs_number, init_line_sets = [],
                 method = "sqeuclidean"):
        self.error = max_distance_error
        self.matchs_number = matchs_number
        self.__line_sets = init_line_sets
        self.method = method


    # Normalice a line inverting its order if needed
    def __normalice_line(line):
        # Compare the beginning to the end
        gr_x = line[0][0] > line[-1][0]
        eq_x = line[0][0] == line[-1][0]
        gr_y = line[0][1] > line[-1][1]
        # The beginning has to be lefter, and if they are equal, upper
        if gr_x or (eq_x and gr_y):
            return np.flip(line, axis = 0)
        else:
            return line

    """
    Add a new line set to performance the match
    @param line_set is the new line set to add
    """
    def add_line_set(self, line_set):
        # Normalice all the lines and append it to the sets of lines
        self.__line_sets.append(
            list(
                map(lambda x: LineMatcher.__normalice_line(x),
                    line_set)))

    """
    Returns all the lines related with the match <<match>>. The result is a 2D
    array. The first dimension are the frames, the second dimension are the
    lines. The line in a index, belongs to the frame in the same index at the
    other dimension
    @param match the match whose lines want to be recovered
    """
    def match2line(self, match):
        frames = np.zeros(len(match.indices))
        lines = []
        for i in range(0,len(match.indices)):
            (f, idx) = match.indices[i]
            frames[i] = f
            lines.append(self.__line_sets[f][idx])

        return [frames, lines]

    """
    Returns all the matches given the parameters
    """
    def matches(self):
        # Extract the means of all lines in each set
        lines_len = len(self.__line_sets[0][0])
        # Extracts all combinations of 2 to performing the distance calculus
        set_indices = np.arange(0, len(self.__line_sets))
        set_idx_combinations = list(combinations(set_indices, 2))
        
        # All the matches found
        found_matches = []
        for (a, b) in set_idx_combinations:
            # Extract distances between line means
            distances = cdist(np.reshape(self.__line_sets[a],(-1,2*lines_len)),
                              np.reshape(self.__line_sets[b],(-1,2*lines_len)),
                              self.method)
            # Filter distances lower than error
            local_matches_idx = np.argwhere(distances < self.error*abs(a-b))
            local_matches = []                          # Local matches found

            # For each match
            for a_l_idx, b_l_idx in local_matches_idx:
                # Estimates position and stores the indices
                match_position = self.__line_sets[a][a_l_idx] \
                                 + self.__line_sets[b][b_l_idx]
                match_position = np.divide(match_position, 2)
                local_matches.append(
                    LineMatcher.Match(match_position,
                                      [(a, a_l_idx),(b, b_l_idx)]))

            # Update the general matches
            iterable_matches = local_matches
            if len(found_matches) == 0:     # Add the first match if it is empty
                found_matches = [local_matches[0]]
                it_matches = local_matches[1:]

            for match in iterable_matches:
                # Match with the matches
                match_difs = np.array(
                    list(map(lambda x: x - match, found_matches)))
                # It exists another match similar
                most_likely = np.argmin(match_difs)
                if match_difs[int(most_likely)] < self.error:
                    # Update the existing match
                    found_matches[most_likely].update(match.position,
                                                      match.indices)
                else:
                    # Create a new match
                    found_matches.append(match)

        # Return the filtered matches by the number of matches
        return list(filter(lambda x: x.match_counter >= self.matchs_number,
                           found_matches))
