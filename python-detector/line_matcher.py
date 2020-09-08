"""
Author: Pablo Luesia Lahoz
Date: March 2020
Description: Implementation of the lineMatcher class, used to match different
             spermatozoids between photograms
"""

import numpy as np
from itertools import combinations, repeat

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
            self.match_counter = 2              # Counter of the match
                
        # Update the current match with a new match
        def update(self, match, line_sets):
            # Position to update
            upd_position = self.position * self.match_counter
            # For all the lines involved in the match
            for idx in match.indices:
                if not idx in self.indices: # The line hasn't added
                    self.indices.append(idx)    # Update the indices
                    # Calculate the new position
                    position2include = line_sets[idx[0]][idx[1]]
                    flip = np.sum((self.position - position2include)**2) >\
                           np.sum((self.position - np.flip(position2include, axis = 0))**2)
                    if flip:
                        upd_position += np.flip(position2include, axis = 0)
                    else:
                        upd_position += position2include
                    self.match_counter += 1     # Update the counter
            # Update the position
            self.position = upd_position / self.match_counter

        # Distance between two lines, and true if it needs to be flipped
        @staticmethod
        def distance(line_a, line_b):
            # Calculate normal distance and the distance with one in reversed order
            dist = np.sum((line_a - line_b)**2)
            f_dist = np.sum((line_a - np.flip(line_b, axis = 0))**2)
            return (dist, False) if f_dist > dist else (f_dist, True)


        # Substraction overload
        def __sub__(self, other):
            # Distance
            return LineMatcher.Match.distance(self.position, other.position)

    """
    Initialize the matcher
    @param max_distance_error is the max error to accept two lines similar
    @param init_lines are the initial set of lines to insert                  
    """
    def __init__(self, max_distance_error, matchs_number, init_line_sets = []):
        self.error = max_distance_error
        self.matchs_number = matchs_number
        self.__line_sets = init_line_sets
        self.method = LineMatcher.Match.distance

    # Compute the distances between the line sets f1 and f2. Robust to 
    # direction. Returns the distance and the flipped distances
    def __line_sets_distances(self, f1, f2):
        # Extract distances between line means
        return np.array( list( map( lambda a:
                                  list( map(lambda b: self.method(a,b),
                                            self.__line_sets[f2]) ),
                              self.__line_sets[f1]) ) )


    # Generate the matches given the f1 and f2 frames, and the lines matched
    def __generate_matches(self, f1, f2, local_matches_idx, flip_value = []):
        local_matches = []                          # Local matches found

        # For each match
        for a_l_idx, b_l_idx in local_matches_idx:
            # Estimates position and stores the indices
            if flip_value[a_l_idx, b_l_idx]:    # Inverts order if needed
                match_position = self.__line_sets[f1][a_l_idx] \
                                    + np.flip(self.__line_sets[f2][b_l_idx], 
                                              axis = 0)
            else:
                match_position = self.__line_sets[f1][a_l_idx] \
                                    + self.__line_sets[f2][b_l_idx]

            match_position = np.divide(match_position, 2)
            local_matches.append(
                LineMatcher.Match(match_position,
                                    [(f1, a_l_idx),(f2, b_l_idx)]))

        return local_matches

    # Extracts the local matches between 2 frames
    def __extract_local_matches(self, f1, f2):
        error = self.error*abs(f1-f2)
        distances = self.__line_sets_distances(f1, f2)
        # Filter distances lower than error
        local_matches_idx = np.argwhere(distances[:,:,0] < error)
        return self.__generate_matches(f1, f2, local_matches_idx, distances[:,:,1])


    # Update the global matches with the local matches
    def __update_global_matches(self, found_matches, local_matches):
        it_matches = local_matches

        # Update the general matches
        if len(found_matches) == 0:     # Add the first match if it is empty
            found_matches = [local_matches[0]]
            it_matches = local_matches[1:]

        for match in it_matches:
            # Match with the matches
            match_difs = np.array(
                list(map(lambda x: x - match, found_matches)))
            # It exists another match similar
            most_likely = np.argwhere(match_difs[:,0] < self.error)
            if len(most_likely) > 0:
                # Update the existing match
                found_matches[most_likely[0,0]].update(match,
                                                     self.__line_sets)
                # Combine the last match joined
                for matches2combine in most_likely[0,1:]:
                    found_matches[most_likely[0,0]].update(
                        found_matches[matches2combine],
                        self.__line_sets)
                # Delete the combined matches
                found_matches = np.delete(found_matches, most_likely[0,1:])
            else:
                # Create a new match
                found_matches = np.append(found_matches, [match], axis = 0)
        return found_matches


    """
    Add a new line set to performance the match
    @param line_set is the new line set to add
    """
    def add_line_set(self, line_set):
        # Normalice all the lines and append it to the sets of lines
        self.__line_sets.append(line_set)

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

        # All the matches found
        found_matches = np.array([], dtype = object)

        # Extracts all combinations of 2 to performing the distance calculus
        #set_indices = np.arange(0, len(self.__line_sets))
        #set_idx_combinations = list(combinations(set_indices, 2))
        #for (a, b) in set_idx_combinations:
        for i in range(0, len(self.__line_sets) -1):
            a = i
            b = i + 1
            # Extracts the local matches
            local_matches = self.__extract_local_matches(a, b)
            # Update the global matches
            found_matches = self.__update_global_matches(found_matches,
                                                         local_matches)

        # Return the filtered matches by the number of matches
        return list(filter(lambda x: x.match_counter >= self.matchs_number,
                           found_matches))
