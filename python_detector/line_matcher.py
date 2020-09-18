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
        def __init__(self, positions, line_indices):
            # Check the number of position and indices
            if len(line_indices) != len(positions):
                raise Exception("Not the same number of positions than indices")
            if len(line_indices) < 2:
                raise Exception("Matches only exists with 2 or more lines")
            
            self.indices = np.array(line_indices)# Match indices for each frame
            # Speed and acceleration of the line matched
            self.speed, self.acceleration = LineMatcher.Match.\
                                            __estimate_mov(positions,
                                                           self.indices)
            self.positions = positions          # Match positions
            # Counter of the match
            self.match_counter = len(line_indices)              


        # Given a set of line indices, and a set of positions, it caluclates
        # the mean speed and acceleration
        def __estimate_mov(positions, line_indices):
            # Time order. The soonest the first
            t_sorted_indices = np.argsort(line_indices, 
                                          axis = 0)[:,0]
            # Time diff
            time_diff = np.diff(line_indices[t_sorted_indices, 0])\
                          .reshape((-1,1,1))
            # Speed of the match
            typed_positions = np.array(positions, subok=True)
            speed = np.diff(typed_positions[t_sorted_indices],
                            axis = 0) / time_diff
            # Estimate the acceleration
            if len(line_indices) > 2:
                acceleration =  np.diff(speed, axis = 0) / time_diff[:-1]
            else:
                acceleration = np.full((1,len(positions[0]),2), 0.0)
            # Returns the mean speed and acceleration
            return np.mean(speed, axis = 0), np.mean(acceleration, axis = 0)

                
        # Update the current match with a new match
        def update(self, match):
            counter = 0
            # For all the lines involved in the match
            for idx in match.indices:
                if not idx[0] in self.indices[:,0]:# The line hasn't been added
                    # Update the indices
                    self.indices = np.append(self.indices, [idx], axis = 0)    
                    new_position = match.positions[counter]
                    # Update the positions, robust to reverse order
                    flip = np.sum((self.positions[0] - new_position)**2) >\
                           np.sum((self.positions[0] \
                                   - np.flip(new_position, axis = 0))**2)
                    if flip:
                        self.positions.append(np.flip(new_position, axis = 0))
                    else:
                        self.positions.append(new_position)
                    self.match_counter += 1     # Update the counter
                counter += 1
            # Update the speed and the acceleration
            self.speed, self.acceleration = LineMatcher.Match.\
                                            __estimate_mov(self.positions,
                                                           self.indices)

        # Distance between two lines, and true if it needs to be flipped
        @staticmethod
        def distance(line_a, line_b):
            # Calculate normal distance and the distance with one in reversed order
            dist = np.sum((line_a - line_b)**2)
            f_dist = np.sum((line_a - np.flip(line_b, axis = 0))**2)
            return (dist, False) if f_dist > dist else (f_dist, True)

        # Predicts the position at the moment time
        def predict_pos(self, time):
            # Extracts the elapsed time for each line in the current match
            diff_time = np.array(
                list(map(lambda t: time - t[0], self.indices)) )
            # Repeat the speed and the acceleration of the match to calculate
            #rep_ela_time = np.repeat(np.reshape([diff_time], (-1,1) ),
            #                     len(self.positions[0]),
            #                     axis = 1)
            rep_ela_time = np.reshape( [[diff_time]],(-1,1,1) )
            rep_speed = np.repeat([self.speed], self.match_counter, axis = 0)
            rep_acc = np.repeat([self.acceleration], self.match_counter,
                                axis = 0)
            # Extracts the current position: s_0 + v*t + (a*t**2)/2
            return np.mean(self.positions + rep_speed*rep_ela_time\
                            + (rep_acc*rep_ela_time**2)/2,
                               axis = 0)

        # Returns the mean time of the match
        def time(self):
            return np.mean(self.indices[:,0])

        # Substraction overload
        def __sub__(self, other):
            # Mean of the time and the positions
            current_time = np.mean(other.indices, axis = 0)[0]
            current_pos = np.mean(other.positions, axis = 0)
            # Predict the position
            pred_pos = self.predict_pos(current_time)
            # Distance between the predicted and the seen positions
            return LineMatcher.Match.distance(pred_pos, current_pos)

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
                match_positions = [ self.__line_sets[f1][a_l_idx],
                                    np.flip(self.__line_sets[f2][b_l_idx], 
                                            axis = 0) ]
            else:
                match_positions = [ self.__line_sets[f1][a_l_idx],
                                    self.__line_sets[f2][b_l_idx] ]
            local_matches.append(
                LineMatcher.Match(match_positions,
                                    [[f1, a_l_idx],[f2, b_l_idx]]))

        return local_matches

    # Extracts the local matches between 2 frames
    def __extract_local_matches(self, f1, f2):
        error = self.error*abs(f1-f2)
        distances = self.__line_sets_distances(f1, f2)
        # Filter distances lower than error
        local_matches_idx = np.argwhere(distances[:,:,0] < error)
        return self.__generate_matches(f1, f2, local_matches_idx,
                                       distances[:,:,1])


    # Update the global matches with the local matches
    def __update_global_matches(self, found_matches, local_matches):
        # No local matches
        if not len(local_matches):
            return found_matches    # No changes

        # Iterator of the matches
        it_matches = local_matches

        # Update the general matches
        if len(found_matches) == 0:     # Add the first match if it is empty
            found_matches = [local_matches[0]]
            it_matches = local_matches[1:]

        # Predict the position from the old matches to the new ones
        time = local_matches[0].time()
        predicted_positions = np.array(list(map(lambda x: 
                                                x.predict_pos(time),
                                                found_matches)))

        for match in it_matches:
            match_difs = np.array(
                list(map(lambda x: self.method(match.predict_pos(time), x),
                         predicted_positions)))
            # It exists another match similar
            most_likely = np.argwhere(match_difs[:,0] < self.error)
            if len(most_likely) > 0:
                # Update the existing match
                found_matches[most_likely[0,0]].update(match)
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
                predicted_positions = np.append(predicted_positions,
                                                [match.predict_pos(time)],
                                                axis = 0)
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

        # Matches between consecutive frames
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
