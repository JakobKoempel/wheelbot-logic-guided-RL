import jax.numpy as jp
from jax import vmap
import jax
from dataclasses import dataclass

def distance_to_segment(pos_xy: jp.ndarray, a: jp.ndarray, b: jp.ndarray):
    # Compute the projection of pos onto the segment a_b
    a_b = b - a
    a_pos = pos_xy - a
    t = jp.dot(a_pos, a_b) / jp.dot(a_b, a_b)
    t = jp.clip(t, 0, 1)  # Clamp to segment
    closest = a + t * a_b
    return jp.linalg.norm(pos_xy - closest)

def angle_of_segment(a: jp.ndarray, b: jp.ndarray):
    #computes angle of line segment a->b to x axis. returns angle in radians in [-π, π]
    d = b - a 
    return jp.arctan2(d[1], d[0])

def angle_difference(angle1, angle2):
    #computes the angle to add to angle2 so that it aligns with angle1. Result lies within (-pi, pi)
    diff = angle1 - angle2 
    return (diff + jp.pi) % (2 * jp.pi) - jp.pi 

def points_to_line_segemnt(points: jp.ndarray) -> jp.ndarray:
    """
    Convert array of points to array of line segments i.e. pairs of points
    input cannot include any padding!
    """
    next_points = jp.roll(points, shift=-1, axis=0)
    line_segment = jp.stack([points, next_points], axis=1)

    return line_segment

def pad_line_segments_to_size(points: jp.ndarray, size: int) -> jp.ndarray:
    """
    Pads a (N, 2, 2) array of point pairs with [[1e10, 1e10], [1e10, 1e10]] rows until it reaches the given size.
    """
    n = points.shape[0]
    if n > size:
        raise Exception(f"Cannot pad points of size {n} to size {size}. Points already have more elements than size.")
    pad_count = size - n
    padding_pair = jp.array([[1e9, 1e9],
                            [1e10, 1e10]])
    pad = jp.tile(padding_pair[None, :, :], (pad_count, 1, 1))
    return jp.concatenate([points, pad], axis=0)

@dataclass(frozen=True)
class Trajectory:
    line_segments: jp.ndarray
    size: int
    lookahead: int = 10

    def get_next_points(self, index : int) -> jp.ndarray:
        """
        Retrieve the next elements starting from index
        """            
        size = self.size
        indices = (jp.arange(0, self.lookahead) + index) % size
        next_line_segments = self.line_segments[indices]
        next_points = next_line_segments[:, 0]
        return next_points
        
    def find_closest_segment_index(self, pos_xy):
        dists = vmap(lambda pair: distance_to_segment(pos_xy, *pair))(self.line_segments)
        return jp.argmin(dists)
    
    def cross_track_error(self, pos_xy):
        #returns the distance to the trajectory
        dists = vmap(lambda pair: distance_to_segment(pos_xy, *pair))(self.line_segments)
        return jp.min(dists)
    
    def cross_angle_error(self, pos_xy, yaw_angle):
        #returns the differene between the current trajectory angle and the current yaw_angle
        min_i = self.find_closest_segment_index(pos_xy)
        min_line_segment = self.line_segments[min_i]
        min_line_segment_angle = angle_of_segment(*min_line_segment)
        return angle_difference(min_line_segment_angle, yaw_angle)
    
    def get_next_distances(self, pos_xy):
        #compute closest line segment
        min_i = self.find_closest_segment_index(pos_xy)
        #index of the endpoint of the closest line segment
        next_i = min_i + 1
        next_points = self.get_next_points(next_i)
        distances = vmap(lambda point: jp.linalg.norm(point - pos_xy))(next_points)
        return distances
    
    def get_next_angles(self, pos_xy, yaw_angle):
        #compute closest line segment
        min_i = self.find_closest_segment_index(pos_xy)
        #index of the endpoint of the closest line segment
        next_i = min_i + 1
        next_points = self.get_next_points(next_i)
        #compute angles of direction for the next points
        direction_angles = vmap(lambda point: angle_of_segment(pos_xy, point))(next_points)
        #compute relative angle i.e. how must the yaw change to align with direction angle
        relative_angles = vmap(lambda angle: angle_difference(angle, yaw_angle))(direction_angles)
        return relative_angles
    
    def get_state(self, pos_xy, yaw_angle):
        e_ct = self.cross_track_error(pos_xy)
        e_ca = self.cross_angle_error(pos_xy, yaw_angle)
        distances = self.get_next_distances(pos_xy)
        angles = self.get_next_angles(pos_xy, yaw_angle)
        return jp.array([e_ct, e_ca, *distances, *angles])
    
    def get_init_pos(self, index: int):
        """
        computes the starting position and angle of the wheelbot
        for a given index on the trajectory
        """
        index = index % self.size
        init_line_segment = self.line_segments[index]
        init_xy = init_line_segment[0]
        init_angle = angle_of_segment(*init_line_segment)

        return init_xy, init_angle
    
    def get_rand_init_pos(self, rng: jp.ndarray):
        index = jax.random.randint(rng, (), 0, self.size)
        init_xy, init_angle = self.get_init_pos(index)
        return init_xy, init_angle

def check_equal(a, b):
    assert jp.allclose(a, b, atol=1e-6), f'{a} =! {b}'


def main():  
    points = jp.array([[2,-2],[2,-5],[5,-5],[5,-2]])
    line_segments = points_to_line_segemnt(points)
    padded_line_segments = pad_line_segments_to_size(line_segments, 6)

    traj = Trajectory(padded_line_segments, 4, 4)

    pos_xy = jp.array([4,-1])
    yaw = -jp.pi
    init_xy_0, init_angle_0 = traj.get_init_pos(0)
    init_xy_1, init_angle_1 = traj.get_init_pos(1)
    init_xy_2, init_angle_2 = traj.get_init_pos(2)
    init_xy_3, init_angle_3 = traj.get_init_pos(3)

    rng = jax.random.PRNGKey(3)
    init_xy_4, init_angle_4 = traj.get_rand_init_pos(rng)

    check_equal(init_xy_0, jp.array([2, -2]))
    check_equal(init_angle_0, -jp.pi / 2)
    check_equal(init_xy_1, jp.array([2, -5]))
    check_equal(init_angle_1, 0)
    check_equal(init_xy_2, jp.array([5, -5]))
    check_equal(init_angle_2, jp.pi / 2)
    check_equal(init_xy_3, jp.array([5, -2]))
    check_equal(init_angle_3, jp.pi)
    check_equal(traj.cross_track_error(pos_xy), 1)
    check_equal(traj.cross_angle_error(pos_xy, yaw), 0)
    check_equal(traj.get_next_distances(pos_xy), jp.array([jp.sqrt(5), jp.sqrt(20),jp.sqrt(17), jp.sqrt(2)]))
    check_equal(traj.get_next_angles(pos_xy, yaw)[3], 3*jp.pi / 4)

    points = jp.array([[3,2],[3,3],[2,3],[1,3],[1,2],[1,1],[2,1],[3,1]]) * 0.001
    line_segments = points_to_line_segemnt(points)
    padded_line_segments = pad_line_segments_to_size(line_segments, 20)
    traj = Trajectory(padded_line_segments, 8, 8)
    pos_xy = jp.array([2,2]) * 0.001
    yaw = 0

    check_equal(traj.cross_track_error(pos_xy), 0.001)
    check_equal(traj.cross_angle_error(pos_xy, yaw), jp.pi / 2)
    check_equal(traj.get_next_distances(pos_xy), jp.array([jp.sqrt(2),1,jp.sqrt(2),1,jp.sqrt(2),1,jp.sqrt(2), 1]) *0.001)
    check_equal(traj.get_next_angles(pos_xy, yaw), jp.array([jp.pi/4, jp.pi/2, 3*jp.pi/4, -jp.pi, -3*jp.pi/4, -jp.pi/2, -jp.pi/4, 0]))
    
    points = jp.array([[0,0],[2,2],[4,0],[2,-2]])
    line_segments = points_to_line_segemnt(points)
    padded_line_segments = pad_line_segments_to_size(line_segments, 20)
    traj = Trajectory(padded_line_segments, 4, 4)
    init_xy_0, init_angle_0 = traj.get_init_pos(0)
    init_xy_1, init_angle_1 = traj.get_init_pos(1)
    init_xy_2, init_angle_2 = traj.get_init_pos(2)
    init_xy_3, init_angle_3 = traj.get_init_pos(3)

    check_equal(init_xy_0, jp.array([0, 0]))
    check_equal(init_angle_0, jp.pi / 4)
    check_equal(init_xy_1, jp.array([2, 2]))
    check_equal(init_angle_1, -jp.pi / 4)
    check_equal(init_xy_2, jp.array([4, 0]))
    check_equal(init_angle_2, -3 * jp.pi / 4)
    check_equal(init_xy_3, jp.array([2, -2]))
    check_equal(init_angle_3, 3 * jp.pi / 4)
    print('All tests passed')

if __name__ == "__main__":
    main()