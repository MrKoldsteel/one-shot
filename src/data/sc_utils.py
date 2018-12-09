import numpy as np
from skimage import feature
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import cv2 as cv
import math


def image_to_edge(image, n_pts=150):
    """
    Calculates the canny edges of the input image and randomly samples n_pts
    from it.  Returns the sampled edge points ((i, j) coordinates of the pixels
    in the sampled edge points) as well as the new image.

    Parameters
    ----------
    image : array_like
        a square array representing an image to find the edges of.
    n_pts : integer
        the number of points to be sampled from the edges of image.

    Returns
    -------
    sampled_edge_points : array_like
        n_pts * 2 array with row i representing the coordinates of the
        corresponding point in the retained edge.
    new_image : array
        same size as original image consisting of the sampled edge points from
        the original image as an image array.
    """
    # isolate the edges in the image and the coordinates in the image which
    # define them.
    edges = feature.canny(image)
    edge_points = np.argwhere(edges)

    # find the number of points in the edges and sample n_pts without
    # replacement from 1, ..., n.
    n = edge_points.shape[0]
    inds = np.random.choice(n, n_pts, replace=False)

    # isolate the sampled edge points and construct the image of the sampled
    # edges from these. May want to cut the returning of new_image for the
    # version of this that will be used for data processing as it will not be
    # used for this.  Really just returned for getting a feel of whether n_pts
    # is sufficient to retain the feel of the image.
    sampled_edge_points = edge_points[inds, :]
    new_image = np.zeros_like(edges)
    i, j = sampled_edge_points[:, 0],  sampled_edge_points[:, 1]
    new_image[i, j] = 1
    return sampled_edge_points, new_image


def point_to_edge_distances(point, edges):
    """
    Calculates the euclidean distance from point to each of the points in the
    collection of edges in edges.

    Parameters
    ----------
    point : array_like
        a 2*1 numpy array containing the coordinates of the point to calculate
        distances from.
    edges : array_like
        a n*2 numpy array containing the collection of points we would like to
        calculate the distance from point to.

    Returns
    -------
    distances : array_like
        an array of size n containing the respective distances from the points
        in edges to point.
    """
    return np.apply_along_axis(
                            lambda x: euclidean(x, point),
                            axis=1,
                            arr=edges
                        )


# Some utilities for numerically stable calculation of the angle between
# two vectors.
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def point_edge_angles(point, edges):
    """
    Calculates angles between point and all of the points in the array edges
    which correspond to the points on the boundary of an image.

    Parameters
    ----------
    point : array_like
        a 2*1 numpy array containing the coordinates of the point to calculate
        distances from.
    edges : array_like
        a n*2 numpy array containing the collection of points we would like to
        calculate the distance from point to.

    Returns
    -------
    angles : array_like
        an array of size n containing the respective angles between point and
        each of the points in edges.
    """
    return np.apply_along_axis(
            lambda x: angle_between(point, x), axis=1, arr=edges
        )


def image_to_shape_contexts(image, n_pts=150,
                            n_radii_bins=5, n_angle_bins=12):
    """
    Returns an array consisting of the shape context histograms for an image.
    binning to form the shape contexts is done in log polar coordinates.

    Parameters
    ----------
    image : array_like
        a square array containing pixel values for the image
    n_pts : integer
        the number of points to retain in the edge sampling of the image
    n_radii_bins : integer
        the number of values to bin over on the radius.  This needs to be done
        so that bin sizes are constant on the log_2 scale.
    n_angle_bins : integer
        the number of values to bin over on the angles.

    Returns
    -------
    shape_contexts : array_like
        an array of size rows * columns where:
            rows = n_pts
            columns = n_radii_bins * n_angle_bins
        and each row contains the histogram defining the shape context for
        the corresponding point in the retained image.
    """
    # set indices corresponding to the edge points and the distance and
    # angle matrices we need to populate.  row i of these contain the
    # distances and angles between point i and the rest of the points in
    # the edge set.
    inds = np.arange(n_pts)
    distances = np.zeros((n_pts, n_pts - 1))
    angles = np.zeros((n_pts, n_pts - 1))

    # set the shape context matrix and the get the sampled edge points
    shape_contexts = np.zeros((n_pts, n_radii_bins * n_angle_bins))
    sampled_edge_points, _ = image_to_edge(image, n_pts)

    for ind in inds:
        distances[ind, :] = point_to_edge_distances(
                        sampled_edge_points[ind],
                        sampled_edge_points[inds != ind]
                    )
        angles[ind, :] = point_edge_angles(
                        sampled_edge_points[ind],
                        sampled_edge_points[inds != ind]
                    )

    # set the bin ends for the radii and angle bins
    alpha = np.median(distances)
    exponents = np.arange(n_radii_bins, dtype=float) - n_radii_bins / 2
    distance_bins = alpha * np.append([0], 2 ** exponents)
    angle_bins = np.arange(n_angle_bins + 1) / n_angle_bins

    # find the radii and angle bin indices
    distance_bin_inds = np.searchsorted(distance_bins, distances, side='right')
    angle_bin_inds = np.searchsorted(angle_bins, angles, side='right')

    for i in np.arange(1, 1 + n_radii_bins):
        for j in np.arange(1, 1 + n_angle_bins):
            # Count the number of points that lie in distance bin i and
            # radius bin j to add counts for another bin of the shape contexts
            # histogram.
            shape_contexts[:, n_angle_bins * (i - 1) + (j - 1)] = np.sum(
                (distance_bin_inds == i) & (angle_bin_inds == j),
                axis=1
            )

    return shape_contexts / shape_contexts.sum(axis=1).reshape(-1, 1)


# The next portion of code attempts to vectorize what was done above and
# will try to write tests relying on what seem to be properly functioning
# utilities which are already in place.  The next function is a utility taken
# from stack overflow which removes the diagonal of a square matrix
# taken from stack-overflow.  isolates the diagonal of a matrix.
def skip_diag(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(
            A.ravel()[1:],
            shape=(m-1,m),
            strides=(s0+s1,s1)
        ).reshape(m,-1)


def edges_to_distances(edges):
    """
    Inputs an n * 2 array of points on the edge of an image and outputs an
    array of pairwise distances.

    Parameters
    ----------
    edges : array_like
        an n * 2 array with row i representing the i-th point on the boundary
    or edge of an image.

    Returns
    -------
    distances : array_like
        an n * n array with element (i, j) representing the euclidean distance
    between points i and j in the edges array (i.e. rows i and j in this array)
    """
    # n_pts = edges.shape[0]
    # pairwise_distances = edges.reshape(n_pts, 1, 2) - edges.reshape(1, n_pts, 2)
    # return np.sqrt((pairwise_distances ** 2).sum(axis=-1))
    return cdist(edges, edges, metric='euclidean')


def edges_to_angles(edges):
    """
    Inputs an n*2 array of points on the edge of an image and outputs an array
    of pairwise angles.

    Parameters
    ----------
    edges : array_like
        an n * 2 array with row i representing the i-th point on the boundary
    or edge of an image.

    Returns
    -------
    angles : array_like
        an n * n array with element (i, j) representing the angle between the
    points i and j in the edges array (i.e. rows i and j in this array)
    """
    # The code below was buggy; calculating angles between edge points and
    # not relative angles.  Hopefully the fix makes a big difference.
    # n_pts = edges.shape[0]
    # convert edges to an array of unit vectors for stability
    # unit_edges = edges / np.apply_along_axis(
    #                                lambda x: np.linalg.norm(x),
    #                                1,
    #                                edges
    #                            ).reshape(-1, 1)
    # reshape to different dimensions to broadcast inner products
    # u, v = unit_edges.reshape(n_pts, 1, 2), unit_edges.reshape(1, n_pts, 2)
    # apply the algorithm taken from stack overflow employed in
    # angle angle_between.
    # return np.arccos(np.clip((u * v).sum(axis=-1), -1, 1))
    #return np.arccos(np.clip(np.dot(unit_edges, unit_edges.T), -1, 1))

    # define a function to calculate the relative angle
    def mod_get_angles(u, v):
        theta = np.arctan2((v - u)[1], (v - u)[0])
        if theta >= 0:
            return theta
        return 2 * np.pi + theta

    return cdist(edges, edges, lambda u, v: mod_get_angles(u, v))


def cosine_distances(edge_set_1, edge_set_2):
    """
    Inputs two n*2 edge sets and outputs the pairwise cosine distances between
    the points in the two sets.

    Parameters
    ----------
    edge_set_1 : array_like
        an n * 2 array with row i representing the i-th point on the boundary
    of an image.
    edge_set_2 : array_like
        an n * 2 array with row i representing the i-th point on the boundary
    of another image.

    Returns
    -------
    cosine_distances : array_like
        an n*n array with element (i, j) representing the cosine distance
    between edge i of edge_set_1 and edge j of edge_set_2.
    """
    n, _ = edge_set_1.shape

    def unit_norm(x):
        if np.all(x == 0):
            return np.inf
        return np.linalg.norm(x)

    u = edge_set_1 / np.apply_along_axis(
                                    lambda x: unit_norm(x),
                                    1,
                                    edge_set_1
                                ).reshape(-1, 1)
    v = edge_set_2 / np.apply_along_axis(
                                    lambda x: unit_norm(x),
                                    1,
                                    edge_set_2
                                ).reshape(-1, 1)
    # u, v = u.reshape(n, 1, 2), v.reshape(1, n, 2)
    # return (1 - np.clip((u * v).sum(axis=-1), -1, 1)) / 2
    return (1 - np.clip(np.dot(u, v.T), -1, 1)) / 2


def edge_from_image_Jitendra(image, n_pts=100, oversampling_rate=3, ddepth=cv.CV_16S, scale=1, delta=0):
    # extract full edge and gradient information
    full_edge = np.argwhere(feature.canny(image))
    # dx, dy = np.gradient(image)
    # try open cv for derivative estimation.. seems to be smoother at the edges.
    dx = cv.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    dy = cv.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    # set the number of samples (oversampled from edges)
    # to select n_pts from and randomly draw this many indices
    N = full_edge.shape[0]
    N_samples = np.min([n_pts * oversampling_rate, N])
    inds = np.random.choice(N, N_samples, replace=False)

    # calculate distances between edge points and set diagonal
    # to infinity.
    dists = cdist(full_edge, full_edge)
    np.fill_diagonal(dists, np.inf)

    # isolate edges to sample from and distances to scan over
    potential_edge = full_edge[inds]
    dists = dists[inds, :][:, inds]

    # Now cycle through inds and keep picking points while
    # inds is not empty and we don't yet have 100 pts.
    # Also set up arrays to store edges and angles.
    edges, angles = [], []
    not_finished = True
    while not_finished:
        # Select a close pair of points
        m, _ = np.argwhere(dists == dists.min()).shape
        i, j = np.argwhere(dists == dists.min())[np.random.choice(m)]

        # add to the edges set and angles set
        x, y = potential_edge[i]
        edges.append((x, y))
        # radians = math.atan2(py[x, y], px[x, y])
        # angles.append(radians + 2 * math.pi * (radians < 0))
        # angles.append(np.arctan2(dy[y, x], dx[y, x]) + np.pi / 2)
        angles.append((dx[x, y], dy[x, y]))

        # and then remove the point from potential_edges and
        potential_edge = np.delete(potential_edge, i, 0)
        dists = np.delete(np.delete(dists, i, 0), i, 1)
        if len(edges) >= n_pts:
            not_finished = False

    return np.array(edges), np.array(angles)


def vec_image_to_shape_contexts(image, n_pts=150,
                                n_radii_bins=5, n_angle_bins=12,
                                r_min=0.125, r_max=2, units=np.mean):
    """
    A vectorized version of image_to_shape_contexts. Performs the exact same
    computation as that function, so refer to its docstring.
    """
    # set indices corresponding to the edge points and the distance and
    # angle matrices we need to populate.  row i of these contain the
    # distances and angles between point i and the rest of the points in
    # the edge set.
    sampled_edge_points, tangent_angles = edge_from_image_Jitendra(image, n_pts=n_pts)
    # sampled_edge_points, _ = image_to_edge(image, n_pts)
    shape_contexts = np.zeros((n_pts, n_radii_bins * n_angle_bins))
    distances = skip_diag(edges_to_distances(sampled_edge_points))
    angles = skip_diag(edges_to_angles(sampled_edge_points))

    # set the bin ends for the radii and angle bins
    alpha = units(distances)
    # exponents = np.arange(n_radii_bins, dtype=float) - n_radii_bins / 2
    # distance_bins = alpha * np.append([0], 2 ** exponents)
    distance_bins = np.logspace(
                                start=np.log(r_min),
                                stop=np.log(r_max),
                                num=1+n_radii_bins,
                                base=np.exp(1)
                            )
    angle_bins = 2 * np.pi * np.arange(n_angle_bins + 1) / n_angle_bins

    # find the radii and angle bin indices
    distance_bin_inds = np.searchsorted(
                                distance_bins,
                                distances / alpha,
                                side='right'
                            )
    angle_bin_inds = np.searchsorted(angle_bins, angles, side='right')

    for i in np.arange(1, n_radii_bins):
        for j in np.arange(1, n_angle_bins):
            # Count the number of points that lie in distance bin i and
            # radius bin j to add counts for another bin of the shape contexts
            # histogram.
            shape_contexts[:, n_angle_bins * i + j] = np.sum(
                (distance_bin_inds == i + 1) & (angle_bin_inds == j + 1),
                axis=1
            )

    return (tangent_angles,
            shape_contexts / shape_contexts.sum(axis=1).reshape(-1, 1))
    #return (sampled_edge_points,
    #        shape_contexts / shape_contexts.sum(axis=1).reshape(-1, 1))
