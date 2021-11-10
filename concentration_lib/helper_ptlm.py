"""
Helper function for the Phan-Thomas-Learned-Miller small samples bounds.
http://proceedings.mlr.press/v139/phan21a.html
"""

# Author: My Phan <myphan@cs.umass.edu>
#         Patrick Saux <patrick.saux@inria.fr> (minor amendments only)
# https://github.com/myphan9/small_sample_mean_bounds

import numpy as np
import numpy.ma as ma
from numpy.linalg import norm


def upper_triangle_vertices(n, lower=0, upper=1):
    # Calculating the vertices of a n-dimensional polyhedron
    # X1<=X2 <= ...<= Xn, 0<=Xi <=1 for all 1<= i <= n
    # n: the sample size
    # vertices(n) is calculated recursively from vertices(n-1) by adding 1
    # at the last coordinate for all vertices in vertices(n-1),
    # and then add the vertices np.zeros(n)
    # vertices(1) = [0,1]
    # vertices(2) = [[0,0],[0,1], [1,1]]
    # vertices(3) = [[0,0,0], [0,0,1],[0,1,1] , [1,1,1] ]

    v = np.array([lower, upper])

    for i in range(2, n + 1):
        v = [np.append(j, upper) for j in v]
        v = np.append([lower * np.ones(i)], v, axis=0)
    return np.array(v)


def vertices_with_intersection(n, lower, upper, T, z, degree):
    """ Find all vertices of the n-dimensional polyhedron defined by
    x1<=x2 <= ...<= xn, lower<=Xi <=upper for all 1<= i <= n
    and <T,x**degree> <= <T,z**degree>
    where x**degree denote element-wise operation.
        Find all intersection of the plane <T,x**degree> = <T,z**degree> with
            the above polyhedron
    Args:
        n: the dimension of the polyhedron
        lower, upper: lower<=xi <=upper
        T, Tz, degree: <T,x**degree> <= <T,z**degree>
    Output:
        The vertices of the polyhedron and all intersection of the plane
        <T,x**degree> = <T,z**degree> with it.
    """
    T = np.array(T)
    z_squared = np.power(z, degree)
    Tz = np.dot(z_squared, T)
    # List of vertices of the n-dimensional polyhedron
    # X1<=X2 <= ...<= Xn,
    # 0<=Xi <=1 for all 1<= i <= n
    DUMMY_INF = -10000
    if lower == -np.inf:
        vertices = upper_triangle_vertices(n, DUMMY_INF, upper)
    else:
        vertices = upper_triangle_vertices(n, lower, upper)

    vertices_squared = np.power(vertices, degree)

    Tv = vertices_squared @ T - Tz
    # Add all vertices v of the polyhedron such that T(v) <= Tz
    # to the candidate
    candidates = vertices[Tv <= 0]
    # Loop through all edges of the n-dimensional polyhedron
    # X1<=X2 <= ...<= Xn,
    # 0<=Xi <=1 for all 1<= i <= n
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            vi = vertices[i]
            vj = vertices[j]
            add_intersection = False
            if lower != - np.inf:
                # The interval (vi, vj) intersects with the plane T(x) = T(z)
                if Tv[i] * Tv[j] <= 0:
                    add_intersection = True
            else:
                # if the edge is in the plane vi[0] == DUMMY_INF, there is no
                # need to intersect.
                if not (vi[0] == DUMMY_INF and vj[0] == DUMMY_INF):
                    add_intersection = True
            if add_intersection:
                # If vi is [0, 0, 0, 1, 1] and vj is [0, 1, 1, 1, 1] then the
                # line (vi,vj) has the form [0, t, t, 1,1] where 0<= t <= 1 is
                # a real number.
                # We set line_direction = [0, MASKED, MASKED, 1,1]
                line_direction = ma.masked_array(vi, mask=vi != vj)
                # Calculate the value of t so that the point [0, t, t, 1,1] is
                # in the plane T(x) = T(z)
                # In the example [0, t, t, 1,1]:
                # T(z) = T[0]*0 + (T[1]+T[2])* t + T[3]*1 + T[4]*1
                # Therefore:
                # t = (T(z) - (T[0]*0 + T[3]*1 + T[4*1]))/(T[1]+T[2])
                # Find the sum of the coefficient T[1] + T[2] at
                # the masked values
                masked_coeff = np.sum(T[ma.getmask(line_direction)])
                if masked_coeff != 0:
                    rr = (
                        Tz - np.dot(
                            T,
                            np.power(
                                line_direction.filled(fill_value=0), degree
                                )
                            )
                        ) / masked_coeff
                    if (degree == 2 and rr >= 0) or degree == 1:
                        t = rr ** (1 / degree)
                        # Fill the masked value by t
                        intersection = line_direction.filled(fill_value=t)
                        # Add the intersection to the list of candidates
                        candidates = np.append(
                            candidates,
                            [intersection],
                            axis=0
                            )
                elif Tz - np.dot(T, np.power(line_direction, degree)) == 0:
                    # the edge is in the plane (T(x) = T(z))
                    intersection_min = line_direction.filled(fill_value=lower)
                    intersection_max = line_direction.filled(fill_value=upper)
                    candidates = np.append(
                        candidates, [intersection_min, intersection_max],
                        axis=0
                        )

    return candidates


def b_alpha_linear_inner(z,
                         alpha,
                         T,
                         lower=0,
                         upper=1,
                         num_samples=10000,
                         degree=1,
                         ):
    """
    Args:
      z: The sample
      alpha: confidence parameter
      T: the coefficient of the linear function mapping the sample z to a
        real number
      lower: lower bound of the support
      upper: upper bound of the support
      num_samples: the number of Monte Carlo samples
      degree: the function mapping the sample z to a real number is defined as
        the dot product <T, z**degree> where z**degree is element-wise
        operation.
    Examples:
      When the function is the l2 norm, T is a vector of 1, and degree = 2.
      When the functino is the sample mean, T is a vector of 1, and degree = 1.
    Returns:
      b_list: For each Monte Carlo sample u, return the maximum b(x,u) over
        all points in the n-dimensional polyhedron
            x1<=x2 <= ...<= xn,
            lower<=Xi <=upper for all 1<= i <= n
            and <T,x> <= <T,z>
      u_delta_list: the list of [u_{i}- u_{i-1} ,1<=i <= n+1] for all Monte
        Carlo samples u (this is for when T is the l2 norm)
    """

    n = len(z)
    z = np.sort(z)
    b_list = []
    u_list = np.random.rand(num_samples, n)

    # Add U_0 = 0
    u_list0 = np.append(np.zeros((num_samples, 1)), u_list, axis=1)
    # Add U_{n+1} = 1
    u_list01 = np.append(u_list0, np.ones((num_samples, 1)), axis=1)
    sorted_u_list = np.sort(u_list01, axis=1)
    # Calculate the list of [u_{i}- u_{i-1} ,1<=i <= n+1] for all sample U
    u_delta_list = sorted_u_list[:, 1:] - sorted_u_list[:, :n+1]

    candidates = vertices_with_intersection(n, lower, upper, T, z, degree)

    # Add upper to the vertices in the candidate set
    candidates1 = np.append(
        candidates, upper * np.ones((candidates.shape[0], 1)), axis=1
        )
    sorted_candidates1 = np.sort(candidates1, axis=1)
    # Calculate m(x,U) = <x, delta_u>
    b_mat = sorted_candidates1 @ u_delta_list.T
    # For each u, take max_x m(x,u)
    b_list = b_mat.max(axis=0)

    return b_list, u_delta_list


def b_alpha_l2norm(z, alpha, upper=1.0, lower=0.0, num_samples=10000):
    """ This function calculuates the value of the bound when the function T is
        the l2 norm and the lower bound of the support is 0.
    Args:
      z: The samlpe
      alpha: confidence parameter
      T: the coefficient of the linear function mapping the sample z to a
        real number
      upper: upper bound of the support
      num_samples: the number of Monte Carlo samples
    Examples:
      When the function is the l2 norm, T is a vector of 1, and degree = 2.
      When the function is the sample mean, T is a vector of 1, and degree = 1.
    Returns:
      the value of the bound when the function T is the l2 norm and the lower
        bound of the support is 0.
    """
    n = len(z)
    T = np.ones(n)

    b_list, u_delta_list = b_alpha_linear_inner(
        z,
        alpha,
        T,
        lower=lower,
        upper=upper,
        num_samples=num_samples,
        degree=2,
        )

    ascending = np.all(
        u_delta_list[:, :n-1] - u_delta_list[:, 1:n] <= 0, axis=1
        )
    intersection = u_delta_list[:, :n] / norm(
        u_delta_list[:, :n],
        axis=1
        )[:, None] * norm(z)

    intersection_bounded = np.all(intersection <= upper, axis=1)
    in_region = np.logical_and(ascending, intersection_bounded)

    intersection1 = np.append(
        intersection, upper * np.ones((intersection.shape[0], 1)), axis=1
        )

    b_u_intersection = np.einsum('ij,ij->i', intersection1, u_delta_list)

    b_list[in_region] = np.maximum(
        b_list[in_region], b_u_intersection[in_region]
        )
    sorted = np.sort(b_list)
    r = sorted[int(np.ceil((1-alpha)*len(sorted)))]

    return r
