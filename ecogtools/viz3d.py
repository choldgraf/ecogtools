"""Utilities for 3D visualization with plotly"""
import matplotlib.pyplot as plt
import numpy as np
from plotly.tools import FigureFactory as ff
from plotly import graph_objs as go
from plotly import offline as offl
from copy import deepcopy
from .viz import array_to_plotly


class ActivitySurfacePlot(object):
    """Plot brain activity using a brain surface using plotly.

    Parameters
    ----------
    colormap : matplotlib colormap
        The colormap for the brain surface.
    eye : array, shape (3,)
        The xyz coordinates of the starting camera position.
    zoom : float | int, must be > 0
        The zooming factor. Larger values are more zoomed in. Values < 1
        correspond to zooming in.
    init_notebook : bool
        Whether to initialize plotly interactive notebook mode.

    Attributes
    ----------
    surfacedata : plotly Mesh3d object | None
        The data returned by calling _trisurf in plotly.
    scatterdata : plotly Scatter3d object | None
        The data returned by calling Scatter3d in plotly.
    """
    def __init__(self, colormap=None, eye=None, zoom=1, init_notebook=True):
        if eye is None:
            self.eye = dict(x=1.25, y=1.25, z=1.25)
        else:
            self.eye = deepcopy(eye)
        for key, val in self.eye.items():
            self.eye[key] = val / float(zoom)
        self.camera = dict(eye=self.eye)
        self.layout = dict(scene=dict(camera=self.camera))
        self.cmap = plt.cm.Greys if colormap is None else colormap
        self.surfacedata = None
        self.scatterdata = None
        if init_notebook is True:
            offl.init_notebook_mode()

    def add_surface(self, xyz, triangles, lighting=None, **kwargs):
        """Add a surface model to the plotly figure.

        xyz : array, shape (n_vertices, 3)
            An xyz array defining the position of each vertex in 3-D
        triangles : array, shape (n_triangles, 3)
            An ijk array defining triangles for the mesh. Each row
            indexes 3 rows of in xyz, which together make a triangle.
        lighting : None | dict
            A dictionary specifying lighting parameters in plotly
        """
        if lighting is None:
            lighting = dict(ambient=.4, specular=1)
        self.xyz = xyz
        self.x = xyz.T[0]
        self.y = xyz.T[1]
        self.z = xyz.T[2]
        self.triangles = triangles
        self.xrange = np.array([xyz.min(0)[0], xyz.max(0)[0]])
        self.yrange = np.array([xyz.min(0)[1], xyz.max(0)[1]])
        self.zrange = np.array([xyz.min(0)[2], xyz.max(0)[2]])

        colors = self.cmap(np.repeat(.5, len(triangles)))
        colors = array_to_plotly(colors)
        self.surfacedata = ff._trisurf(
            x=self.x, y=self.y, z=self.z, simplices=self.triangles,
            color_func=colors, **kwargs)
        self.surfacedata[0]['lighting'] = lighting
        self.facecolors = self.surfacedata[0]['facecolor'].copy()
        self.tri_centroids = xyz[triangles].mean(1)
        self.layout['scene'].update(dict(xaxis=dict(range=self.xrange),
                                         yaxis=dict(range=self.yrange),
                                         zaxis=dict(range=self.zrange)))

    def set_activity(self, ixs_triangles, activity, spread=20,
                     vmin=None, vmax=None, cmap=None):
        """Set activity on the brain according to foci at vertices.

        Currently, activity is maximal at specified vertices, and spreads
        spherically outwards according to spread. It drops off linearly with
        distance.

        Parameters
        ----------
        ixs_triangles : array, shape (n_active_verts,)
            The indices of self.triangles corresponding to centers of activity
        activity : array, shape (n_active_verts,)
            The max activity level to plot at each center
        spread : int
            The extent to which activity spreads outward from the center. It
            will taper off linearly with distance.
        """
        if self.surfacedata is None:
            raise ValueError('Add surface before setting activity')
        ixs_triangles = np.atleast_1d(ixs_triangles)
        activity = np.atleast_1d(activity)
        vmin = activity.min() if vmin is None else vmin
        vmax = activity.max() if vmax is None else vmax
        cmap = plt.cm.hot if cmap is None else cmap
        if len(activity) == 1:
            activity = np.repeat(activity, len(ixs_triangles))
        if len(ixs_triangles) != len(activity):
            raise ValueError('Activity and ixs length mismatch')

        # Scale activity to 0 and 1
        activity = (activity - vmin) / float(vmax - vmin)
        activity = np.clip(activity, 0, 1)

        # Iterate through centroids and calculate scalings
        act_all = np.zeros([self.triangles.shape[0]])
        for ix, act in zip(ixs_triangles, activity):
            act_centroid = self.xyz[self.triangles[ix]].mean(0)
            dist_from_activity = self.tri_centroids - act_centroid
            dist_from_activity = np.sqrt((dist_from_activity ** 2).sum(1))
            dist_clip = 1 - np.clip(dist_from_activity, 0, spread) / spread
            act_all += dist_clip

        # Now constrain act_all to be between 0 and 1 for plotting
        act_all = np.clip(act_all, 0, 1)

        # Finally convert to facecolor strings and update data
        colors = cmap(act_all)
        colors = array_to_plotly(colors)
        msk_active = act_all != 0
        new_facecolors = self.facecolors.copy()
        new_facecolors[msk_active] = colors[msk_active]
        self.surfacedata[0]['facecolor'] = new_facecolors

    def add_scatter_3d(self, xyz, surfacecolor=None, edgecolor=None,
                       s=None, alpha=1):
        """Add scatterplot data to a surface plot.

        Parameters
        ----------
        xyz : array, shape (n_points, 3)
            The xyz coordinates of points in the scatterplot
        s : array, shape (n_points,) | (1,)
            The size of each scatterplot.
        """
        xyz = np.atleast_1d(xyz)
        if xyz.ndim == 1 or xyz.shape[-1] == 1:
            # Assume values are triangle indices
            xyz = self.xyz[self.triangles[xyz]]
            # Take the average over the 3 points in the triangle
            xyz = xyz.mean(1)
        if xyz.shape[-1] != 3:
            raise ValueError('xyz must be shape (n_points, 3) if'
                             ' not triangle ixs')
        x, y, z = xyz.T
        if edgecolor is not None:
            edgecolor = np.atleast_1d(edgecolor)
            if len(edgecolor) == 1:
                edgecolor = np.repeat(edgecolor, xyz.shape[0])
        if surfacecolor is not None:
            surfacecolor = np.atleast_1d(surfacecolor)
            if len(surfacecolor) == 1:
                surfacecolor = np.repeat(surfacecolor, xyz.shape[0])

        if s is None:
            s = np.repeat(50, xyz.shape[0])
        else:
            s = np.atleast_1d(s)
            if len(s) == 1:
                s = np.repeat(s, xyz.shape[0])

        self.scatterdata = go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=s, color=surfacecolor, opacity=alpha,
                        line=dict(color=edgecolor)))

    def pull_scatter_to_eye(self, pull_amt=10):
        """Pull scatterplot points closer to camera.

        Makes it easier to visualize scatterplots so they don't intersect
        with the brain.
        """
        if self.scatterdata is None:
            raise ValueError('Make scatterplot before calling this function.')
        eye = np.array([self.eye[ii] for ii in ['x', 'y', 'z']])
        points = np.array([self.scatterdata[ii] for ii in ['x', 'y', 'z']]).T
        newpts = pull_points_to_ref(eye, points, pull_amt=pull_amt)
        newpts = newpts.T
        for key, pts in zip(['x', 'y', 'z'], newpts):
            self.scatterdata[key] = pts

            # Now update the ranges if they've changed
            this_axis = self.layout['scene']['%saxis' % key]
            this_axis['range'] = [np.min(np.hstack([this_axis['range'], pts])),
                                  np.max(np.hstack([this_axis['range'], pts]))]

    def find_nearest_triangles(self, xyz):
        """Find nearest surface triangle to a set of xyz positions.

        Parameters
        ----------
        xyz : array, shape (n_points, 3)
            The xyz positions you wish to map onto trinagles

        Returns
        -------
        ixs : shape (n_points,)
            Indices corresponding to the nearest triangle for each xyz.
        """
        xyz = np.asarray(xyz)
        ixs = np.zeros(xyz.shape[0])
        for ii, pt in enumerate(xyz):
            dists = self.tri_centroids - pt
            dists = (dists ** 2).sum(-1)
            ixs[ii] = np.argmin(dists)
        return ixs.astype(int)

    def plot(self, surface=True, scatter=True,
             interactive=True, filename=None):
        """Create a plotly plot with the data in self.

        This will either embed the 3d visualization in the notebook
        (interactive == True), or export it as an HTML file
        (interactive == False)
        """
        data = []
        if surface is True and self.surfacedata is not None:
            data += self.surfacedata
        if scatter is True and self.scatterdata is not None:
            data += [self.scatterdata]
        fig = go.Figure(data=data, layout=self.layout)
        if interactive is True:
            offl.iplot(fig)
        elif filename is not None:
            offl.plot(fig, filename=filename)
        else:
            raise ValueError('If not interactive, must specify filename.')
        return fig


def pull_points_to_ref(base_point, data_points, pull_amt=10):
    """Pull xyz values in `data_points` towards `base_point`.

    Parameters
    ----------
    base_point : array, shape (3,)
        The xyz coordinates of the base point.
    data_points : array, shape (n_points, 3)
        The xyz coordinates of points to pull towards base point.
    pull_amt : int | float
        The amount to pull each data point towards the base point.
    """
    # Distance vector between the two
    diff_vectors = data_points - base_point[np.newaxis, :]
    norm_vectors = np.linalg.norm(diff_vectors, axis=-1)[:, np.newaxis]
    # Normalize to unit length, and scale to amount we want
    unit_vectors = diff_vectors / norm_vectors
    move_vectors = unit_vectors * pull_amt
    return data_points + move_vectors
