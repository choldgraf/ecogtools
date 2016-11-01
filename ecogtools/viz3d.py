"""Utilities for 3D visualization with plotly"""
import matplotlib.pyplot as plt
import numpy as np
from plotly.tools import FigureFactory as ff
from plotly import graph_objs as go
from plotly import offline as offl
from copy import deepcopy
from .viz import array_to_plotly
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi import mlab


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
    def __init__(self, colormap=None, xyz=None, triangles=None, **kwargs):
        self.cmap = plt.cm.Greys if colormap is None else colormap
        self.surfacedata = None
        self.scatterdata = None
        self.fig = mlab.figure(**kwargs)

    def add_surface(self, xyz, triangles, lighting=None, gyri_mask=None):
        """Add a surface model to the plotly figure.

        xyz : array, shape (n_vertices, 3)
            An xyz array defining the position of each vertex in 3-D
        triangles : array, shape (n_triangles, 3)
            An ijk array defining triangles for the mesh. Each row
            indexes 3 rows of in xyz, which together make a triangle.
        lighting : None | dict
            A dictionary specifying lighting parameters in plotly
        gyri_mask : None | array, shape (n_triangles)
            A boolean array specifying gyri and sulci. If specified, plots
            will show a different base facecolor for each.
        """
        if lighting is None:
            lighting = dict(ambient=.4, specular=1)
        if gyri_mask is not None:
            if gyri_mask.shape[0] != triangles.shape[0]:
                raise ValueError('Gyri mask must be same length as triangles')
            if gyri_mask.ndim != 1:
                raise ValueError('Gyri mask must be 1-D')
            self.gyri_mask = gyri_mask
        self.xyz = xyz
        self.x, self.y, self.z = xyz.T
        self.triangles = triangles
        self.activity = np.ones_like(self.x)

        if gyri_mask is not None:
            colors = np.where(gyri_mask, .4, .6)
        else:
            colors = np.repeat(.5, len(xyz))

        self.surfacedata = mlab.triangular_mesh(
            self.x, self.y, self.z, self.triangles,
            colormap=self.cmap.name, scalars=colors)

        self.tri_centroids = xyz[triangles].mean(1)

    def set_activity(self, activity, triangles=None, xyz=None, spread=20,
                     vmin=None, vmax=None, cmap=None):
        """Set activity on the brain according to foci at vertices.

        Currently, activity is maximal at specified vertices, and spreads
        spherically outwards according to spread. It drops off linearly with
        distance.

        Parameters
        ----------
        activity : array, shape (n_active_points,) | scalar
            The max activity level to plot at each center. If scalar, the same
            activity value is plotted at all points given in triangles or xyz
        ixs_triangles : array, shape (n_active_points,)
            The indices of self.triangles corresponding to centers of activity
        xyz : array, shape (n_active_points, 3)
            If ixs_triangles is not given, the xyz coordinates of focii of
            activation.
        spread : int
            The extent to which activity spreads outward from the center. It
            will taper off linearly with distance.
        """
        if self.surfacedata is None:
            raise ValueError('Add surface before setting activity')
        activity = np.atleast_1d(activity)
        vmin = activity.min() if vmin is None else vmin
        vmax = activity.max() if vmax is None else vmax
        if cmap is None:
            cmap = 'hot'
        else:
            self.cmap = getattr(plt.cm, cmap)

        if triangles is not None:
            triangles = np.atleast_1d(triangles)
            activity_centroids = self.tri_centroids[triangles]
        else:
            activity_centroids = xyz
        if len(activity) == 1:
            activity = np.repeat(activity, len(activity_centroids))
        if len(activity_centroids) != len(activity):
            raise ValueError('Activity and ixs length mismatch')

        # Scale activity to 0 and 1
        activity = (activity - vmin) / float(vmax - vmin)
        activity = np.clip(activity, 0, 1)

        # Iterate through centroids and calculate scalings
        act_all = np.zeros([self.xyz.shape[0]])
        for act_centroid, act in zip(activity_centroids, activity):
            dist_from_activity = self.xyz - act_centroid
            dist_from_activity = np.sqrt((dist_from_activity ** 2).sum(1))
            dist_clip = 1 - np.clip(dist_from_activity, 0, spread) / spread
            act_all += dist_clip

        # Now constrain act_all to be between 0 and 1 for plotting
        act_all = np.clip(act_all, 0, 1)
        self.activity = act_all

        # Remove old surface and plot new
        self.surfacedata = mlab.triangular_mesh(
            self.x, self.y, self.z, self.triangles,
            colormap=self.cmap.name, scalars=act_all)

    def add_scatter_3d(self, xyz):
        """Add scatterplot data to a surface plot.

        This is mostly useful for seeing where electrodes are located
        in order to position the camera properly before taking a snapshot.

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
        self.scatterpts = xyz
        self.scatterdata = mlab.points3d(x, y, z)

    def convert_to_2d(self, rem_value=None):
        """Take a snapshot of the current view and return xy points."""
        xy = convert_3d_to_2d(self.fig, self.scatterpts)
        img = take_snapshot(scatter=self.scatterdata)
        img, xy = crop_and_remove_background(img, xy[:, :2],
                                             rem_value=rem_value)

        return img, xy


def convert_3d_to_2d(fig, xyz):
    """Convert 3d points to 2d, given a camera position.

    Parameters
    ----------
    fig : instance of Mayavi figure
        The Mayavi figure for your plot.
    xyz : array, shape (n_points, 3)
        The 3-D coordinate to convert to 2d.

    Returns
    -------
    xy : array, shape (n_points, 2)
        The 2d coordinates of each point in xyz projected onto the
        camera of the Mayavi figure.
    """
    if xyz.shape[1] == 3:
        xyz = np.column_stack([xyz, np.ones(xyz.shape[0])])

    # applying the first transform will give us 'unnormalized' view
    # coordinates we also have to get the transform matrix for the
    # current scene view
    comb_trans_mat = _get_world_to_view_matrix(fig.scene)
    view_coords = _apply_transform_to_points(
        xyz, comb_trans_mat)

    # to get normalized view coordinates, we divide through by the fourth
    # element
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

    # the last step is to transform from normalized view coordinates to
    # display coordinates.
    view_to_disp_mat = _get_view_to_display_matrix(fig.scene)
    xy = _apply_transform_to_points(
        norm_view_coords, view_to_disp_mat)
    xy = xy[:, :2]
    return xy


def get_camera_view():
    """Return the camera view."""
    az, el, dist, foc = mlab.view()
    return az, el, dist, foc


def set_camera_view(ax, el, dist, foc):
    """Set the camera view."""
    mlab.view(az, el, dist, foc)


def take_snapshot(scatter=None, fig=None, ):
    """Return an image of the brain w/o scatterplots.

    Parameters
    ----------
    scatter : instance of Mayavi 3d scatterplot
        The scatterplot object created when calling points3d.
    fig : instance of Mayavi figure
        If `scatter` is not given, the figure that the scatterplot
        is embedded in. The first component of the figure is assumed
        to be the scatterplot object.

    Returns
    -------
    img : array, shape (M, N, 3)
        An image of the current scene in Mayavi.
    """
    if scatter is None:
        if fig is None:
            fig = mlab.gcf()
        scatter = fig.children[-1]
    scatter.visible = False
    img = mlab.screenshot()
    scatter.visible = True
    return img


def crop_and_remove_background(im, xy, rem_value=None):
    """Clean up a snapshot image.

    This will set all background pixels to white and crop the image to
    remove as much background as possible. It will also redefine xy
    coordinates to fit in the new cropped image.

    Parameters
    ----------
    im : array, shape (M, N, 3)
        A matplotlib image
    xy : array, shape (n_points, 2)
        The xy points plotted on the array
    rem_value : int | float
        The value to define as the image background. All pixels
        that have this value (averaged across RGB) will be set to white.

    Returns
    -------
    im_crop : array, shape (M, N, 3)
        A matplotlib image after cropping
    xy_crop : array, shape (n_points, 2)
        The xy points plotted on the array repositioned for the crop
    """
    rem_value = 128 if rem_value is None else rem_value
    greys = im.mean(-1)
    rows = greys.mean(1)
    columns = greys.mean(0)

    # X limits
    ixs_x = np.where(columns != rem_value)
    ixs_y = np.where(rows != rem_value)

    xmin = np.min(ixs_x)
    xmax = np.max(ixs_x)
    ymin = np.min(ixs_y)
    ymax = np.max(ixs_y)

    im_crop = im[ymin:ymax, xmin:xmax, :].copy()
    greys_crop = np.mean(im_crop, -1)

    # Remove background
    mask = greys_crop == rem_value
    xbk, ybk = np.where(mask)
    for ix, iy in zip(xbk, ybk):
        im_crop[ix, iy, :3] = [255, 255, 255]

    # Change the xy values
    xy_crop = xy.copy()
    xy_crop[:, 0] = xy_crop[:, 0] - xmin
    xy_crop[:, 1] = xy_crop[:, 1] - ymin
    return im_crop, xy_crop


def set_lights(fig, elevation=5, azimuth=60, intensity=1.,
               n_lights=None, light_num=None):
    """Set light properties."""
    light_num = 0 if light_num is None else light_num
    light = fig.scene.light_manager.lights[light_num]
    for ii in ['elevation', 'azimuth', 'intensity']:
        var = locals()[ii]
        if var is not None:
            setattr(light, ii, var)

    if n_lights is not None:
        fig.scene.light_manager.number_of_lights = n_lights


# --- Mlab Functions ---
def _get_world_to_view_matrix(mlab_scene):
    """returns the 4x4 matrix that is a concatenation of the
    modelview transform and perspective transform. Takes as
    input an mlab scene object."""

    if not isinstance(mlab_scene, MayaviScene):
        raise TypeError('argument must be an instance of MayaviScene')

    cam = mlab_scene.camera

    # The VTK method needs the aspect ratio and near and far
    # clipping planes in order to return the proper transform.
    # So we query the current scene object to get the parameters we need.
    scene_size = tuple(mlab_scene.get_size())
    clip_range = cam.clipping_range
    aspect_ratio = float(scene_size[0])/float(scene_size[1])

    # this actually just gets a vtk matrix object, we can't really
    # do anything with it yet
    vtk_comb_trans_mat = cam.get_composite_perspective_transform_matrix(
        aspect_ratio, clip_range[0], clip_range[1])

    # get the vtk mat as a numpy array
    np_comb_trans_mat = vtk_comb_trans_mat.to_array()

    return np_comb_trans_mat


def _get_view_to_display_matrix(mlab_scene):
    """ Return a 4x4 matrix to convert view coordinates to display coordinates.

    It's assumed that the view should take up the entire window and that the
    origin of the window is in the upper left corner.
    """

    if not (isinstance(mlab_scene, MayaviScene)):
        raise TypeError('argument must be an instance of MayaviScene')

    # this gets the client size of the window
    x, y = tuple(mlab_scene.get_size())

    # normalized view coordinates have the origin in the middle of the space
    # so we need to scale by width and height of the display window and shift
    # by half width and half height. The matrix accomplishes that.
    view_to_disp_mat = np.array([[x/2.0,      0.,   0.,   x/2.0],
                                 [   0.,  -y/2.0,   0.,   y/2.0],
                                 [   0.,      0.,   1.,      0.],
                                 [   0.,      0.,   0.,      1.]])

    return view_to_disp_mat


def _apply_transform_to_points(points, trans_mat):
    """Applies a 4x4 transformation matrix to a matrix of homogeneous points.

    The array of points should have shape Nx4.
    """

    if not trans_mat.shape == (4, 4):
        raise ValueError('transform matrix must be 4x4')

    if not points.shape[1] == 4:
        raise ValueError('point array must have shape Nx4')

    return np.dot(trans_mat, points.T).T
