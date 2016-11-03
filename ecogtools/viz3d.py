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
from surfer import Brain


class ECoGBrain(Brain):
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
    def convert_points_to_2d(self, xyz):
        """Project xyz points onto the camera plane so they are 2d.

        Parameters
        ----------
        xyz : array, shape (n_points, 3)
            The 3d points you wish to convert to 2d. They will be projected
            onto the current plane of the camera.

        Returns
        -------
        xy : array, shape (n_points, 2)
            The xyz points projected onto the current plane of the camera. This
            can now be plotted as a scatterplot along with the image returned
            by `self.screenshot`
        """
        if len(self.foci.values()) == 0:
            raise ValueError('Need a 3d scatterplot to convert to 2d')
        brain = self.brain_matrix[0, 0]
        xy = convert_3d_to_2d(brain._f, xyz)
        return xy

    def screenshot(self, with_foci=False, with_colorbar=False,
                   return_xy_points=True, clean_image=True):
        """Take a snapshot of the current view.

        Parameters
        ----------
        with_focii : bool
            If False, remove foci before taking the screenshot
        with_colorbar : bool
            If False, remove the colorbar before taking the screenshot
        return_xy_points : bool
            If True, also return xy values of all foci plotted on the brain for
            the returned screenshot.
        clean_image : bool
            If True, crop and remove the background of the returned image, with
            xy points scaled accordingly.

        Returns
        -------
        im : ndarray, shape (m, n, 3)
            An image of the current mayavi view.
        [xy] : ndarray, shape (n_points, 2)
            The xy points of all foci on the image.
        """
        if with_foci is False:
            for ifoc in self.foci.values():
                ifoc.visible = False
        if with_colorbar is False:
            self._colorbar_visibility(False, 0, 0)

        im = super(type(self), self).screenshot()

        if with_foci is False:
            for ifoc in self.foci.values():
                ifoc.visible = True
        if with_colorbar is False:
            self._colorbar_visibility(True, 0, 0)

        if clean_image is True and return_xy_points is False:
            raise ValueError('Need xy points to clean image')
        if return_xy_points is True:
            foci = self.foci.values()
            xyz = np.hstack([(ig.mlab_source.x,
                              ig.mlab_source.y,
                              ig.mlab_source.z) for ig in foci]).T
            xy = self.convert_points_to_2d(xyz)

            if clean_image is True:
                bg_color = np.mean(self._bg_color)
                im, xy = crop_and_remove_background(im, xy, bg_color)
            return im, xy
        return im

    def add_foci_surface_activity(self, xyz, activity, spread=10, **kwargs):
        """Set activity on the brain according to foci at vertices.

        Currently, activity is maximal at specified vertices, and spreads
        spherically outwards according to spread. It drops off linearly with
        distance.

        Parameters
        ----------
        xyz : array, shape (n_active_points, 3)
            If ixs_triangles is not given, the xyz coordinates of focii of
            activation.
        activity : array, shape (n_active_points,) | scalar
            The max activity level to plot at each center. If scalar, the same
            activity value is plotted at all points given in triangles or xyz
        spread : int
            The extent to which activity spreads outward from the center. It
            will taper off linearly with distance.
        """
        xyz_activity = np.atleast_2d(xyz)
        xyz_space = np.vstack([val.coords for val in self.geo.values()])
        activity = np.atleast_1d(activity)
        if activity.shape[0] == 1:
            activity = np.repeat(activity, xyz.shape[0])
        elif activity.shape[0] != xyz.shape[0]:
            raise ValueError('Activity and focii length mismatch')

        # Iterate through centroids and calculate scalings
        act_all = np.zeros([xyz_space.shape[0]])
        for act_centroid, act in zip(xyz_activity, activity):
            # Distance from this centroid to all points in the space
            dist_from_activity = xyz_space - act_centroid
            dist_from_activity = np.sqrt((dist_from_activity ** 2).sum(1))

            # Now clip the maximum distance it spreads
            dist_clip = 1 - np.clip(dist_from_activity, 0, spread) / spread
            act_all += dist_clip

        # Create the overlay
        self.add_data(act_all, **kwargs)


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
