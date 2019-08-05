import os
import numpy as np
import pkg_resources

from scipy import ndimage

from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.io import fits

import pandas as pd

import matplotlib.pyplot as plt


dsharp_sources = pd.read_csv(
    pkg_resources.resource_filename(__name__, os.path.join('data', 'DSHARP_sources.csv')),
    skipinitialspace=True)


def download_disk(disk, suffix='continuum', type='image', authenticate=False):
    """
    Download the specified disk from the project website (password protected).

    Arguments:
    ----------

    disk : str
        disk name such as 'AS 209', ...

    Keywords:
    ---------

    suffix : str
        specifies the version of the image, possibilities include
        'continuum' or 'CO'

    authenticate : bool
        wether or not authentication should be used (asking for username/passw.)
    """

    fname = '{}_{}'.format(disk.replace(' ', ''), suffix)
    if type == 'image':
        url = 'https://almascience.eso.org/almadata/lp/DSHARP/images/{}.fits'
    elif type == 'profile':
        url = 'https://almascience.eso.org/almadata/lp/DSHARP/profiles/{}.profile.txt'
    elif type == 'SED':
        url = 'https://almascience.eso.org/almadata/lp/DSHARP/SEDs/{}.SED.txt'
    else:
        raise ValueError('type must be image, profile, or SED!')

    url = url.format(fname)

    if not os.path.isfile(fname):
        import requests
        import getpass

        if authenticate:
            username = getpass.getpass(prompt='Username: ')
            password = getpass.getpass(prompt='Password: ')
            req = requests.get(url, auth=(username, password), stream=True)
        else:
            req = requests.get(url, stream=True)

        print('Downloading file \'{}\' ... '.format(fname), end='', flush=True)

        fullpath = pkg_resources.resource_filename(__name__, os.path.join('data', fname))

        with open(fullpath, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print('Done!')
    else:
        print('file already exists, will not download: {}'.format(fname))


def plot_DHSARP_disk(
        disk='HD 163296', suffix='script', fname=None, cmap='inferno', dpc=None,
        ax=None, range=200, p0=[0, 0], **kwargs):
    """
    Plot the specified disk where the version is given by the suffix

    Arguments:
    ----------

    disk : str
        disk name such as 'AS 209', ...

    Keywords:
    ---------

    suffix : str
        specifies the version of the image, possibilities include
        'taper', 'hires', 'script'

    fname : str
        to set a specific, possibly non-DSHARP filename

    dpc : float
        if set it is used as distance in parsec, otherwise it's read from
        the LP_sources data or from input

    ax : None | axes
        in which axes to plot - if none, will create new figure

    p0 : list of length 2
        shift of image center in au

    range : float
        plotting range around the center

    cmap : cmap | string
        what's passed to pcolormesh as cmap

    title : None | string
        if given, overwrites title plottet in the figure, which defaults to
        the name of the disk

    Other keywords are passed to plot_fits

    Output:
    -------
    returns figure and axes handle
    """
    if fname is None:
        source = dsharp_sources.loc[disk]
        fname = '{}_{}_image.fits'.format(disk.replace(' ', ''), suffix)
        if not os.path.isfile(fname):
            print('no file {} was found'.format(fname))
            yn = ''
            while yn not in ['y', 'n']:
                yn = input('download it [y/n]: ').lower()
            if yn == 'y':
                download_disk(disk.replace(' ', ''), suffix=suffix)
            else:
                print('quitting')
                return

        dpc = dpc or source['distance [pc]']

    else:
        while dpc is None:
            try:
                dpc = float(input('distance in PC for {}: '.format(fname)))
            except ValueError:
                dpc = None

    with fits.open(fname) as hdulist:
        header = hdulist[0].header
        pixel_size_x = header['CDELT1'] * 3600. * 1e3
        pixel_size_y = header['CDELT2'] * 3600. * 1e3

    disk = kwargs.pop('title', disk)

    fig, ax = plot_fits(
        fname=fname, ax=ax, cmap=cmap, range=range, p0=p0,
        pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y, dpc=dpc,
        title=disk, **kwargs)

    return fig, ax


def plot_fits(
        fname, ax=None, cmap='inferno', range=None, p0=[0, 0], pixel_size_x=1,
        pixel_size_y=1, dpc=None, vmin=None, vmax=None, rsqaure=False,
        title=None, coronagraph_mask=None, fct='pcolormesh', beam=None,
        autoshift=False):
    """
    fname : float
        path to file

    ax : None | axes
        where to plot the figure, create if None

    cmap : colormap
        which colormap to pass to pcolormesh

    range : None | float
        which range in mas (or au if dpc given) to plot around the center

    mask : None | float
        what size of center circle to plot

    p0 : list of two floats
        where the center is located in mas

    pixel_size : float
        size of a pixel in mas

    dpc : float
        distance in parsec

    vmin, vmax : float
        which lower and upper bound to use

    rsquare : bool
        if true, multiply the intensity with r**2

    title : str
        title to plot in top left corner

    coronagraph_mask : None | float
        size of central circle to e.g. cover coronograph region
        in mas (or au if dpc given)

    beam : list
        beam size (FWHM) for convolution in mas

    fct : str
        which bound method of the axes object to use for plotting.
        For transparent pdfs, it's for example better to use the slower pcolor
        while pcolormesh is much faster.

    autoshift : bool
        to put the brightest pixel in the center
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    hdulist = fits.open(fname)
    Snu = np.squeeze(hdulist[0].data)

    x = np.arange(Snu.shape[0]) * abs(pixel_size_x)  # in mas
    y = np.arange(Snu.shape[1]) * abs(pixel_size_y)  # in mas

    if dpc is not None:
        x *= dpc * 1e-3
        y *= dpc * 1e-3
        if beam is not None:
            beam = np.array(beam) * dpc * 1e-3

    x -= x[-1] / 2.0
    y -= y[-1] / 2.0
    x -= p0[0]
    y -= p0[1]

    if autoshift:
        cy, cx = np.unravel_index(Snu.argmax(), Snu.shape)
        x -= x[cx]
        y -= y[cy]

    if range is not None:
        ix0 = np.abs(x + range).argmin()
        ix1 = np.abs(x - range).argmin()
        iy0 = np.abs(y + range).argmin()
        iy1 = np.abs(y - range).argmin()

        x = x[ix0:ix1 + 1]
        y = y[iy0:iy1 + 1]
        Snu = Snu[iy0:iy1 + 1, ix0:ix1 + 1]

    std = Snu[:20, :20].std()
    if vmin is None:
        vmin = 1.5 * std
    if vmax is None:
        vmax = 100 * std  # 0.75 * Snu.max()

    print('{}: vmin = {:.2g}, vmax = {:.2g}'.format(title, vmin, vmax))

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

    if beam is not None:
        print('beam = {}'.format(beam))
        sigma = beam / (2 * np.sqrt(2 * np.log(2)))
        Snu = ndimage.gaussian_filter(Snu, sigma)

    getattr(ax, fct)(
        -x, y, Snu, cmap=cmap, norm=norm, rasterized=True,
        # edgecolor=(1.0, 1.0, 1.0, 0.3), linewidth=0.0015625
        )

    ax.set_aspect('equal')
    if dpc is None:
        ax.set_xlabel(r'$\Delta$RA [mas]')
        ax.set_ylabel(r'$\Delta$DEC [mas]')
    else:
        ax.set_xlabel(r'$\Delta$RA [au]')
        ax.set_ylabel(r'$\Delta$DEC [au]')

    if range is not None:
        ax.set_xlim([range, -range])
        ax.set_ylim([-range, range])
    else:
        ax.set_xlim(ax.get_xlim()[::-1])

    if title is not None:
        ax.text(0.05, 0.95, title, color='w', transform=ax.transAxes, verticalalignment='top')

    if coronagraph_mask is not None:
        ax.add_artist(plt.Circle((0, 0), radius=coronagraph_mask, color='0.5'))

    ax.set_facecolor('k')

    return fig, ax
