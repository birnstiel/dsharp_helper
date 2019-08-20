import os
import numpy as np
import pkg_resources

from scipy import ndimage

from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.io import fits
from astropy import constants as c

import pandas as pd

import matplotlib.pyplot as plt

# create a pandas data frame of the source information

sources = pd.read_csv(
    pkg_resources.resource_filename(__name__, os.path.join('data', 'DSHARP_sources.csv')),
    skipinitialspace=True)
sources.index = pd.Index([name.replace(' ', '') for name in sources.index])


def purge_data(disks=None):
    """Remove the downloaded DSHARP data files from the package.

    Keywords:
    ---------
    disks : None | str | list
        can be None, then all are removed
        can be a str, then the data of that disk is removed
        can be a list of str, then the data for those disks is removed
    """
    if disks is None:
        disks = [k for k in sources.index]
    else:
        if type(disks) == str:
            disks = [disks]
        elif type(disks) != list:
            raise TypeError('disks must be None or a disk name (str) or a list of those')

    for disk in disks:
        disk = disk.replace(' ', '')
        for fname in ['{}_continuum.fits', '{}_CO.fits', '{}.profile.txt', '{}.SED.txt']:
            fname = fname.format(disk)
            fullpath = pkg_resources.resource_filename(__name__, os.path.join('data', fname))
            if os.path.isfile(fullpath):
                print('Deleting {}'.format(fname))
                os.unlink(fullpath)


def download_disk(fname, type='image', authenticate=False):
    """
    Download the specified disk from the project website (password protected).

    Arguments:
    ----------

    fname : str
        file name, such as 'AS209_continuum.fits'

    Keywords:
    ---------

    type : str
        specifies the type of file such as image, profile, or SED.

    authenticate : bool
        wether or not authentication should be used (asking for username/passw.)
    """
    if type == 'image':
        url = 'https://almascience.eso.org/almadata/lp/DSHARP/images/' + fname
    elif type == 'profile':
        url = 'https://almascience.eso.org/almadata/lp/DSHARP/profiles/' + fname
    elif type == 'SED':
        url = 'https://almascience.eso.org/almadata/lp/DSHARP/SEDs/' + fname
    else:
        raise ValueError('type must be image, profile, or SED!')

    fullpath = pkg_resources.resource_filename(__name__, os.path.join('data', fname))

    if not os.path.isfile(fullpath):
        import requests
        import getpass

        if authenticate:
            username = getpass.getpass(prompt='Username: ')
            password = getpass.getpass(prompt='Password: ')
            req = requests.get(url, auth=(username, password), stream=True)
        else:
            req = requests.get(url, stream=True)

        print('Downloading file \'{}\' ... '.format(fname), end='', flush=True)

        with open(fullpath, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print('Done!')
    else:
        print('file already exists, will not download: {}'.format(fname))


def get_datafile(disk, suffix='continuum', type='image'):
    """
    Get a path to the local data file, download it if it's not present.

    Arguments:
    ----------

    disk : str
        disk name such as 'AS 209', ...

    Keywords:
    ---------

    suffix : str
        specifies the version of the image, possibilities include
        'continuum', 'CO'

    type : str
        can be 'profile', 'image', or 'SED'
    """

    disk_name = disk.replace(' ', '')
    if type == 'image':
        fname = '{}_{}.fits'.format(disk_name, suffix)
    elif type == 'profile':
        fname = '{}.profile.txt'.format(disk_name)
    elif type == 'SED':
        fname = '{}.SED.txt'.format(disk_name)
    else:
        raise ValueError('type must be image, profile, or SED!')

    fullpath = pkg_resources.resource_filename(__name__, os.path.join('data', fname))

    if not os.path.isfile(fullpath):
        download_disk(fname, type=type)

    return fullpath


def I_nu_from_T_b(T_b, lam_obs=0.125):
    "Calculate Intensity from brightness temperature"
    c_light = c.c.cgs.value
    nu_obs  = c_light / lam_obs
    return 2 * nu_obs**2 * c.k_B.cgs.value * T_b / c_light**2


def get_profile(disk):
    """
    Parameters:
    ----------

    disk : str
        name of disk

    """
    fname = get_datafile(disk, type='profile')
    data = np.loadtxt(fname)

    # radius in arcseconds

    r_as = data[:, 1]

    # intensity in brightness temperature

    T_b   = data[:, 4]
    dT_b  = data[:, 5]  # uncertainty on T_b

    # convert to intensity in CGS
    I_nu    = I_nu_from_T_b(T_b)
    I_nu_u = I_nu_from_T_b(T_b + dT_b)
    I_nu_l = I_nu_from_T_b(T_b - dT_b)

    return {
        'r_as': r_as,
        'I_nu': I_nu,
        'I_nu_u': I_nu_u,
        'I_nu_l': I_nu_l,
        'data': data
        }


def get_sed(disk):
    """
    Parameters:
    ----------

    disk : str
        name of disk

    """
    fname = get_datafile(disk, type='SED')

    mask = np.ones(5, dtype=bool)
    mask[-1] = False
    data = np.loadtxt(fname, usecols=np.arange(4))

    with open(fname) as fid:
        header = ''.join([line for line in fid if line.startswith('#')])

    references = np.loadtxt(fname, usecols=4, dtype=str)

    return {
        'data': data,
        'header': header,
        'references': references
        }


def plot_profile(disk):
    """Plot DSHARP radial profile.

    Parameters
    ----------
    disk : str
        name of disk

    Returns
    -------
        Returns figure and axes object
    """
    data = get_profile(disk)

    r_as = data['r_as']
    I_nu = data['I_nu']
    I_nu_u = data['I_nu_u']
    I_nu_l = data['I_nu_l']

    f, ax = plt.subplots()
    ax.semilogy(r_as, I_nu)
    ax.fill_between(r_as, I_nu_l, I_nu_u, color='r', alpha=0.5)
    ax.set_ylim(0.5 * I_nu.min(), 1.5 * I_nu.max())
    ax.set_xlabel(r'radius [arcsec]')
    ax.set_ylabel(r'$I_\nu$ [erg / (s cm$^2$ Hz sr)]')
    return f, ax


def plot_sed(disk):
    """Plot SED of DSHARP disk.

    Parameters
    ----------
    disk : str
        name of disk

    Returns
    -------
        Returns figure and axes object
    """
    data = get_sed(disk)['data']

    lam_mic = data[:, 0]
    flux_dens = data[:, 1]
    df_stat = data[:, 2]
    df_sys = flux_dens * data[:, 3]

    error = (df_stat**2 + df_sys**2)**0.5

    f, ax = plt.subplots()
    ax.errorbar(lam_mic, flux_dens, yerr=error, ls='', marker='o', barsabove=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'wavelength [micron]')
    ax.set_ylabel(r'$F_\nu$ [Jy]')
    return f, ax


def plot_DHSARP_continuum(
        disk='HD 163296', fname=None, cmap='inferno', dpc=None,
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
        'continuum'

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
        if given, overwrites title plotted in the figure, which defaults to
        the name of the disk

    Other keywords are passed to plot_fits

    Output:
    -------
    returns figure and axes handle
    """
    if fname is None:
        source = sources.loc[disk]
        dpc = dpc or source['distance [pc]']
        fname = get_datafile(disk)

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
        autoshift=False, PA=None):
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

    PA : None | float
        if float: rotate by this amount
    """
    from scipy.ndimage import rotate
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    hdulist = fits.open(fname)
    Snu = np.squeeze(hdulist[0].data)
    Snu[np.isnan(Snu)] = 0.0

    if PA is not None:
        Snu = rotate(Snu, PA, reshape=False, mode='constant', cval=0.0)

    x = np.arange(Snu.shape[0], dtype=float) * abs(pixel_size_x)  # in mas
    y = np.arange(Snu.shape[1], dtype=float) * abs(pixel_size_y)  # in mas

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
