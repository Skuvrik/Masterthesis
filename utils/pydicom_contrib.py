import gc
import os
import time
import sys

import pydicom
from pydicom.sequence import Sequence

# Try importing numpy
try:
    import numpy as np
    have_numpy = True
except ImportError:
    np = None  # NOQA
    have_numpy = False


# Helper functions and classes
class ProgressBar(object):
    """ To print progress to the screen.
    """

    def __init__(self, char='-', length=20):
        self.char = char
        self.length = length
        self.progress = 0.0
        self.nbits = 0
        self.what = ''

    def Start(self, what=''):
        """ Start(what='')
        Start the progress bar, displaying the given text first.
        Make sure not to print anything untill after calling
        Finish(). Messages can be printed while displaying
        progess by using printMessage().
        """
        self.what = what
        self.progress = 0.0
        self.nbits = 0
        sys.stdout.write(what + " [")

    def Stop(self, message=""):
        """ Stop the progress bar where it is now.
        Optionally print a message behind it."""
        delta = int(self.length - self.nbits)
        sys.stdout.write(" " * delta + "] " + message + "\n")

    def Finish(self, message=""):
        """ Finish the progress bar, setting it to 100% if it
        was not already. Optionally print a message behind the bar.
        """
        delta = int(self.length - self.nbits)
        sys.stdout.write(self.char * delta + "] " + message + "\n")

    def Update(self, newProgress):
        """ Update progress. Progress is given as a number
        between 0 and 1.
        """
        self.progress = newProgress
        required = self.length * (newProgress)
        delta = int(required - self.nbits)
        if delta > 0:
            sys.stdout.write(self.char * delta)
            self.nbits += delta

    def PrintMessage(self, message):
        """ Print a message (for example a warning).
        The message is printed behind the progress bar,
        and a new bar is started.
        """
        self.Stop(message)
        self.Start(self.what)


def _dummyProgressCallback(progress):
    """ A callback to indicate progress that does nothing. """
    pass


_progressBar = ProgressBar()


def _progressCallback(progress):
    """ The default callback for displaying progress. """
    if isinstance(progress, str):
        _progressBar.Start(progress)
        _progressBar._t0 = time.time()
    elif progress is None:
        dt = time.time() - _progressBar._t0
        _progressBar.Finish(f'{dt:2.2f} seconds')
    else:
        _progressBar.Update(progress)


def _listFiles(files, path):
    """List all files in the directory, recursively. """

    for item in os.listdir(path):
        item = os.path.join(path, item)
        if os.path.isdir(item):
            _listFiles(files, item)
        else:
            files.append(item)


def _splitSerieIfRequired(serie, series):
    """ _splitSerieIfRequired(serie, series)
    Split the serie in multiple series if this is required.
    The choice is based on examing the image position relative to
    the previous image. If it differs too much, it is assumed
    that there is a new dataset. This can happen for example in
    unspitted gated CT data.
    """

    # Sort the original list and get local name
    serie._sort()
    L = serie._datasets

    # Init previous slice
    ds1 = L[0]

    # Check whether we can do this
    if "ImagePositionPatient" not in ds1:
        return

    # Initialize a list of new lists
    L2 = [[ds1]]

    # Init slice distance estimate
    distance = 0

    for index in range(1, len(L)):

        # Get current slice
        ds2 = L[index]

        # Get positions
        pos1 = float(ds1.ImagePositionPatient[2])
        pos2 = float(ds2.ImagePositionPatient[2])

        # Get distances
        newDist = abs(pos1 - pos2)
        # deltaDist = abs(firstPos-pos2)

        # If the distance deviates more than 2x from what we've seen,
        # we can agree it's a new dataset.
        if distance and newDist > 2.1 * distance:
            L2.append([])
            distance = 0
        else:
            # Test missing file
            if distance and newDist > 1.5 * distance:
                print(f'Warning: missing file after "{ds1.filename}"')
            distance = newDist

        # Add to last list
        L2[-1].append(ds2)

        # Store previous
        ds1 = ds2

    # Split if we should
    if len(L2) > 1:

        # At what position are we now?
        i = series.index(serie)

        # Create new series
        series2insert = []
        for L in L2:
            newSerie = DicomSeries(serie.suid, serie._showProgress)
            newSerie._datasets = Sequence(L)
            series2insert.append(newSerie)

        # Insert series and remove self
        for newSerie in reversed(series2insert):
            series.insert(i, newSerie)
        series.remove(serie)


def _getPixelDataFromDataset(ds):
    """ Get the pixel data from the given dataset. If the data
    was deferred, make it deferred again, so that memory is
    preserved. Also applies RescaleSlope and RescaleIntercept
    if available. """

    # Get original element
    el = ds['PixelData']

    # Get data
    data = ds.pixel_array

    # Remove data (mark as deferred)
    ds['PixelData'] = el
    del ds._pixel_array

    # Obtain slope and offset
    slope = 1
    offset = 0
    needFloats = False
    needApplySlopeOffset = False
    if 'RescaleSlope' in ds:
        needApplySlopeOffset = True
        slope = ds.RescaleSlope
    if 'RescaleIntercept' in ds:
        needApplySlopeOffset = True
        offset = ds.RescaleIntercept
    if int(slope) != slope or int(offset) != offset:
        needFloats = True
    if not needFloats:
        slope, offset = int(slope), int(offset)

    # Apply slope and offset
    if needApplySlopeOffset:

        # Maybe we need to change the datatype?
        if data.dtype in [np.float32, np.float64]:
            pass
        elif needFloats:
            data = data.astype(np.float32)
        else:
            # Determine required range
            minReq, maxReq = data.min(), data.max()
            minReq = min(
                [minReq, minReq * slope + offset, maxReq * slope + offset])
            maxReq = max(
                [maxReq, minReq * slope + offset, maxReq * slope + offset])

            # Determine required datatype from that
            dtype = None
            if minReq < 0:
                # Signed integer type
                maxReq = max([-minReq, maxReq])
                if maxReq < 2 ** 7:
                    dtype = np.int8
                elif maxReq < 2 ** 15:
                    dtype = np.int16
                elif maxReq < 2 ** 31:
                    dtype = np.int32
                else:
                    dtype = np.float32
            else:
                # Unsigned integer type
                if maxReq < 2 ** 8:
                    dtype = np.uint8
                elif maxReq < 2 ** 16:
                    dtype = np.uint16
                elif maxReq < 2 ** 32:
                    dtype = np.uint32
                else:
                    dtype = np.float32

            # Change datatype
            if dtype != data.dtype:
                data = data.astype(dtype)

        # Apply slope and offset
        data *= slope
        data += offset

    # Done
    return data


# The public functions and classes

def find_shape(dataset):
    """Find the expected shape of `dataset.pixel_array` without reading the pixel data.
    The returned shape is a tuple"""
    shape = dataset.Rows, dataset.Columns
    frames = dataset.get('NumberOfFrames', 1) or 1
    if frames > 1:
        shape = (frames,) + shape
    samples = dataset.SamplesPerPixel
    if samples > 1:
        conf = dataset.PlanarConfiguration
        if conf == 0:
            shape += (samples,)
        elif conf == 1:
            shape = (samples,) + shape
        else:
            raise ValueError(f"Invalid Planar Configuration: '{conf}'")
    return shape


def read_files(path, showProgress=False, readPixelData=False, force=False):
    """ read_files(path, showProgress=False, readPixelData=False)
    Reads dicom files and returns a list of DicomSeries objects, which
    contain information about the data, and can be used to load the
    image or volume data.
    The parameter "path" can also be a list of files or directories.
    If the callable "showProgress" is given, it is called with a single
    argument to indicate the progress. The argument is a string when a
    progress is started (indicating what is processed). A float indicates
    progress updates. The paremeter is None when the progress is finished.
    When "showProgress" is True, a default callback is used that writes
    to stdout. By default, no progress is shown.
    if readPixelData is True, the pixel data of all series is read. By
    default the loading of pixeldata is deferred until it is requested
    using the DicomSeries.get_pixel_array() method. In general, both
    methods should be equally fast.
    """

    # Init list of files
    files = []

    # Obtain data from the given path
    if isinstance(path, str):
        # Make dir nice
        basedir = os.path.abspath(path)
        # Check whether it exists
        if not os.path.isdir(basedir):
            raise ValueError('The given path is not a valid directory.')
        # Find files recursively
        _listFiles(files, basedir)

    elif isinstance(path, (tuple, list)):
        # Iterate over all elements, which can be files or directories
        for p in path:
            if os.path.isdir(p):
                _listFiles(files, os.path.abspath(p))
            elif os.path.isfile(p):
                files.append(p)
            else:
                print(f"Warning, the path '{p}' is not valid.")
    else:
        raise ValueError('The path argument must be a string or list.')

    # Set default progress callback?
    if showProgress is True:
        showProgress = _progressCallback
    if not hasattr(showProgress, '__call__'):
        showProgress = _dummyProgressCallback

    # Set defer size
    deferSize = 16383  # 128**2-1
    if readPixelData:
        deferSize = None

    # Gather file data and put in DicomSeries
    series = {}
    count = 0
    showProgress('Loading series information:')
    for filename in files:

        # Skip DICOMDIR files
        if filename.count("DICOMDIR"):
            continue

        # Try loading dicom ...
        try:
            dcm = pydicom.dcmread(filename, deferSize, force=force)
        except pydicom.filereader.InvalidDicomError:
            continue  # skip non-dicom file
        except Exception as why:
            if showProgress is _progressCallback:
                _progressBar.PrintMessage(str(why))
            else:
                print('Warning:', why)
            continue

        # Get SUID and register the file with an existing or new series object
        try:
            suid = dcm.SeriesInstanceUID
        except AttributeError:
            continue  # some other kind of dicom file
        if suid not in series:
            series[suid] = DicomSeries(suid, showProgress)
        series[suid]._append(dcm)

        # Show progress (note that we always start with a 0.0)
        showProgress(float(count) / len(files))
        count += 1

    # Finish progress
    showProgress(None)

    # Make a list and sort, so that the order is deterministic
    series = list(series.values())
    series.sort(key=lambda x: x.suid)

    # Split series if necessary
    for serie in reversed([serie for serie in series]):
        _splitSerieIfRequired(serie, series)

    # Finish all series
    showProgress('Analysing series')
    series_ = []
    for i in range(len(series)):
        try:
            series[i]._finish()
            series_.append(series[i])
        except Exception:
            pass  # Skip serie (probably report-like file without pixels)
        showProgress(float(i + 1) / len(series))
    showProgress(None)

    return series_


class DicomSeries(object):
    """ DicomSeries
    This class represents a series of dicom files that belong together.
    If these are multiple files, they represent the slices of a volume
    (like for CT or MRI). The actual volume can be obtained using loadData().
    Information about the data can be obtained using the info attribute.
    """

    # To create a DicomSeries object, start by making an instance and
    # append files using the "_append" method. When all files are
    # added, call "_sort" to sort the files, and then "_finish" to evaluate
    # the data, perform some checks, and set the shape and spacing
    # attributes of the instance.

    def __init__(self, suid, showProgress):
        # Init dataset list and the callback
        self._datasets = Sequence()
        self._showProgress = showProgress

        # Init props
        self._suid = suid
        self._info = None
        self._shape = None
        self._spacing = None

    @property
    def suid(self):
        """ The Series Instance UID. """
        return self._suid

    @property
    def shape(self):
        """ The shape of the data (nz, ny, nx).
        If None, the series contains a single dicom file. """
        return self._shape

    @property
    def spacing(self):
        """ The spacing (voxel distances) of the data (dz, dy, dx).
        If None, the series contains a single dicom file. """
        return self._spacing

    @property
    def info(self):
        """ A DataSet instance containing the information as present in the
        first dicomfile of this series. """
        return self._info

    @property
    def description(self):
        """ A description of the dicom series. Used fields are
        PatientName, shape of the data, SeriesDescription,
        and ImageComments.
        """

        info = self.info

        # If no info available, return simple description
        if info is None:
            return "DicomSeries containing %i images" % len(self._datasets)

        fields = []

        # Give patient name
        if 'PatientName' in info:
            fields.append("" + str(info.PatientName))

        # Also add dimensions
        if self.shape:
            tmp = [str(d) for d in self.shape]
            fields.append('x'.join(tmp))

        # Try adding more fields
        if 'SeriesDescription' in info:
            fields.append("'" + info.SeriesDescription + "'")
        if 'ImageComments' in info:
            fields.append("'" + info.ImageComments + "'")

        # Combine
        return ' '.join(fields)

    def __repr__(self):
        adr = hex(id(self)).upper()
        data_len = len(self._datasets)
        return "<DicomSeries with %i images at %s>" % (data_len, adr)

    def get_pixel_array(self):
        """ get_pixel_array()
        Get (load) the data that this DicomSeries represents, and return
        it as a numpy array. If this serie contains multiple images, the
        resulting array is 3D, otherwise it's 2D.
        If RescaleSlope and RescaleIntercept are present in the dicom info,
        the data is rescaled using these parameters. The data type is chosen
        depending on the range of the (rescaled) data.
        """

        # Can we do this?
        if not have_numpy:
            msg = "The Numpy package is required to use get_pixel_array.\n"
            raise ImportError(msg)

        # It's easy if no file or if just a single file
        if len(self._datasets) == 0:
            raise ValueError('Serie does not contain any files.')
        elif len(self._datasets) == 1:
            ds = self._datasets[0]
            slice = _getPixelDataFromDataset(ds)
            return slice

        # Check info
        if self.info is None:
            raise RuntimeError("Cannot return volume if series not finished.")

        # Set callback to update progress
        showProgress = self._showProgress

        # Init data (using what the dicom packaged produces as a reference)
        ds = self._datasets[0]
        slice = _getPixelDataFromDataset(ds)
        vol = np.zeros(self.shape, dtype=slice.dtype)
        vol[0] = slice

        # Fill volume
        showProgress('Loading data:')
        ll = self.shape[0]
        for z in range(1, ll):
            ds = self._datasets[z]
            vol[z] = _getPixelDataFromDataset(ds)
            showProgress(float(z) / ll)

        # Finish
        showProgress(None)

        # Done
        gc.collect()
        return vol

    def _append(self, dcm):
        """ _append(dcm)
        Append a dicomfile (as a pydicom.dataset.FileDataset) to the series.
        """
        self._datasets.append(dcm)

    def _sort(self):
        """ sort()
        Sort the datasets by instance number.
        """
        self._datasets.sort(key=lambda k: k.InstanceNumber)

    def _finish(self):
        """ _finish()
        Evaluate the series of dicom files. Together they should make up
        a volumetric dataset. This means the files should meet certain
        conditions. Also some additional information has to be calculated,
        such as the distance between the slices. This method sets the
        attributes for "shape", "spacing" and "info".
        This method checks:
          * that there are no missing files
          * that the dimensions of all images match
          * that the pixel spacing of all images match
        """

        # The datasets list should be sorted by instance number
        L = self._datasets
        if len(L) == 0:
            return
        elif len(L) < 2:
            # Set attributes
            ds = self._datasets[0]
            self._info = self._datasets[0]
            self._shape = find_shape(ds)
            self._spacing = ds.PixelSpacing
            return

        # Get previous
        ds1 = L[0]

        # Init measures to calculate average of
        distance_sum = 0.0

        # Init measures to check (these are in 2D)
        dimensions = find_shape(ds1)

        # row, column
        spacing = ds1.PixelSpacing

        for index in range(len(L)):
            # The first round ds1 and ds2 will be the same, for the
            # distance calculation this does not matter

            # Get current
            ds2 = L[index]

            # Get positions
            pos1 = float(ds1.ImagePositionPatient[2])
            pos2 = float(ds2.ImagePositionPatient[2])

            # Update distance_sum to calculate distance later
            # TODO: use ImageOrientationPatient's normal to calculate this distance
            distance_sum += abs(pos1 - pos2)

            # Test measures
            dimensions2 = find_shape(ds2)
            spacing2 = ds2.PixelSpacing
            if dimensions != dimensions2:
                # We cannot produce a volume if the dimensions match
                raise ValueError('Dimensions of slices does not match.')
            if spacing != spacing2:
                # We can still produce a volume, but we should notify the user
                msg = 'Warning: spacing does not match.'
                if self._showProgress is _progressCallback:
                    _progressBar.PrintMessage(msg)
                else:
                    print(msg)
            # Store previous
            ds1 = ds2

        # Create new dataset by making a deep copy of the first
        info = pydicom.dataset.Dataset()
        firstDs = self._datasets[0]
        for key in firstDs.keys():
            if key != 'PixelData':
                el = firstDs[key]
                info.add_new(el.tag, el.VR, el.value)

        # Finish calculating average distance
        # (Note that there are len(L)-1 distances)
        distance_mean = distance_sum / (len(L) - 1)

        # Store information that is specific for the series
        self._shape = (len(L),) + dimensions
        spacing.insert(0, distance_mean)
        self._spacing = spacing

        # Store
        self._info = info