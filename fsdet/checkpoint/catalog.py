"""FS-Det Model Handler."""

from detectron2.utils.file_io import PathHandler, PathManager


class FsDetHandler(PathHandler):
    """
    Resolve anything that's in FsDet model zoo.
    """

    PREFIX = "fsdet://"
    URL_PREFIX = "http://dl.yf.io/fs-det/models/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.URL_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)

'''
Finally, the FsDetHandler is registered with the PathManager using PathManager.register_handler(FsDetHandler()). 
This allows the PathManager to use the FsDetHandler for resolving paths and opening files with the fsdet:// prefix.

With this custom handler, you can now use fsdet:// paths to refer to models in the FsDet model zoo, 
and the PathManager will resolve them correctly.
'''

PathManager.register_handler(FsDetHandler())
