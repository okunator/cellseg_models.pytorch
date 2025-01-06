import re
from pathlib import Path

try:
    import requests

    HAS_REQUESTS = True
except ModuleNotFoundError:
    HAS_REQUESTS = False
from tqdm import tqdm


class Downloader:
    """A simple downloader to download data given an url.

    NOTE: Includes downloading from google drive.

    ALSO NOTE: Does not have any fancy functionality. Just downloading..
    """

    def __init__(self, save_dir: str) -> None:
        """Initialize the downloader."""
        if not HAS_REQUESTS:
            raise ModuleNotFoundError(
                "To use `Downloader`, the requests package is needed. "
                "Install with `pip install requests`."
            )

        self.save_dir = Path(save_dir)
        if not self.save_dir.exists() or not self.save_dir.is_dir():
            raise ValueError(
                "The given `save_dir` does not exist or is not a folder."
                f" {self.save_dir}"
            )

    def gdrive_download(
        self, file_id: str, file_name: str, chunk_size: int = 32768
    ) -> None:
        """Download a file from google drive.

        Parameters
        ----------
            file_id : str
                Google drive file ID.
            file_name : str
                Name of the file to be saved. Typically .zip file.
            chunk_size : int, default=32768
                Chunk size for loading the file in bytes. Here 32^3.

        Example
        -------
            >>> save_dir = "/path/to/save_dir"
            >>> gdrive_id = "123gdriveID"
            >>> downloader = Downloader(save_dir)
            >>> downloader.gdrive_download(gdrive_id, file_name="dd.zip")
        """
        path = self.save_dir / file_name

        url = f"https://drive.google.com/uc?id={file_id}"
        self._download(url, path.as_posix(), gdrive=True, chunk_size=chunk_size)

    def download(self, url: str, chunk_size: int = 32768) -> None:
        """Download a file from url.

        Parameters
        ----------
            url : str
                The url of the file to be downloaded.
            chunk_size : int, default=32768
                Chunk size for loading the file in bytes. Here 32^3.

        Example
        -------
            >>> save_dir = "/path/to/save_dir"
            >>> url = "https://url/to/file/"
            >>> downloader = Downloader(save_dir)
            >>> downloader.download(url)  # download from a url.
        """
        path = self.save_dir / Path(url).name
        self._download(url, path.as_posix(), gdrive=False, chunk_size=chunk_size)

    def _download(self, url: str, out_path: str, gdrive: bool, chunk_size: int) -> None:
        response = self._get_response(url, gdrive=gdrive)
        nbytes = int(response.headers.get("content-length", 0))

        with (
            tqdm(total=nbytes, unit="iB", unit_scale=True) as pbar,
            open(out_path, "wb") as f,
        ):
            for chunk in response.iter_content(chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    def _get_response(self, url: str, gdrive: bool = False) -> requests.Response:
        session = requests.session()
        response = session.get(url, stream=True, verify=True)

        if gdrive:
            url = self._get_url_from_gdrive_confirmation(response.text)
            response = session.get(url, stream=True, verify=True)

        return response

    def _get_url_from_gdrive_confirmation(self, contents: str) -> str:
        """Parse the response text if furher confirmation is needed.

        From:
        ----
        https://github.com/wkentaro/gdown/blob/main/gdown/download.py
        """
        url = ""
        for line in contents.splitlines():
            m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
            if m:
                url = "https://docs.google.com" + m.groups()[0]
                url = url.replace("&amp;", "&")
                break
            m = re.search('id="downloadForm" action="(.+?)"', line)
            if m:
                url = m.groups()[0]
                url = url.replace("&amp;", "&")
                break
            m = re.search('"downloadUrl":"([^"]+)', line)
            if m:
                url = m.groups()[0]
                url = url.replace("\\u003d", "=")
                url = url.replace("\\u0026", "&")
                break
            m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
            if m:
                error = m.groups()[0]
                raise RuntimeError(error)
        if not url:
            raise RuntimeError(
                "Cannot retrieve the public link of the file. "
                "You may need to change the permission to "
                "'Anyone with the link', or have had many accesses."
            )
        return url
