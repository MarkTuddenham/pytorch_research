"""Saving data & artefacts."""

from pathlib import Path
import pickle
import csv

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union


class Persistence:
    """Class for persisting many results and resource to a common location."""

    save_path: str = "./res"

    def __init__(self, name: str) -> None:
        """Init the persistence obj."""
        self.folder_name: str = name
        self.csvs: Dict[str, Dict[str, Any]] = {}

        # self.unique_params = {}  # make a new dir if these are diff
        # self.shared_params = {}  # store these in a file and overwrite if changes

        Path(self._get_path()).mkdir(parents=True, exist_ok=True)

    def _get_path(self, name: str = '') -> str:
        """Get the path of the common location with an optional file name."""
        def term(p: str):
            return p if p[-1] == '/' else p + '/'
        return term(Persistence.save_path) + term(self.folder_name) + name

    def save_obj(self, data: Any, name: str) -> None:
        """Save an arbitrary binary object."""
        with open(self._get_path(name + '.pkl'), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name: str) -> Any:
        """Load an arbitrary binary object."""
        with open(self._get_path(name + '.pkl'), 'rb') as f:
            return pickle.load(f)

    def create_csv(self, name: str, headers: Union[str, Sequence[str]]) -> None:
        """Create a csv file with headers."""
        if isinstance(headers, str):
            data = headers
            dialect = csv.Sniffer().sniff(data)
            sep = dialect.delimiter
            count = len(headers.split(sep))
        else:
            sep = ','
            count = len(headers)
            data = f'{sep} '.join(headers)

        self.csvs[name] = {'count': count, 'sep': sep}

        with open(self._get_path(name + '.csv'), 'w') as f:
            f.write(data + '\n')

    def append_csv(self, name: str, data: Sequence[Any]) -> None:
        """Append a line of data to a csv file."""
        if self.csvs[name]['count'] != len(data):
            raise ValueError('Incorrect number of data; expected ' +
                             f'{self.csvs[name]["count"]}, encountered {len(data)}.')

        with open(self._get_path(name + '.csv'), 'a') as f:
            f.write(f'{self.csvs[name]["sep"]} '.join(str(d) for d in data) + '\n')

    def savefig(self, fig: 'matplotlib.figure', name: str, ext: str = 'svg', **mpl_kwargs) -> None:
        """Save a matplotlib figure."""
        # TODO do we need to check path exists before saving
        fig.savefig(self._get_path(name + '.' + ext), **mpl_kwargs)
