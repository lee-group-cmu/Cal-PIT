"""calpit imports and runs its core without the optional dependencies."""

import subprocess
import sys
import textwrap


def test_import_without_optional_dependencies() -> None:
    script = textwrap.dedent(
        """
        import sys
        import warnings

        for name in ("splinebasis", "h5py", "matplotlib"):
            sys.modules[name] = None  # Makes `import name` raise ImportError.
        warnings.simplefilter("error")

        import numpy as np

        import calpit
        from calpit import datasets, metrics, nn, utils

        assert calpit.CalPit is not None
        for build, extra in [
            (lambda: nn.IsplineNN(3), "spline"),
            (lambda: datasets.PhotometryDataset("data.hdf5"), "hdf5"),
            (lambda: utils.plot_pit(np.linspace(0, 1, 10), 0.95), "plot"),
        ]:
            try:
                build()
            except ImportError as error:
                assert f"calpit[{extra}]" in str(error), error
            else:
                raise AssertionError(f"no ImportError for the {extra} extra")
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
