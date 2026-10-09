"""calpit imports and runs its core without the optional dependencies."""

import subprocess
import sys
import textwrap


def test_import_without_optional_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import warnings

        BLOCKED = {"torch", "lightning", "sklearn", "qp", "splinebasis", "h5py", "matplotlib"}


        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"No module named {name!r}")


        # An import hook rather than sys.modules[name] = None, which SciPy's
        # check for torch tensors would trip over.
        sys.meta_path.insert(0, Blocker())
        warnings.simplefilter("error")

        import numpy as np
        from scipy import stats

        import calpit
        from calpit import qp_io

        y_grid = np.linspace(-5, 5, 101)
        cde = calpit.GridCDE(np.tile(stats.norm.pdf(y_grid), (4, 1)), y_grid)
        assert cde.pit(np.zeros(4)).shape == (4,)

        for build, extra in [
            (lambda: calpit.CalPIT().fit(np.zeros((4, 2)), pit=np.full(4, 0.5)), "torch"),
            (lambda: __import__("calpit.nn"), "torch"),
            (lambda: qp_io.to_qp(cde), "qp"),
            (lambda: calpit.utils.plot_pit(np.linspace(0, 1, 10), 0.95), "plot"),
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


def test_import_does_not_load_torch() -> None:
    script = "import sys, calpit; assert 'torch' not in sys.modules, 'calpit imported torch'"
    subprocess.run([sys.executable, "-c", script], check=True)
