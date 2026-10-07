"""pytest configuration: add src to path."""
import sys
from pathlib import Path
import pytest

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# These legacy modules mix generated tests with skipif-gated installed datasets.
# Merely having a dataset on disk must never opt a normal pytest run into it.
_DATASET_MODULES = {
    'test_controls.py', 'test_bank.py', 'test_atlas_prep.py',
    'test_check_registration.py', 'test_clearance.py', 'test_grid.py',
    'test_margin_export_sift2.py', 'test_recovery.py', 'test_parcellation.py',
    'test_track.py', 'test_presets.py', 'test_underlay.py', 'test_surface.py',
    'test_volume_seed_pack.py', 'test_serve.py', 'test_serve_fidelity_block.py',
}

def pytest_addoption(parser):
    parser.addoption('--run-case-data', action='store_true', default=False,
                     help='Explicitly run installed dataset integrations; may invoke tracking.')

def pytest_configure(config):
    config.addinivalue_line('markers', 'case_data: explicitly gated installed dataset integration')

def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name in _DATASET_MODULES and list(item.iter_markers('skipif')):
            item.add_marker(pytest.mark.case_data)
    if config.getoption('--run-case-data'):
        return
    omitted=[item for item in items if item.get_closest_marker('case_data')]
    items[:]=[item for item in items if not item.get_closest_marker('case_data')]
    if omitted:
        config.hook.pytest_deselected(items=omitted)
