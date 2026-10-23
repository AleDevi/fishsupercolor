"""Export / Import model buttons shared by the Training and Landmarks screens."""
import os

from PySide6.QtWidgets import QFileDialog, QMessageBox

from .. import modelshare


def export_dialog(parent, project, name):
    if not name:
        QMessageBox.information(parent, 'Export model', 'Choose the model to export in "Use in the analysis" first.')
        return
    dest, _ = QFileDialog.getSaveFileName(parent, 'Export model', os.path.join(os.path.expanduser('~'),
                                          f'{project.name}_{name}.zip'), 'Guppy Colour model (*.zip)')
    if not dest:
        return
    try:
        modelshare.export_model(project, name, dest)
        QMessageBox.information(parent, 'Export model', f'Saved {dest}.\nOpen another project and use Import model… '
                                'to use it there, or send the file to a colleague.')
    except Exception as e:
        QMessageBox.warning(parent, 'Export model', str(e))


def import_dialog(parent, project, kind):
    """Returns the new model name, or None."""
    note = ('\n\nLandmark models contain neural-network weights: import only files from people you trust.'
            if kind == 'landmarks' else '')
    src, _ = QFileDialog.getOpenFileName(parent, 'Import model', os.path.expanduser('~'), 'Guppy Colour model (*.zip)')
    if not src:
        return None
    try:
        manifest, _ = modelshare.read_package(src)
        if manifest['kind'] != kind:
            raise ValueError(f"this is a {manifest['kind']} model: import it in the "
                             f"{'Training' if manifest['kind'] == 'colours' else 'Landmarks'} screen")
        if QMessageBox.question(parent, 'Import model', f"Import {manifest.get('name')} from project "
                                f"“{manifest.get('project')}” ({(manifest.get('exported') or '')[:10]})?{note}") \
                != QMessageBox.Yes:
            return None
        name, _ = modelshare.import_model(project, src)
        QMessageBox.information(parent, 'Import model', f'Imported as {name} and put in use.')
        return name
    except Exception as e:
        QMessageBox.warning(parent, 'Import model', f'Not imported: {e}')
        return None
