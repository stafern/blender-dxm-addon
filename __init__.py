bl_info = {
    "name": "Earth Defense Force 2017 Formats",
    "author": "stafern / Smileynator / BlueAmulet",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "File > Import-Export",
    "description": "Import DXM from Earth Defense Force 2017",
    "warning": "",
    "support": 'COMMUNITY',
    "category": "Import",
}


if "bpy" in locals():
    import importlib
    if "import_dxm" in locals():
        importlib.reload(import_dxm)

import bpy
from bpy.props import (
        StringProperty,
        )
from bpy_extras.io_utils import (
        ImportHelper,
        ExportHelper,
        )


class ImportDXM(bpy.types.Operator, ImportHelper):
    """Load a DXM file"""
    bl_idname = "import_scene.dxm"
    bl_label = "Import DXM"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".dxm"
    filter_glob: StringProperty(default="*.dxm", options={'HIDDEN'})

    def execute(self, context):
        from . import import_dxm

        keywords = self.as_keywords(ignore=())

        return import_dxm.load(self, context, **keywords)

    def draw(self, context):
        pass

def menu_func_import(self, context):
    self.layout.operator(ImportDXM.bl_idname, text="Earth Defense Force 2017 Model (.dxm)")

classes = (
    ImportDXM
)


def register():
    #for cls in classes:
    bpy.utils.register_class(classes)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

    #for cls in classes:
    bpy.utils.unregister_class(classes)


if __name__ == "__main__":
    register()
