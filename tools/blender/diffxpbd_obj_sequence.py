bl_info = {
    "name": "OBJ Sequence (.obj)",
    "author": "DiffXPBD",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > OBJ Sequence (.obj)",
    "category": "Import-Export",
}

import re
from pathlib import Path

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper


def find_sequence(sample):
    match = re.fullmatch(r"(.+)_([0-9]+)", sample.stem)
    if sample.suffix.lower() != ".obj" or not match or not sample.is_file():
        raise ValueError("Select an existing OBJ named like skirt_42.obj")

    prefix = match[1]
    pattern = re.compile(rf"{re.escape(prefix)}_([0-9]+)\.[oO][bB][jJ]")
    frames = {}
    for path in sample.parent.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            frame = int(match[1])
            if frame in frames:
                raise ValueError(f"Multiple files match frame {frame}: {frames[frame].name}, {path.name}")
            frames[frame] = path
    return prefix, sorted(frames.items())


class IMPORT_SCENE_OT_diffxpbd_obj_sequence(bpy.types.Operator, ImportHelper):
    """Select one numbered OBJ to import its sequence"""

    bl_idname = "import_scene.diffxpbd_obj_sequence"
    bl_label = "OBJ Sequence (.obj)"
    bl_options = {'UNDO'}

    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'})

    def execute(self, context):
        scene = context.scene
        try:
            name, frames = find_sequence(Path(self.filepath))
            if not frames:
                raise ValueError("No matching OBJ files found")
            if frames[-1][0] > scene.bl_rna.properties["frame_end"].hard_max:
                raise ValueError("A timestep exceeds Blender's maximum timeline frame")
        except (OSError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

        collection = bpy.data.collections.new(f"{name} OBJ Sequence")
        scene.collection.children.link(collection)
        imported = []
        try:
            for frame, path in frames:
                result = bpy.ops.wm.obj_import(filepath=str(path), use_split_objects=False)
                if 'FINISHED' not in result:
                    raise RuntimeError(f"Could not import {path.name}")
                objects = tuple(context.selected_objects)
                if not objects:
                    raise RuntimeError(f"No objects imported from {path.name}")
                imported.extend(objects)
                for obj in objects:
                    for source in tuple(obj.users_collection):
                        source.objects.unlink(obj)
                    collection.objects.link(obj)
                    for key_frame, hidden in ((frame - 1, True), (frame, False), (frame + 1, True)):
                        obj.hide_viewport = hidden
                        obj.hide_render = hidden
                        obj.keyframe_insert(data_path="hide_viewport", frame=key_frame)
                        obj.keyframe_insert(data_path="hide_render", frame=key_frame)
                    # Leave the object enabled while the next OBJ import updates selection.
                    obj.hide_viewport = False
        except RuntimeError as error:
            for obj in imported:
                mesh_data = obj.data if obj.type == 'MESH' else None
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh_data is not None and mesh_data.users == 0:
                    bpy.data.meshes.remove(mesh_data)
            bpy.data.collections.remove(collection)
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

        first, last = frames[0][0], frames[-1][0]
        scene.frame_end = max(1, last)
        scene.frame_start = first
        scene.frame_end = last
        scene.frame_set(first)
        self.report({'INFO'}, f"Imported {len(frames)} OBJ frames")
        return {'FINISHED'}


def import_menu(self, context):
    self.layout.operator(IMPORT_SCENE_OT_diffxpbd_obj_sequence.bl_idname)


def register():
    bpy.utils.register_class(IMPORT_SCENE_OT_diffxpbd_obj_sequence)
    bpy.types.TOPBAR_MT_file_import.append(import_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(import_menu)
    bpy.utils.unregister_class(IMPORT_SCENE_OT_diffxpbd_obj_sequence)


if __name__ == "__main__":
    register()
