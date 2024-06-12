import bpy
import sys
import os
import fnmatch
from math import radians
from pathlib import Path

class SmplExporter():
    
    def __init__(self, args):
        
        males =   ["wayne", "scott", "solomon", "lawrence", "stewart", \
                   "nidal", "zhao", "lu", "zhang", "carlos", \
                   "jorge", "itoi", "daiki", "jaime", "li"]
        females = ["carla", "sophie", "catherine", "miranda", "kieks", \
                   "ayana", "luqi", "hailing", "kexin", "goto", \
                   "reamey", "yingqing", "tiffnay", "hanieh", "katya"]
        
        self.args = dotdict(args)
        self.file = self.args.bvh
        self.bvh_name = Path(self.args.bvh).stem
        self.extract_path = self.args.extract_path
        if self.extract_path: person = self.bvh_name.split("_")[1]
        else: person = self.bvh_name.split("_")[0]
        if "bmap_preset" in self.args: self.bmap_preset = self.args.bmap_preset
        else: self.bmap_preset = "beat2smpl"
        
        if person in males:
            self.gender = "male"
            self.weight = 48
        elif person in females:
            self.gender = "female"
            self.weight = 43
        else: int(f"Unknown person: {self.person}")
        
    def purge_orphans(self):
        if bpy.app.version >= (3, 0, 0):
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        else:
            result = bpy.ops.outliner.orphans_purge()
            if result.pop() != "CANCELLED":
                self.purge_orphans()
    
    def export_smpl_fbx(self):
        
        # log out redirection
        dirname = os.path.dirname(self.file)
        basename_without_ext = os.path.basename(self.file).split(".")[0]
        logfile = os.path.join(dirname, basename_without_ext + "_smpl_blender.log")
        open(logfile, 'a').close() 
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)
        
        self._deleteAllObjects()
        
        assert self.file.endswith(".bvh"), "[Blender] Invalid file format"
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        gender = self.gender
        context = bpy.context
        
        gender_obj = "SMPLX-" + gender
        gender_mesh = "SMPLX-mesh-" + gender
        area = next(area for area in context.screen.areas if area.type == 'VIEW_3D')
        scene = bpy.data.scenes['Scene']
        bpy.ops.import_anim.bvh(filepath=self.file)
        bpy.context.scene.source_rig = "Camera"
        bpy.data.window_managers["WinMan"].smplx_tool.smplx_gender = gender
        bpy.ops.scene.smplx_add_gender()
        # bpy.data.window_managers["WinMan"].smplx_tool.smplx_height = 1.7
        # bpy.data.window_managers["WinMan"].smplx_tool.smplx_weight = weight
        # bpy.ops.object.smplx_measurements_to_shape()
        
        # Manual procedure >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        bone_list = [
            "Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "Spine1", "LeftFoot", "RightFoot",
            "RightForeFoot", "RightToeBase", "LeftForeFoot", "LeftToeBase", "RightToeBaseEnd", "LeftToeBaseEnd"
        ]
        for objs in bpy.data.objects:
            if objs.type == 'ARMATURE': armature = objs.data
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in armature.bones:
            if bone.name in bone_list: armature.edit_bones.remove(bone)
        bpy.ops.object.mode_set(mode='OBJECT')  
        bvhrig = bpy.data.objects[self.bvh_name]
        bvhrig.rotation_euler = (radians(180), 0, 0)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        scene.source_rig = self.bvh_name
        scene.target_rig = gender_obj
        bpy.ops.arp.auto_scale()
        bpy.ops.arp.build_bones_list()
        bpy.ops.arp.import_config_preset(preset_name=self.bmap_preset)
        with open(self.file, "r") as pose_data:
            frame_len = 0
            for line in pose_data.readlines():
                frame_len += 1
            frame_len -= 433
        bpy.ops.arp.retarget('EXEC_DEFAULT', frame_start=0, frame_end=frame_len)
        
        scene.frame_start = 0
        scene.frame_end = frame_len
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[gender_mesh].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[gender_mesh]
        
        if self.extract_path: fbx_name = os.path.join(self.extract_path, basename_without_ext + ".fbx")
        else: fbx_name = os.path.join(dirname, basename_without_ext + ".fbx")
        bpy.ops.object.smplx_export_fbx(filepath=fbx_name)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        self.purge_orphans()
        
        # disable log out redirection
        os.close(fd)
        os.dup(old)
        os.close(old)
    
    def _get_context_window(self):
        for window in bpy.context.window_manager.windows:
            if window:
                return window
        return None
    
    def _deleteAllObjects(self):
        deleteListObjects = ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'HAIR', 'POINTCLOUD', 'VOLUME', 'GPENCIL',
                             'ARMATURE', 'LATTICE', 'EMPTY', 'LIGHT', 'LIGHT_PROBE', 'CAMERA', 'SPEAKER']
        for o in bpy.context.scene.objects:
            for i in deleteListObjects:
                if o.type == i: o.select_set(False)
                else: o.select_set(True)
        bpy.ops.object.delete()
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
   
if __name__ == "__main__":
    assert len(sys.argv)  in [7, 8], "[Blender] Invalid number of arguments"
    try:
        if bpy.app.background:
            args = sys.argv[sys.argv.index("--") + 1:]
            args = [arg for arg in args if arg != "--"]
            dict_args = {
                "bvh": args[0],             # combined bvh file
                "extract_path": args[1],    # path to extract the smpl fbx file
            }  
            if len(sys.argv) == 8: dict_args["bmap_preset"] = args[2]  
        blender = SmplExporter(dict_args)
        blender.export_smpl_fbx()
    
    except SystemExit as ex:
        if ex.code is None:
            exit_status = 0
        else: exit_status = ex.code
        print(f"[BLENDER-Render] Exit with status {exit_status}")
        if bpy.app.background: sys.exit(exit_status)