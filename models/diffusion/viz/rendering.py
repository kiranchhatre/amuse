
import os
import bpy
import sys
import json
from math import radians
from pathlib import Path

class BlenderVisualizer():

    def __init__(self, args):
        
        males =   ["wayne", "scott", "solomon", "lawrence", "stewart", \
                   "nidal", "zhao", "lu", "zhang", "carlos", \
                   "jorge", "itoi", "daiki", "jaime", "li"]
        females = ["carla", "sophie", "catherine", "miranda", "kieks", \
                   "ayana", "luqi", "hailing", "kexin", "goto", \
                   "reamey", "yingqing", "tiffnay", "hanieh", "katya"]
        
        self.args = dotdict(args)
        
        # modality based scene setup
        if self.args.modality == "face":
            # params
            self.face_location = (0, 0, 0)
            self.face_scale = 0.15
            self.cam_scale = 20
            self.cam_location = (70, -150, 100)
            # facemesh
            self.mesh_name = "Neutral"
        elif self.args.modality == "both":
            # params
            self.body_location = (-2, 0, 0)
            self.body_scale = 1
            self.face_location = (6, 0, -3)
            self.face_scale = 0.04
            self.cam_scale = 20
            self.cam_location = (70, -150, 100)
            # facemesh
            self.mesh_name = "Neutral"
        elif self.args.modality == "bvh":
            # params
            self.body_location = (0, 0, 0)
            self.body_scale = 1
            self.cam_scale = 18
            self.cam_location = (0, -150, 0)
        self.sun_location = self.cam_location
        
        # body imports
        if self.args.modality in ["both", "bvh"]:
            self.male_mesh = "Base_male"
            self.male_mesh1 = "Base_male:Body"
            self.male_mesh2 = "Base_male:High-poly"
            self.female_mesh = "Base_female"
            self.female_mesh1 = "Base_female:Body"
            self.female_mesh2 = "Base_female:High-poly"
            self.bvh_name = Path(self.args.bvh).stem
            self.person = self.bvh_name.split("_")[0]
            self.clothed_tag = True if self.args.clothed_arg == "clothed" else False
            if not self.clothed_tag:
                self.gender = "Base_male" if self.person in males else "Base_female" if self.person in females else int(f"Unknown person: {self.person}")
            else: self.gender = "Male" if self.person in males else "Female" if self.person in females else int(f"Unknown person: {self.person}")
            self.log_name = self.bvh_name
        else: self.log_name = Path(self.args.json).stem
            
    def render_clip(self):
        
        # log out redirection
        viz_dir = Path(self.args.video).parent
        logfile = viz_dir / f"{self.log_name}_blender.log"
        open(logfile, 'a').close()
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)
        
        self._deleteAllObjects()
        
        # set up scene
        scene = bpy.data.scenes['Scene']
        scene.render.fps = self.args.fps
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        
        if self.args.modality in ["face", "both"]:
            # face-mesh, bvh imports
            bpy.ops.import_scene.fbx(filepath=self.args.fbx, global_scale=self.face_scale)
            # bpy.ops.import_anim.bvh(filepath=self.args.bvh)
            face = bpy.data.objects[self.mesh_name]
            # bvh = bpy.data.objects[self.bvh_name]
            face.location = self.face_location
            
            # face wireframe material
            face.data.materials.clear()
            mat = bpy.data.materials.new(name='Material')
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (.5, .5, .5, 1)
            face.data.materials.append(mat)
         
        # delete additional objects                                                         # obsolete
        # obj_to_keep = [self.mesh_name, self.bvh_name, self.male_mesh, self.female_mesh]
        # for object in bpy.data.objects:
        #     if object.name not in obj_to_keep:
        #         object.select_set(True)
        #     else: object.select_set(False)
        # bpy.ops.object.delete()
        
        # camera
        bpy.ops.object.camera_add(location=self.cam_location)
        camera = bpy.data.objects['Camera']
        camera.rotation_euler = (radians(90), 0, 0)
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = self.cam_scale
        
        # light
        bpy.ops.object.light_add(type='SUN', location=self.sun_location)
        light = bpy.data.objects['Sun']
        light.rotation_euler = (radians(90), 0, 0)
        light.data.energy = 1

        if self.args.modality in ["face", "both"]:
            # face anim
            with open(self.args.json, 'r') as f:
                df = json.load(f)
                num_frames = self._assign_facial_animation(df, face)

        if self.args.modality in ["bvh", "both"]:
            # body anim
            if not self.clothed_tag:
                bodymesh = self.args.mesh_male if self.gender == "Base_male" else self.args.mesh_female
            else: bodymesh = self.args.mesh_male if self.gender == "Male" else self.args.mesh_female
            bpy.ops.import_scene.makehuman_mhx2(filepath=bodymesh)
            body = bpy.data.objects[self.gender]
            body.location = self.body_location
            # scene.McpSourceRig = "BEAT"                                       # TODO: fix this        
            bpy.ops.mcp.load_and_retarget(filepath=self.args.bvh,                     
                                        startFrame = 1, endFrame = self.args.endFrame,  
                                        scale = self.body_scale)        
        
        # eevee settings
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_samples = 128
        bpy.context.scene.camera = camera
        if self.args.modality in ["face", "both"]: self._render_setting(num_frames)
        else: self._render_setting(self.args.endFrame)
        bpy.ops.render.render(animation=True)
        self._deleteAllObjects()
        
        # disable log out redirection
        os.close(fd)
        os.dup(old)
        os.close(old)

    def _assign_facial_animation(self, df, face):
        for frame in df['frames']:
            for j, k in enumerate(frame['weights']):
                key_shape_name = df['names'][j]
                if key_shape_name in face.data.shape_keys.key_blocks:
                    key_shape = face.data.shape_keys.key_blocks[key_shape_name]
                    key_shape.value = k
                    current_frame = self.args.fps * frame['time']
                    key_shape.keyframe_insert("value", frame=current_frame)
        return current_frame

    def _render_setting(self, num_frames):
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area.spaces[0].region_3d.view_perspective = 'CAMERA'
        bpy.context.scene.render.filepath = self.args.video
        bpy.context.scene.render.fps = self.args.fps
        bpy.data.scenes[0].frame_start = 1
        bpy.data.scenes[0].frame_end = int(num_frames + 1)

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
    assert len(sys.argv) == 15, "[Blender] Invalid number of arguments"
    try:
        if bpy.app.background:
            args = sys.argv[sys.argv.index("--") + 1:]
            args = [arg for arg in args if arg != "--"]
            dict_args = {
                "bvh": args[0],             # combined bvh file
                "json": args[1],            # json file
                "endFrame": int(args[2]),   # endframes
                "video": args[3],           # output video file
                "mesh_male": args[4],       # male mhx2 file
                "mesh_female": args[5],     # female mhx2 file
                "fbx": args[6],             # face fbx file
                "fps": int(args[7]),        # fps
                "modality": args[8],        # modality kind to render
                "clothed_arg": args[9],     # clothed or not
                "wav": "",                  # wav file
                "mtl": "",                  # face mtl file
                "obj": "",                  # face obj file
                "abc": ""                   # face alembic file  
            }    
        blender = BlenderVisualizer(dict_args)
        blender.render_clip()
    
    except SystemExit as ex:
        if ex.code is None:
            exit_status = 0
        else: exit_status = ex.code
        print(f"[BLENDER-Render] Exit with status {exit_status}")
        if bpy.app.background: sys.exit(exit_status)