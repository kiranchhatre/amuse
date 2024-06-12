
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
        
        self.modified_smplx_viz = True if Path(self.args.fbx).suffix == ".npz" else False 
        
        self.fbx_name = Path(self.args.fbx).stem
        self.person = self.fbx_name.split("_")[0]
        self.log_name = self.fbx_name
        self.bmap = self.args.bmap
        self.halfbody = self.args.halfbody
        self.endFrame = self.args.endFrame
        
        if not self.modified_smplx_viz:                     # FBX based
            if self.bmap not in ["beat2smpl"]:              # MoGlow based
                self.cam_location = (0, -4, -0.35)
                self.headlight_location = (0, 0, 1)
                if self.person in males:
                    self.floor_location = (0, 0, -1.29028)
                    self.mesh_name = "SMPLX-mesh-male"
                elif self.person in females:
                    self.floor_location = (0, 0, -1.29345)
                    self.mesh_name = "SMPLX-mesh-female"
                else: int(f"Unknown person: {self.person}")
            else:                                           # CaMN based (not working, manual setup required)
                self.cam_location = (0, -4, -0.35) # (0, -5.49, 0.49)
                self.headlight_location = (0, 0, 1) # (0, -0.91, 2.3)
                if self.person in males:
                    self.floor_location = (0, 0, -1.29028) # (0, 0, -0.54961)
                    self.mesh_name = "SMPLX-mesh-male"
                elif self.person in females:
                    self.floor_location = (0, 0, -1.29345) # (0, 0, -0.54961)
                    self.mesh_name = "SMPLX-mesh-female"
                else: int(f"Unknown person: {self.person}")
                self.mesh_name = "SMPLX-mesh-neutral" # overwriting for CaMN retargetted fbx
        else:                                               # NPZ based
            self.cam_location = (0, -4.7, -0.35)
            if self.person in males:
                self.floor_location = (0, 0, -1.42961)
                self.mesh_name = "SMPLX-mesh-male"
            elif self.person in females:
                self.floor_location = (0, 0, -1.13961)
                self.mesh_name = "SMPLX-mesh-female"
            else: int(f"Unknown person: {self.person}")
    
        if self.halfbody == "True": 
            self.wall_pos = (0, 0.5, 0)
            self.cam_location = (0, -2.5, 0)
        else: self.wall_pos = (0, 1.5, 0)
    
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
         
        # camera
        bpy.ops.object.camera_add(location=self.cam_location)
        camera = bpy.data.objects['Camera']
        camera.rotation_euler = (radians(89), 0, 0)
        camera.data.type = 'PERSP'
        camera.data.lens = 75
        
        # key light
        bpy.ops.object.light_add(type='SPOT', radius=1, align='WORLD', location=(2.8, -3, 4.11), rotation=(radians(40), radians(30), radians(9)))
        bpy.data.objects['Spot'].name = 'Keylight'
        keylight = bpy.data.objects['Keylight']
        keylight.data.energy = 1200
        
        # fill light
        bpy.ops.object.light_add(type='AREA', align='WORLD', location=(-2, -3, 1.37), rotation=(radians(60), 0, radians(-33)))
        bpy.data.objects['Area'].name = 'Filllight'
        filllight = bpy.data.objects['Filllight']
        filllight.data.energy = 40
        
        # light above head
        bpy.ops.object.light_add(type='AREA', align='WORLD', location=self.headlight_location, scale=(1, 1, 1))
        bpy.data.objects['Area'].name = 'Headlight'
        headlight = bpy.data.objects['Headlight']
        headlight.data.shape = 'DISK'
        headlight.data.size = 0.5
        headlight.data.energy = 20
        headlight.data.diffuse_factor = 0.5

        # MOTION
        if not self.modified_smplx_viz:
            bpy.ops.import_scene.fbx(filepath=self.args.fbx, global_scale=1.0) # FBX based
            self.endFrame = bpy.data.actions[0].frame_range[1]
        else: 
            bpy.ops.object.smplx_add_animation(filepath=self.args.fbx, target_framerate=self.args.fps)
            for obj in bpy.data.objects:
                if str(self.fbx_name) in obj.name: armature = obj
            armature.rotation_euler = (radians(90), 0, 0)
        # body_armature = bpy.data.objects["SMPLX-male"]
        body_mesh = bpy.data.objects[self.mesh_name]
        body_mesh.data.materials.clear()
        mat = bpy.data.materials.new(name='Material')
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].subsurface_method = 'BURLEY'
        # Skin
        # mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.436, 0.227, 0.131, 1)
        # mat.node_tree.nodes["Principled BSDF"].inputs[1].default_value = 0.01
        # mat.node_tree.nodes["Principled BSDF"].inputs[2].default_value[0] = 3.67
        # mat.node_tree.nodes["Principled BSDF"].inputs[2].default_value[1] = 1.37
        # mat.node_tree.nodes["Principled BSDF"].inputs[2].default_value[2] = 0.68
        # metallic blue
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.238397, 0.55834, 0.701102, 1)
        body_mesh.data.materials.append(mat)
        
        # floor
        bpy.ops.mesh.primitive_plane_add(size=5, enter_editmode=False, align='WORLD', location=self.floor_location)
        bpy.data.objects["Plane"].name = "Floor"
        floor = bpy.data.objects["Floor"]
        floor.data.materials.clear()
        mat = bpy.data.materials.new(name='Material')
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].subsurface_method = 'BURLEY'
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.730461, 0.47932, 0.242281, 1)
        floor.data.materials.append(mat)
        
        # wall
        bpy.ops.mesh.primitive_plane_add(size=5, enter_editmode=False, align='WORLD', location=self.wall_pos, rotation=(radians(90), 0, 0))
        bpy.data.objects["Plane"].name = "Wall"
        wall = bpy.data.objects["Wall"]
        wall.data.materials.clear()
        mat = bpy.data.materials.new(name='Material')
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].subsurface_method = 'BURLEY'
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 1, 0.887923, 1)
        wall.data.materials.append(mat)     
        
        # render
        if self.args.render_mode == "BLENDER_EEVEE":
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            bpy.context.scene.eevee.taa_samples = 128
            bpy.context.scene.camera = camera
        elif self.args.render_mode == "CYCLES":
            bpy.context.scene.camera = camera
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.scene.cycles.samples = 256
            bpy.context.scene.cycles.subsurface_samples = 256
        else: raise ValueError(f"Unknown render mode: {self.args.render_mode}")
            
        self._render_setting(self.endFrame)
        bpy.ops.render.render(animation=True)
        self._deleteAllObjects()
        
        # disable log out redirection
        os.close(fd)
        os.dup(old)
        os.close(old)

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
    assert len(sys.argv) == 12, "[Blender] Invalid number of arguments"
    try:
        if bpy.app.background:
            args = sys.argv[sys.argv.index("--") + 1:]
            args = [arg for arg in args if arg != "--"]
            dict_args = {
                "fbx": args[0],             # combined bvh file
                "endFrame": int(args[1]),   # endframes
                "video": args[2],           # output video file
                "render_mode": args[3],     # EEVEE or CYCLES
                "fps": int(args[4]),        # fps 
                "bmap": args[5],            # bmap
                "halfbody": args[6],        # halfbody
            }    
        blender = BlenderVisualizer(dict_args)
        blender.render_clip()
    
    except SystemExit as ex:
        if ex.code is None:
            exit_status = 0
        else: exit_status = ex.code
        print(f"[BLENDER-Render] Exit with status {exit_status}")
        if bpy.app.background: sys.exit(exit_status)