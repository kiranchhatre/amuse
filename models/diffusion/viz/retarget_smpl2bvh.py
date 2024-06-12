import bpy
import sys
import os
import re   
import numpy as np

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}

offsets_ = np.array([0.31232587, -35.140743, 1.2036551]) # computed from smplx rest pose

class bvh:
    @staticmethod
    def load(filename:str, order:str=None) -> dict:
        """Loads a BVH file.

        Args:
            filename (str): Path to the BVH file.
            order (str): The order of the rotation channels. (i.e."xyz")

        Returns:
            dict: A dictionary containing the following keys:
                * names (list)(jnum): The names of the joints.
                * parents (list)(jnum): The parent indices.
                * offsets (np.ndarray)(jnum, 3): The offsets of the joints.
                * rotations (np.ndarray)(fnum, jnum, 3) : The local coordinates of rotations of the joints.
                * positions (np.ndarray)(fnum, jnum, 3) : The positions of the joints.
                * order (str): The order of the channels.
                * frametime (float): The time between two frames.
        """

        f = open(filename, "r")

        i = 0
        active = -1
        end_site = False

        # Create empty lists for saving parameters
        names = []
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)

        # Parse the file, line by line
        for line in f:

            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "{" in line: continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offmatch.groups()))])
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis:2 + channelie]
                    if any([p not in channelmap for p in parts]):
                        continue
                    order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match("\s*JOINT\s+(\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                fnum = int(fmatch.group(1))
                positions = offsets[None].repeat(fnum, axis=0)
                rotations = np.zeros((fnum, len(offsets), 3))
                continue

            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                continue

            dmatch = line.strip().split(' ')
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1

        f.close()

        return {
            'rotations': rotations,
            'positions': positions,
            'offsets': offsets,
            'parents': parents,
            'names': names,
            'order': order,
            'frametime': frametime
        }

class quat:

    # Calculate quaternions from euler angles.
    @staticmethod
    def from_euler(e, order='zyx'):
        axis = {
            'x': np.asarray([1, 0, 0], dtype=np.float32),
            'y': np.asarray([0, 1, 0], dtype=np.float32),
            'z': np.asarray([0, 0, 1], dtype=np.float32)}

        q0 = quat.from_angle_axis(e[..., 0], axis[order[0]])
        q1 = quat.from_angle_axis(e[..., 1], axis[order[1]])
        q2 = quat.from_angle_axis(e[..., 2], axis[order[2]])

        return quat.mul(q0, quat.mul(q1, q2))

    # Calculate quaternions from axis angles.
    @staticmethod
    def from_angle_axis(angle, axis):
        c = np.cos(angle / 2.0)[..., None]
        s = np.sin(angle / 2.0)[..., None]
        q = np.concatenate([c, s * axis], axis=-1)
        return q

    # Multiply two quaternions (return rotations).
    @staticmethod
    def mul(x, y):
        x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
        y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

        return np.concatenate([
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    @staticmethod
    def to_scaled_angle_axis(x, eps=1e-5):
        return 2.0 * quat.log(x, eps)

    @staticmethod
    def log(x, eps=1e-5):
        length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,None]
        halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / length)
        return halfangle * x[...,1:]

class ConvertSMPL_BVH():
    
    def __init__(self, args):
        
        males =   ["wayne", "scott", "solomon", "lawrence", "stewart", \
                   "nidal", "zhao", "lu", "zhang", "carlos", \
                   "jorge", "itoi", "daiki", "jaime", "li"]
        females = ["carla", "sophie", "catherine", "miranda", "kieks", \
                   "ayana", "luqi", "hailing", "kexin", "goto", \
                   "reamey", "yingqing", "tiffnay", "hanieh", "katya"]
        
        # AMASS betas
        thin = np.array([1.11687825, -1.36551024,  0.55103563, -1.5038104,  -1.64676488, 
                         -0.82590038,  2.29900317, -0.22205,    -1.07184384, -0.8144531,  
                         0.96917122,  0.75640019,  0.53769029, -0.45284027, -0.05088619, 
                         0.15528568], dtype='float64')
        man = np.array([-1.07569918, -0.67030116, -0.1089249 ,  0.93177645, -0.5917884 ,
                        -3.09409259,  0.73120595, -0.1079956 , -2.0914698 ,  1.97981589,
                        -0.11120027, -1.9694126 ,  0.30192958,  0.25115846, -0.5360993 , 
                        2.03600332], dtype='float64')
        woman = np.array([-1.96817042, -0.38755523, -0.51895929,  1.54698493, -1.86540255,
                        -2.71480545,  1.14851103,  1.01697068, -0.75062194,  1.72701065,
                        -0.0734116 , -1.72183191, -0.21500692, -0.73933191,  1.08557402, 
                        1.56411365], dtype='float64')
        
        self.args = dotdict(args)
        self.src_bvh = self.args.bvh
        self.smpl_bvh = self.args.smpl_bvh
        self.infer_pipe = self.args.infer_pipe
        self.target_mesh = "SMPLX_TPOSE_FLAT"
        basename = os.path.basename(self.args.bvh)
        self.bvh_name = os.path.splitext(basename)[0]
        self.extract_path = self.args.extract_path
        if self.infer_pipe: person = self.bvh_name.split("_")[0]
        else: person = self.bvh_name.split("_")[1]
        
        if person in males: 
            self.gender = "male"
            self.betas = man
        elif person in females: 
            self.gender = "female"
            self.betas = woman
        else: int(f"Unknown person: {person}")
        
    def purge_orphans(self):
        if bpy.app.version >= (3, 0, 0):
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        else:
            result = bpy.ops.outliner.orphans_purge()
            if result.pop() != "CANCELLED":
                self.purge_orphans()
    
    def export_smpl_bvh(self):
        
        # log out redirection
        logfile = os.path.join(self.extract_path, self.bvh_name + "_smpl_blender.log")
        open(logfile, 'a').close() 
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)
        
        self._deleteAllObjects()
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        context = bpy.context
        area = next(area for area in context.screen.areas if area.type == 'VIEW_3D')
        scene = bpy.data.scenes['Scene']
        bpy.ops.import_anim.bvh(filepath=self.src_bvh)
        bpy.ops.import_anim.bvh(filepath=self.smpl_bvh)
        scene.source_rig = self.bvh_name
        scene.target_rig = self.target_mesh
        bpy.ops.arp.auto_scale()
        bpy.ops.arp.build_bones_list()
        bpy.ops.arp.import_config_preset(preset_name="beat_2_bvh_smpl")
        with open(self.src_bvh, "r") as pose_data:
            frame_len = 0
            for line in pose_data.readlines():
                frame_len += 1
            frame_len -= 433
        bpy.ops.arp.retarget('EXEC_DEFAULT', frame_start=0, frame_end=frame_len)
        
        scene.frame_start = 0
        scene.frame_end = frame_len
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[self.target_mesh].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[self.target_mesh]
        self.smplx_bvh_out = os.path.join(self.extract_path, self.bvh_name + ".bvh")
        bpy.ops.export_anim.bvh(filepath=str(self.smplx_bvh_out))
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        self.purge_orphans()
        
        # disable log out redirection
        os.close(fd)
        os.dup(old)
        os.close(old)
    
    def export_smpl_npz(self):
        bvh_read = bvh.load(self.smplx_bvh_out)
        rotations_read = bvh_read['rotations']
        positions_read = bvh_read['positions']
        frametime = bvh_read['frametime']
        fps = int(np.ceil(1 / frametime))
        
        # rotation
        order_bvh_SMPLX = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 17, 36, 13, 18, 37, 19, 38, 20, 39, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
        rotations_order_bvh_smplx = []
        for m in range(rotations_read.shape[0]):
            rotations_list = []
            for i in order_bvh_SMPLX:
                if i in [14, 15, 16]:
                    rotations_list.append([0, 0, 0])
                else:
                    rotations_list.append(rotations_read[m][i])
            rotations_order_bvh_smplx.append(np.array(rotations_list))

        quat_bvh_smplx = quat.from_euler(np.radians(rotations_order_bvh_smplx))
        rotation_matrix_bvh_smplx = quat.to_scaled_angle_axis(quat_bvh_smplx)
        smpl_rotations_by_axis = np.array(rotation_matrix_bvh_smplx)

        # position
        smpl_root_position = []
        for i in positions_read:
            smpl_root_position.append((i[0] - offsets_)/100)
        smpl_root_position = np.array(smpl_root_position)
        
        npz_file = os.path.join(self.extract_path, self.bvh_name + ".npz")
        gender__ = np.array(self.gender, dtype='<U7')
        mocap_frame_rate_ = np.array(fps, dtype='float64')
        betas__ = self.betas
        np.savez(
            npz_file,
            poses=smpl_rotations_by_axis,
            trans=smpl_root_position,
            gender=gender__,
            mocap_frame_rate=mocap_frame_rate_,
            betas=betas__
        )
    
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
    assert len(sys.argv) in [8, 9], "[Blender] Invalid number of arguments"
    try:
        if bpy.app.background:
            args = sys.argv[sys.argv.index("--") + 1:]
            args = [arg for arg in args if arg != "--"]
            dict_args = {
                "bvh": args[0],             # combined bvh file
                "extract_path": args[1],    # path to extract the smpl fbx file
                "smpl_bvh": args[2],        # smpl bvh file
                "infer_pipe": True if len(sys.argv) == 9 else False
            }    
        blender = ConvertSMPL_BVH(dict_args)
        blender.export_smpl_bvh()
        blender.export_smpl_npz()
    
    except SystemExit as ex:
        if ex.code is None:
            exit_status = 0
        else: exit_status = ex.code
        print(f"[BLENDER-Render] Exit with status {exit_status}")
        if bpy.app.background: sys.exit(exit_status)