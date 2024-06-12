
import bpy
import sys
from pathlib import Path
 
if __name__ == "__main__":

    addon_path = sys.argv[-1]
    
    installable_addons = ['Stop-motion-OBJ', 'facebaker']
    addons = ['import_runtime_mhx2', 'retarget_bvh', 'auto_rig_pro-master', 'smplx_blender_addon']
    repo_addon_path = Path(addon_path, "addons")
    
    for addon in installable_addons:
        if addon not in bpy.context.preferences.addons.keys():
            print(f"[BLENDER] {addon} is not enabled. Enabling...")
            _path = [p for p in repo_addon_path.iterdir() if p.name.startswith(addon)][0]
            bpy.ops.preferences.addon_install(overwrite=True, filepath=str(_path))
            bpy.ops.preferences.addon_enable(module=addon)
            bpy.ops.wm.save_userpref()
            assert addon in bpy.context.preferences.addons.keys(), f"[BLENDER] {addon} is not enabled."
        else: print(f"[BLENDER] Already enabled: {addon}")
    
    for addon in addons:
        if addon not in bpy.context.preferences.addons.keys():
            print(f"[BLENDER] {addon} is not enabled. Enabling...")
            bpy.ops.preferences.addon_enable(module=addon)
            bpy.ops.wm.save_userpref()
        else: print(f"[BLENDER] Already enabled: {addon}")
    
    # if not bpy.context.preferences.system.use_scripts_auto_execute:           # default True in bpy 3.4.1
    #     print("[BLENDER] Enabling python scripts auto execute...")
    #     bpy.context.preferences.system.use_scripts_auto_execute = True
    # else: print("[BLENDER] Python scripts auto execute is already enabled.") 

