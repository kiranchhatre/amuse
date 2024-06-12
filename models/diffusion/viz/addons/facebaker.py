# Borrowed from CaMN

bl_info = {
    "name" : "FaceBaker",
    "author" : "Naoya Iwamoto",
    "version" : (0,1),
    "blender" : (2, 83, 5),
    "support": "TESTING",
    "location" : "View3D > Add > Mesh",
    "description" : "Blendeshape Baker from Mesh Sequences",
    "warning" : "",
    "wiki_url" : "",
    "tracker_url" : "",
    "category" : "Object"
}

import bpy
from bpy.types import Operator, Panel, PropertyGroup, CollectionProperty, StringProperty, IntProperty
from bpy.props import PointerProperty, CollectionProperty, IntProperty, StringProperty

import json

class FaceBaker_panel(Panel):
    # パネルのラベル名を定義する
    # パネルを折りたたむパネルヘッダーに表示される
    bl_label = "FaceBaker Panel"
    # クラスのIDを定義する
    # 命名規則は CATEGORY_PT_name
    bl_idname = "FaceBaker"
    # パネルを使用する領域を定義する
    # 利用可能な識別子は以下の通り
    #   EMPTY：無し
    #   VIEW_3D：3Dビューポート
    #   IMAGE_EDITOR：UV/画像エディター
    #   NODE_EDITOR：ノードエディター
    #   SEQUENCE_EDITOR：ビデオシーケンサー
    #   CLIP_EDITOR：ムービークリップエディター
    #   DOPESHEET_EDITOR：ドープシート
    #   GRAPH_EDITOR：グラフエディター
    #   NLA_EDITOR：非線形アニメーション
    #   TEXT_EDITOR：テキストエディター
    #   CONSOLE：Pythonコンソール
    #   INFO：情報、操作のログ、警告、エラーメッセージ
    #   TOPBAR：トップバー
    #   STATUSBAR：ステータスバー
    #   OUTLINER：アウトライナ
    #   PROPERTIES：プロパティ
    #   FILE_BROWSER：ファイルブラウザ
    #   PREFERENCES：設定
    bl_space_type = 'VIEW_3D'
    # パネルが使用される領域を定義する
    # 利用可能な識別子は以下の通り
    # ['WINDOW'、 'HEADER'、 'CHANNELS'、 'TEMPORARY'、 'UI'、
    #  'TOOLS'、 'TOOL_PROPS'、 'PREVIEW'、 'HUD'、 'NAVIGATION_BAR'、
    #  'EXECUTE'、 'FOOTER'の列挙型、 'TOOL_HEADER']
    bl_region_type = 'UI'
    # パネルタイプのオプションを定義する
    # DEFAULT_CLOSED：作成時にパネルを開くか折りたたむ必要があるかを定義する。
    # HIDE_HEADER：ヘッダーを非表示するかを定義する。Falseに設定するとパネルにはヘッダーが表示される。
    # デフォルトは {'DEFAULT_CLOSED'}
    bl_options = {'DEFAULT_CLOSED'}
    # パネルの表示順番を定義する
    # 小さい番号のパネルは、大きい番号のパネルの前にデフォルトで順序付けられる
    # デフォルトは 0
    bl_order = 0
    # パネルのカテゴリ名称を定義する
    # 3Dビューポートの場合、サイドバーの名称になる
    # デフォルトは名称無し
    bl_category = "FaceBaker"
 
    # 描画の定義
    def draw(self, context):
        # Operatorをボタンとして配置する
        draw_layout = self.layout
        draw_layout.label(text= "Select Mesh:", icon='SELECT_SET')

        # 要素行を作成する
        select_row = draw_layout.row()
        # オブジェクト選択用のカスタムプロパティを配置する
        select_row.prop(context.scene.facebaker_obj_src, "prop_objectselect", text='neutral')

        # 要素行を作成する
        select_row = draw_layout.row()
        # オブジェクト選択用のカスタムプロパティを配置する
        select_row.prop(context.scene.facebaker_obj_exp, "prop_objectselect", text='expression')

        #path to directory
        draw_layout.label(text= "Bake Facial Blendshape:", icon='IMPORT')
        select_row = draw_layout.row()
        button_row = draw_layout.row()
        button_row.operator("facebaker.bake_blendshape")

        draw_layout.label(text= "Load Animation Data:", icon='IMPORT')

        '''
        col = draw_layout.column(align=True)
        col.label(text='Registration')
        col.separator(factor = 1.0)
        row = col.template_list('FACE_UL_list', '', scene, 'facebaker_facial_items', scene, 'facebaker_facial_index')
        row = col.row(align=True)
        op = row.operator('facebaker.facial_items', text='Register Secondary Objects', icon='ADD')
        op.operation = 'ADD'

        '''
        select_row = draw_layout.row()
        col = draw_layout.column()
        #col.template_list("FACE_UL_list", "", context.scene, "objects", context.scene,"active_object_index")
        col.template_list("FACE_UL_list", "", context.scene, "facebaker_facial_items", context.scene, "active_object_index")

        #col.label(text='Select Objects')
        #col.menu('facebaker.facial_items_add', text='', icon='ADD')
        col = draw_layout.column(align=True)
        col.operator('facebaker.facial_items_add', text='Add Items', icon='ADD')
        col.operator('facebaker.facial_items_remove', text='Remove Items', icon='REMOVE')
        #col.separator(factor = 3.0)

        # Rename blendshape
        draw_layout.label(text= "Rename Blendshape Target:", icon='EXPORT')
        select_row = draw_layout.row()
        select_row.prop(context.scene.facebaker_obj_src, "name_list_path", text='path')
        button_row = draw_layout.row()
        button_row.operator("facebaker.blendshape_rename")

        select_row = draw_layout.row()
        select_row.prop(context.scene.facebaker_obj_src, "animation_data_path", text='path')
        button_row = draw_layout.row()
        button_row.operator("facebaker.load_animation_multi")

        draw_layout.label(text= "Export Animation Data:", icon='EXPORT')
        select_row = draw_layout.row()
        select_row.prop(context.scene.facebaker_obj_exp, "animation_data_path", text='path')
        button_row = draw_layout.row()
        button_row.operator("facebaker.export_animation")

        draw_layout.label(text= "Export Blendshape Data:", icon='EXPORT')
        select_row = draw_layout.row()
        select_row.prop(context.scene.facebaker_obj_src, "blendshape_data_path", text='path')
        button_row = draw_layout.row()
        button_row.operator("facebaker.export_blendshapes")

def remove_keyframe(mesh):
    # check whether keyframes exists
    if mesh.shape_keys.animation_data is None:
        print("no keyframes")        
    elif mesh.shape_keys.animation_data.action is None:
        print("no keyframes")
    else:
        curve_keys = mesh.shape_keys.animation_data.action.fcurves
        for shape_key_fcurve in curve_keys:
            for key in shape_key_fcurve.keyframe_points:
                key_frame_id = key.co[0]
                for block in mesh.shape_keys.key_blocks:
                    block.keyframe_delete("value", frame=key_frame_id) 

class FaceBaker_export_blendshapes(Operator):
    bl_idname = "facebaker.export_blendshapes"
    bl_label = "Export Blendshapes (.obj)"
    dl_description = "Export Blendshapes Data (.obj)"
    
    def execute(self, context):
        obj_src = context.scene.facebaker_obj_src
        src_mesh = obj_src.prop_objectselect.data
        blendshape_path = obj_src.blendshape_data_path

        for id, shape_key in enumerate(src_mesh.shape_keys.key_blocks[1:]):
            # reset value
            for shape_key_tmp in src_mesh.shape_keys.key_blocks[1:]:
                shape_key_tmp.value = 0.0

            shape_key.value = 1.0
            name = blendshape_path + str(id).zfill(4) + '_' + shape_key.name + '.obj'        
            bpy.ops.export_scene.obj(filepath=name, use_selection=True, keep_vertex_order=True, use_materials=False)    

        return {'FINISHED'}

class FaceBaker_blendshape_rename(Operator):
    bl_idname = "facebaker.blendshape_rename"
    bl_label = "Rename Blendshape (.json)"
    dl_description = "Load Blendshape Name List Data (.json)"

    def execute(self, context):
        obj_src = context.scene.facebaker_obj_src
        face_objects = context.scene.facebaker_facial_items
        name_list_path = obj_src.name_list_path

        with open(name_list_path) as f:
            df = json.load(f)

        print("Num of Expressions: ", len(df["name_list"]))

        for item in face_objects:
            obj = bpy.data.objects[item.name]
            mesh = obj.data

            if len(mesh.shape_keys.key_blocks) == 0:
                print("No shape key on ", mesh)
            else:
                # Assign names
                print(len(mesh.shape_keys.key_blocks[1:]), len(df["name_list"]))
                if len(mesh.shape_keys.key_blocks[1:]) == len(df["name_list"]):
                    for index, key_shape in enumerate(mesh.shape_keys.key_blocks[1:]):
                        key_shape.name = df["name_list"][index]
                else:
                    print( "num of blendshape: {0} and name_list has {1}".format(len(mesh.shape_keys.key_blocks[1:], len(df["name_list"])) ))

        return {'FINISHED'}

class FaceBaker_load_animation_multi(Operator):
    bl_idname = "facebaker.load_animation_multi"
    bl_label = "Load Animation (.json)"
    dl_description = "Load Animation Data (.json)"

    def execute(self, context):
        obj_src = context.scene.facebaker_obj_src
        anim_path = obj_src.animation_data_path
        frame_rate = bpy.context.scene.render.fps
        face_objects = context.scene.facebaker_facial_items

        with open(anim_path) as f:
            df = json.load(f)

        print("Num of Expressions: ", len(df["names"]))
        print("Num of Frames: ", len(df["frames"]))

        # loading facial data
        for item in face_objects:
            obj = bpy.data.objects[item.name]
            mesh = obj.data

            if len(mesh.shape_keys.key_blocks) == 0:
                print("No shape key on ", mesh)
            else:
                remove_keyframe(mesh)

                # Assign names
                # if len(mesh.shape_keys.key_blocks[1:]) == len(df["names"]):
                #     for index, key_shape in enumerate(mesh.shape_keys.key_blocks[1:]):
                #         key_shape.name = df["names"][index]

                # assign animation for each mesh
                for frame in df['frames']:
                    for w_id, w_val in enumerate(frame['weights']):
                        key_shape_name = df["names"][w_id]                        

                        if key_shape_name in mesh.shape_keys.key_blocks:
                            key_shape = mesh.shape_keys.key_blocks[key_shape_name]
                            key_shape.value = w_val
                            current_frame = frame_rate * frame['time']
                            key_shape.keyframe_insert("value", frame=current_frame) 

        # set frame
        bpy.data.scenes[0].frame_start = 1
#        bpy.data.scenes[0].frame_end = current_frame + 1
        bpy.context.scene.frame_set(bpy.data.scenes[0].frame_start)

        return {'FINISHED'}

class FaceBaker_load_animtion(Operator):
    bl_idname = "facebaker.load_animation"
    bl_label = "Load Animation (.json)"
    dl_description = "Load Animation Data (.json)"
    
    def execute(self, context):

        obj_src = context.scene.facebaker_obj_src
        src_mesh = obj_src.prop_objectselect.data
        anim_path = obj_src.animation_data_path
        frame_rate = bpy.context.scene.render.fps

        if len(src_mesh.shape_keys.key_blocks) == 0:
            print("No shape key on ", src_mesh)
            return {'CANCELLED'}

        with open(anim_path) as f:
            df = json.load(f)

        print("Num of Expressions: ", len(df["names"]))
        print("Num of Frames: ", len(df["frames"]))

        remove_keyframe(src_mesh)

        # assign animation on source model
        for frame in df['frames']:
            for w_id, w_val in enumerate(frame['weights']):
                key_shape_name = df["names"][w_id]

                if key_shape_name in src_mesh.shape_keys.key_blocks:
                    key_shape = src_mesh.shape_keys.key_blocks[key_shape_name]
                    key_shape.value = w_val
                    current_frame = frame_rate * frame['time']
                    key_shape.keyframe_insert("value", frame=current_frame) 

        # set frame
        bpy.data.scenes[0].frame_start = 1
        bpy.data.scenes[0].frame_end = current_frame + 1
        bpy.context.scene.frame_set(bpy.data.scenes[0].frame_start)

        return {'FINISHED'}

class FaceBaker_export_animation(Operator):
    bl_idname = "facebaker.export_animation"
    bl_label = "Export Animation (.json)"
    dl_description = "Export Animation Data (.json)"
    
    def execute(self, context):

        obj_src = context.scene.facebaker_obj_src
        src_mesh = obj_src.prop_objectselect.data
        anim_path = context.scene.facebaker_obj_exp.animation_data_path
        frame_rate = bpy.context.scene.render.fps

        if len(src_mesh.shape_keys.key_blocks) == 0:
            print("No shape key on ", src_mesh)
            return {'CANCELLED'}

        export_anim = {}
        export_anim['names'] = [face.name for face in src_mesh.shape_keys.key_blocks[1:]]
        export_anim['frames'] = []

        for frame_id in range(bpy.data.scenes[0].frame_start, bpy.data.scenes[0].frame_end+1):

            bpy.context.scene.frame_set(frame_id)
            face_weight = [float(face.value) for face in src_mesh.shape_keys.key_blocks[1:]]

            per_frame_data = {}
            per_frame_data['weights'] = face_weight
            per_frame_data['time'] = frame_id * 1.0 / frame_rate
            per_frame_data['rotation'] = []
            export_anim['frames'].append(per_frame_data)

        with open(anim_path, 'w') as f:
            json.dump(export_anim, f, indent=4)

        bpy.context.scene.frame_set(bpy.data.scenes[0].frame_start)

        return {'FINISHED'}


class FaceBaker_bake_blendshape(Operator):
    bl_idname = "facebaker.bake_blendshape"
    bl_label = "Bake Blendshape"
    dl_description = "Bake Blendshape from Sequences"
    
    def execute(self, context):

        obj_src = context.scene.facebaker_obj_src.prop_objectselect
        src_mesh = obj_src.data

        # set first frame
        first_frame = bpy.context.scene.frame_start
        last_frame = bpy.context.scene.frame_end + 1
        bpy.context.scene.frame_set(first_frame)

        #obj_src.shape_key_clear()
        #obj_src.shape_key_add()

        if len(src_mesh.shape_keys.key_blocks) == 0:
            print("No shape key on", src_mesh)
            return {'CANCELLED'}

        for n in range(first_frame, last_frame): 
            bpy.context.scene.frame_set(n)

            # expression
            obj_exp = context.scene.facebaker_obj_exp.prop_objectselect
            exp_mesh = obj_exp.data

            # Create new shape key
            sk = obj_src.shape_key_add()
            sk.name = "exp_" + str(str(n).zfill(3))
            
            # position each vert
            for i in range(len(exp_mesh.vertices)):
                sk.data[i].co = exp_mesh.vertices[i].co

        bpy.context.scene.frame_set(first_frame)

        for key in src_mesh.shape_keys.key_blocks:
            key.slider_min = 0
            key.slider_max = 1

        return {'FINISHED'}

class FaceBaker_object_select_properties(PropertyGroup):
    # オブジェクト選択時のチェック関数を定義する
    def prop_object_select_poll(self, context, ):
        # メッシュオブジェクトのみ選択可能
        if(context and context.type in ('MESH', )):
            return True
        return False

    # シーン上のパネルに表示するオブジェクト選択用のカスタムプロパティを定義する
    prop_objectselect: PointerProperty(
        name = "Select Object",         # プロパティ名
        type = bpy.types.Object,        # タイプ
        description = "",               # 説明文
        poll = prop_object_select_poll, # チェック関数
    )

    animation_data_path: bpy.props.StringProperty(
        name="File Path",
        description="Only .json files will be listed",
        subtype="FILE_PATH")

    trained_data_path: bpy.props.StringProperty(
        name="File Path",
        description="Only .json files will be listed",
        subtype="FILE_PATH")

    blendshape_data_path: bpy.props.StringProperty(
        name="File Path",
        description="Only .json files will be listed",
        subtype="FILE_PATH")

    name_list_path: bpy.props.StringProperty(
        name="Blendshape Name Path",
        description="Only .json files will be listed",
        subtype="FILE_PATH")


class FaceBaker_facial_items(bpy.types.PropertyGroup):
    bl_idname = "facebaker.facial_items"
    active_index = bpy.props.IntProperty()
    name: StringProperty(name='name',description='object name')

    # オブジェクト選択時のチェック関数を定義する
    def prop_object_select_poll(self, context, ):
        # メッシュオブジェクトのみ選択可能
        if(context and context.type in ('MESH', )):
            return True
        return False

    # prop_objectselect: PointerProperty(
    #     name = "Select Object",         # プロパティ名
    #     type = bpy.types.Object,        # タイプ
    #     description = "",               # 説明文
    #     poll = prop_object_select_poll, # チェック関数
    # )

    # operation: bpy.props.EnumProperty(
    #          items = [
    #             ('ADD',"add","add an object to face objects"),
    #             ('REMOVE',"remove","remove an object from face objects"),
    #             ('CLEAR',"clear","clear list - remove all objects"),
    #          ],
    #          name = "Operation",
    #          description = "List operation",
    #          default='ADD',
    #         )

    # def execute(self, context):        
    #     scene = context.scene
    #     face_objects = scene.faceit_face_objects
        
    #     # add to face objects collection property
    #     if self.operation == 'ADD':
    #         objects_add = list(context.selected_objects)
    #         if not objects_add:
    #             objects_add.append(context.object)

    #         print('ADD')

    #     # remove from face objects
    #     elif self.operation == 'REMOVE' and len(face_objects) > 0:
    #         print('REMOVE')
            
    #     # clear all face objects
    #     elif self.operation == 'CLEAR':
    #         print('CLEAR')
  
        return {'FINISHED'}

class FaceBaker_facial_items_add(bpy.types.Operator):
    bl_idname = "facebaker.facial_items_add"
    bl_label = "Add Item"

    def execute(self, context):
        face_objects = context.scene.facebaker_facial_items

        # object select list
        objects_add = list(context.selected_objects)
        if not objects_add:
            objects_add.append(context.object)
        for obj in objects_add:
            # check if that item exists
            obj_exists = any([obj.name in item.name for item in face_objects])
            if not obj_exists:
                item = face_objects.add()
                item.name = obj.name

            # set active index to new item
            context.scene.facebaker_facial_index = face_objects.find(obj.name)

        return {'FINISHED'}

class FaceBaker_facial_items_remove(bpy.types.Operator):
    bl_idname = "facebaker.facial_items_remove"
    bl_label = "Remove Item"
    def execute(self, context):
        face_objects = context.scene.facebaker_facial_items

        for obj in context.selected_objects:
            context.scene.facebaker_facial_index = face_objects.find(obj.name)
            context.scene.facebaker_facial_items.remove(context.scene.facebaker_facial_index)

        '''
        face_objects = context.scene.facebaker_facial_items
        if len(face_objects) > 0:
            context.scene.facebaker_facial_items.remove(context.scene.facebaker_facial_index)
        '''

        return {'FINISHED'}


class FaceBaker_export_bs_animation_seq(Operator):
    bl_idname = "facebaker.export_bs_animation_seq"
    bl_label = "Export Blendshape Animation Sequences"
    dl_description = "Export Blendshape Animation Sequences"
    
    def execute(self, context):
        obj_src = context.scene.facebaker_obj_src

        frame_start = bpy.data.scenes[0].frame_start
        frame_end = bpy.data.scenes[0].frame_end

        for frame_id in range(frame_start, frame_end):
            bpy.context.scene.frame_set(frame_id)

            bpy.ops.export_scene.obj(filepath = '', use_selection=True)


        return {'FINISHED'}     

def set_active_object(object_name):
    '''
    select the object 
    @object_name: String
    '''
    obj = bpy.data.objects.get(object_name)
    if obj:
        bpy.data.objects[object_name].select_set(state=True)
        bpy.context.view_layer.objects.active = obj
    else:
        print('WARNING! Object {} does not exist'.format(obj.name))
        return{'CANCELLED'}

def update_object_index(self, context):
    scene = self
    if scene.facial_index_updated != scene.facebaker_facial_index:
        scene.facial_index_updated = scene.facebaker_facial_index

        #futils.clear_object_selection()
        # obj = bpy.data.objects.get(scene.facebaker_facial_items[scene.facebaker_facial_index].name)
        # obj.select_set(True)

        # set_active_object(scene.faceit_face_objects[scene.facebaker_facial_index].name)
        '''
        scene.facial_index_updated = -2
        '''

#-------------------------------

class FACE_UL_list(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon_value=layout.icon(item))
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)

classes = (
    FaceBaker_panel,
    FaceBaker_facial_items,
    FaceBaker_facial_items_add,
    FaceBaker_facial_items_remove,
    FaceBaker_object_select_properties,
    FaceBaker_bake_blendshape,
    FaceBaker_load_animtion,
    FaceBaker_load_animation_multi,
    FaceBaker_export_animation,
    FaceBaker_export_blendshapes,
    FaceBaker_blendshape_rename,
    FACE_UL_list,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.facebaker_obj_src = PointerProperty(type=FaceBaker_object_select_properties)
    bpy.types.Scene.facebaker_obj_exp = PointerProperty(type=FaceBaker_object_select_properties)
    bpy.types.Scene.facebaker_facial_items = CollectionProperty(type=FaceBaker_facial_items)
    bpy.types.Scene.facebaker_facial_index = IntProperty(default = 0, update=update_object_index)
    bpy.types.Scene.facial_index_updated = IntProperty(default = 1,)
    bpy.types.Scene.active_object_index = IntProperty()

    print("[FACEBAKER] Enable plugin")

def unregister():
    # remove custom property in scene
    del bpy.types.Scene.facebaker_obj_src
    del bpy.types.Scene.facebaker_obj_exp
    del bpy.types.Scene.facebaker_facial_items
    del bpy.types.Scene.facebaker_facial_index
    del bpy.types.Scene.facial_index_updated
    del bpy.types.Scene.active_object_index

    for c in classes:
        bpy.utils.unregister_class(c)
    print("[FACEBAKER] Disable plugin")

if __name__ == "__main__":
    register()  