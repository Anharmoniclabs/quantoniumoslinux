"""Run inside Blender. Creates a real armature with weighted proxy limbs, not JC geometry."""
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def material(name,color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color,1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color,1)
    bsdf.inputs['Roughness'].default_value = .8
    return mat


def box(name,location,scale,mat):
    bpy.ops.mesh.primitive_cube_add(size=1,location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    return obj


def main():
    input_path,out = [Path(v).resolve() for v in sys.argv[sys.argv.index('--')+1:]]
    spec = json.loads(input_path.read_text())
    size = spec['size']
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 12
    scene.render.resolution_x = spec['width']
    scene.render.resolution_y = spec['height']
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.fps = spec['fps']
    scene.frame_start,scene.frame_end = 1,len(spec['frames'])
    scene.world.color = (.12,.16,.24)
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'Medium High Contrast'
    angle = math.radians(12)
    floor = round(.87*size)/size
    target = Vector((0,0,(floor-.5)*2/math.cos(angle)))
    bpy.ops.object.camera_add(location=target+Vector((0,-8,8*math.tan(angle))))
    camera = bpy.context.object
    camera.name = 'Registered_Shot_Camera'
    camera.rotation_euler = (target-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 2*spec['width']/spec['height']
    camera.data.lens = 50
    scene.camera = camera
    bpy.context.view_layer.update()

    def world(x,y,depth=0.):
        # Invert the fixed orthographic camera. Depth changes are inferred; the
        # visible projection remains constrained by the supplied source image.
        z = target.z + ((.5-y/size)*2-depth*math.sin(angle))/math.cos(angle)
        return Vector(((x/size-.5)*2,depth,z))

    def project(v):
        co = world_to_camera_view(scene,camera,v)
        x = (co.x*spec['width']-(spec['width']-spec['height'])/2)/spec['height']*size
        return [x,(1-co.y)*size]

    floor_mat = material('Courtyard slate',(.16,.18,.24))
    wall_mat = material('Dusk plaster',(.38,.32,.35))
    trim_mat = material('Stone trim',(.26,.25,.31))
    far_mat = material('Distant blue',(.17,.22,.32))
    box('Ground plane',(0,1,-.04),(12,14,.08),floor_mat)
    box('Terrace wall',(0,2,.18),(12,.25,.36),wall_mat)
    box('Terrace cap',(0,2,.40),(12,.38,.08),trim_mat)
    for x in (-2.7,-2.1,2.1,2.7):
        box('Column',(x,1.5,1.4),(.13,.20,2.8),trim_mat)
        box('Column capital',(x,1.5,2.8),(.25,.30,.10),wall_mat)
    for i in range(13):
        x = (i-6)*.62
        h = 1.35+(i*17%11)*.095
        box('Distant architecture',(x,6,h/2),(.55,.7,h),far_mat)
    for x in (-4,-3,-2,-1,0,1,2,3,4):
        box('Paving seam',(x,0,.002),(.008,10,.002),trim_mat)
    for y in (-2,-1,0,1):
        box('Paving cross seam',(0,y,.002),(10,.008,.002),trim_mat)
    bpy.ops.object.light_add(type='AREA',location=(-3,-4,6))
    light = bpy.context.object
    light.name = 'Warm key'
    light.data.energy,light.data.shape,light.data.size = 850,'DISK',5
    light.data.color = (1.,.79,.62)
    bpy.ops.object.light_add(type='AREA',location=(3,2,4))
    light = bpy.context.object
    light.name = 'Cool rim'
    light.data.energy,light.data.size = 600,4
    light.data.color = (.53,.68,1.)
    light.rotation_euler = (Vector((0,0,1))-light.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath = str(out/'background.png')
    bpy.ops.render.render(write_still=True)

    # A volume proxy establishes camera/occlusion/motion, not character likeness.
    proxy_mat = material('Motion proxy only',(.27,.55,.65))
    joints = []
    for name in spec['joint_names']:
        obj = bpy.data.objects.new('Joint_'+name,None)
        scene.collection.objects.link(obj)
        obj.empty_display_type = 'SPHERE'
        obj.empty_display_size = .025
        joints.append(obj)
    depths = [0.,0.,0.,-.06,-.09,-.12,.06,.09,.12,-.04,-.06,.04,.06,0.]
    first = [world(x,y,depths[j]) for j,(x,y) in enumerate(spec['frames'][0]['points'])]
    for obj,pos in zip(joints,first):
        obj.location = pos
    bpy.ops.object.armature_add(location=(0,0,0))
    rig = bpy.context.object
    rig.name = 'Image_Fitted_Blocking_Armature'
    rig.show_in_front = True
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in list(rig.data.edit_bones):
        rig.data.edit_bones.remove(bone)
    for index,(a,b) in enumerate(spec['bones']):
        bone = rig.data.edit_bones.new(f'limb_{index:02d}')
        bone.head,bone.tail = first[a],first[b]
    bpy.ops.object.mode_set(mode='OBJECT')
    for index,(a,b) in enumerate(spec['bones']):
        name = f'limb_{index:02d}'
        bone = rig.pose.bones[name]
        con = bone.constraints.new('COPY_LOCATION')
        con.target = joints[a]
        con = bone.constraints.new('STRETCH_TO')
        con.target = joints[b]
        direction = first[b]-first[a]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=8,
                                           location=(first[a]+first[b])/2)
        mesh = bpy.context.object
        mesh.name = 'Weighted_proxy_'+name
        mesh.rotation_mode = 'QUATERNION'
        mesh.rotation_quaternion = direction.to_track_quat('Z','Y')
        radius = .075 if index < 2 else .035
        mesh.scale = (radius,radius,direction.length/2)
        bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
        mesh.data.materials.append(proxy_mat)
        group = mesh.vertex_groups.new(name=name)
        group.add(list(range(len(mesh.data.vertices))),1.,'REPLACE')
        mod = mesh.modifiers.new('Image fitted skeleton','ARMATURE')
        mod.object = rig
    max_error = 0.
    contact_z = []
    exported = []
    for frame in spec['frames']:
        projected = []
        for j,(obj,(x,y)) in enumerate(zip(joints,frame['points'])):
            obj.location = world(x,y,depths[j])
            obj.keyframe_insert(data_path='location',frame=frame['frame'])
            uv = project(obj.location)
            max_error = max(max_error,math.hypot(uv[0]-x,uv[1]-y))
            projected.append(uv)
        if frame['grounded']:
            contact_z.append(abs(joints[-1].location.z))
        exported.append({**frame,'points':projected})
    if max_error > .01 or max(contact_z,default=0) > .0001:
        raise RuntimeError(f'Projection/contact gate failed: {max_error}, {contact_z}')
    # Audit the evaluated armature, not merely the intended joint coordinates.
    max_rig_error = 0.
    for frame in spec['frames']:
        scene.frame_set(frame['frame'])
        bpy.context.view_layer.update()
        for index,(a,b) in enumerate(spec['bones']):
            bone = rig.pose.bones[f'limb_{index:02d}']
            max_rig_error = max(max_rig_error,
                (rig.matrix_world@bone.head-joints[a].matrix_world.translation).length,
                (rig.matrix_world@bone.tail-joints[b].matrix_world.translation).length)
    if max_rig_error > .002:
        raise RuntimeError(f'Evaluated rig does not follow image controls: {max_rig_error}')
    scene['qualification'] = 'Image-fitted 2.5D motion proxy. Estimated joints and depth; not a reconstructed, textured JC character.'
    rig['source_sha256'] = json.dumps([s['source_sha256'] for s in spec['sources']])
    for index,source in enumerate(spec['sources']):
        image = bpy.data.images.load(source['packed_reference'],check_existing=False)
        image.name = f'Original_reference_{index+1:02d}'
        image.pack()
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'motion_shot.blend'))
    artifacts = ['background.png','motion_shot.blend']
    artifacts += [f'reference_{i+1:02d}.png' for i in range(len(spec['sources']))]
    artifacts += [f'joint_review_{i+1:02d}.png' for i in range(len(spec['sources']))]
    for f in (1,round(len(exported)*.5),len(exported)):
        scene.frame_set(f)
        name = f'rig_proof_{f:03d}.png'
        scene.render.filepath = str(out/name)
        bpy.ops.render.render(write_still=True)
        artifacts.append(name)
    result = dict(input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  data_fingerprint=spec['data_fingerprint'],blender_version=bpy.app.version_string,
                  max_projection_error_px=max_error,max_support_height_world=max(contact_z,default=0),
                  max_evaluated_bone_error_world=max_rig_error,
                  frames=exported,fps=spec['fps'],size=size,
                  artifacts={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in artifacts},
                  limitation=scene['qualification'])
    tmp = out/'shot_manifest.tmp.json'
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(out/'shot_manifest.json')
    print('BLENDER SHOT VERIFIED',max_error,'px projection error',flush=True)


if __name__ == '__main__':
    main()
