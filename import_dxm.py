# DXM loader for Blender

import os
import bpy
import mathutils
import math
import numpy as np

from struct import pack, unpack
from .shader import new_socket, Shader, get_shader

# Original model is Y UP, but blender is Z UP by default, we convert that here.
bone_up_Y = mathutils.Matrix(((1.0, 0.0, 0.0, 0.0),
                            (0.0, 0.0, -1.0, 0.0),
                            (0.0, 1.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0, 1.0)))

# Read helper functions
def read_ushort(file):
    return unpack('<H', file.read(2))[0]


def read_int(file):
    return unpack('<i', file.read(4))[0]


def read_uint(file):
    return unpack('<I', file.read(4))[0]


def read_float(file):
    return unpack('<f', file.read(4))[0]


def read_str(file):
    data = bytearray()
    while True:
        char = file.read(1)
        if char == b'\0':
            break
        data.extend(char)
    return data.decode('shift-jis')

def read_wstr(file):
    data = bytearray()
    while True:
        char = file.read(2)
        if char == b'\0\0':
            break
        data.extend(char)
    return data.decode('utf-16')


def read_matrix(file):
    mat = mathutils.Matrix()
    for y in range(4):
        for x in range(4):
            mat[x][y] = read_float(file)
    return mat

def read_matrix2(file):
    pox = read_float(file)
    poy = read_float(file)
    poz = read_float(file)
    thx = read_float(file)
    thy = read_float(file)
    thz = read_float(file)
    eul = mathutils.Euler((math.radians(thx), math.radians(thy), math.radians(thz)), 'XYZ')
    mat_rot = eul.to_matrix()
    mat_loc = mathutils.Matrix.Translation((pox, poy, poz))
    mat = mat_loc @ mat_rot.to_4x4()
    return mat

# Parsing functions
def parse_names(f, count, offset):
    names = []
    f.seek(offset)
    for i in range(count):
        base = f.tell()
        str_offset = read_uint(f)
        next = f.tell()
        assert next - base == 4

        if str_offset != 0:
            f.seek(base+str_offset)
            names.append(read_wstr(f))
            f.seek(next)
        else:
            names.append(None)
    return names


def parse_bones(f, count, offset):
    bones = []
    f.seek(offset)
    for i in range(count):
        bone = {}
        base = f.tell()
        
        bone['ID'] = i
        bone['matrix_parent'] = mathutils.Matrix.Identity(4)
        bone['parent'] = -1       # All bones are initially set to non-parent status.
        
        bone['matrix_local'] = read_matrix2(f) # Transformation Matrix local
        bone['matrix_invbind'] = read_matrix2(f) # Transformation Matrix Invert Bind Pose
        
        # If one is set, the other is always set as well
        bone['unk3'] = read_float(f)  # Float3 Contact positions? Set for "actual bones" but 0 for "aim point" bones?
        bone['unk4'] = read_float(f)
        bone['unk5'] = read_float(f)

        bone['unk6'] = read_float(f)  # Unknown Float3? Set for "actual bones" but 0 for "aim point" bones?
        bone['unk7'] = read_float(f)
        bone['unk8'] = read_float(f)

        bone['unk48'] = read_float(f) # Unknown Float4(0x48)
        
        bone_name = read_int(f)
                
        bone['child_count'] = read_int(f) # Child bone count
        bone['first_child'] = read_int(f) # Offset to First Child. 0 for none.

        bone['index'] = read_int(f) # Bone Index

        next = f.tell()
        assert next - base == 92

        f.seek(base + bone_name)
        bone['name'] = read_wstr(f) # Bone name

        f.seek(next)
        bones.append(bone)
    
    for i in range(count):         
        for j in range(bones[i]['child_count']):
            index = i + j + int(bones[i]['first_child'] / 0x5C)
            bones[index]['parent'] = bones[i]['ID']
        
    return bones


def parse_textures(f, count, offset):
    textures = []
    f.seek(offset)
    for i in range(count):
        texture = {}
        base = f.tell()
        texture['index'] = read_uint(f)
        name = read_uint(f)
        filename = read_uint(f)
        f.read(4) # Always zero
        next = f.tell()
        assert next - base == 16

        f.seek(base+name)
        texture['name'] = read_wstr(f)
        f.seek(base+filename)
        texture['filename'] = read_wstr(f)

        f.seek(next)
        textures.append(texture)
    return textures


def parse_mat_param(f, count, offset):
    mat_params = []
    f.seek(offset)
    for i in range(count):
        mat_param = {}
        base = f.tell()

        mat_param_name = read_int(f)

        mat_param['type'] = read_int(f) # Type (0 is RGBA, 1 is value(float32*1), 2 is Offset to string)
        unk = read_int(f)
        Data_ofs = read_int(f)

        next = f.tell()
        assert next - base == 16

        f.seek(base + mat_param_name)
        mat_param['name'] = read_str(f) # MaterialParam name
        mat_param['tex_name'] = "" #For except TextureParameter
        
        if mat_param['type'] == 0:
            f.seek(base + Data_ofs)
            mat_param['val0'] = read_float(f)
        elif mat_param['type'] == 1:
            f.seek(base + Data_ofs)
            mat_param['val0'] = read_float(f)
            mat_param['val1'] = read_float(f)
            mat_param['val2'] = read_float(f)
            mat_param['val3'] = read_float(f)            
        elif mat_param['type'] == 2:
            f.seek(base + Data_ofs)
            mat_param['val0'] = read_int(f)
            f.seek(base + Data_ofs + mat_param['val0'])
            mat_param['tex_name'] = read_wstr(f)

        f.seek(next)
        mat_params.append(mat_param)
    return mat_params


def parse_mat_txr(f, count, offset):
    mat_txrs = []
    f.seek(offset)
    for i in range(count):
        mat_txr = {}
        base = f.tell()
        mat_txr['texture'] = read_uint(f)
        string = read_uint(f)
        mat_txr['unk0'] = read_ushort(f)
        mat_txr['unk1'] = read_ushort(f)
        f.read(12) # Always zero
        mat_txr['unk2'] = read_uint(f)
        next = f.tell()
        assert next - base == 28

        f.seek(base+string)
        mat_txr['map'] = read_str(f)    #Indicate Texture Type for MDB.(ex)abledo,normal,damage

        f.seek(next)
        mat_txrs.append(mat_txr)
    return mat_txrs


def parse_materials(f, count, offset):
    materials = []
    f.seek(offset)
    for i in range(count):
        material = {}
        base = f.tell()

        material['index'] = i

        material_name = read_int(f)
        shader_name = read_int(f)
        render_name = read_int(f)
        param_count = read_int(f)
        param_offset = read_int(f)
        unk1 = read_int(f)
        unk2 = read_int(f)

        next = f.tell()
        assert next - base == 28

        f.seek(base + material_name)
        material['name'] = read_wstr(f) # Material name
        f.seek(base + shader_name)
        material['shader'] = read_wstr(f) # Shader name
        f.seek(base + render_name)
        material['Render_Name'] = read_str(f) # Render name(?)

        material['params'] = parse_mat_param(f, param_count, base+param_offset)

        f.seek(next)
        materials.append(material)

    return materials


def parse_vertex_layout(f, count, offset):
    layout = []
    texcnt = 0
    f.seek(offset)
    for i in range(count):
        element = {}
        base = f.tell()
        element['offset'] = read_int(f)
        element['type'] = read_int(f)
        element_name = read_int(f)
        
        element['channel'] = "0"    #Because there is only one except for texture.
        
        if element_name == 0:
            element['name'] = "position"
        elif element_name == 1:
            element['name'] = "normal"
        elif element_name == 3:
            element['name'] = "tangent"
        elif element_name == 4:
            element['name'] = "texcoord"    #(DiffuseTexture)
            element['channel'] = texcnt
            texcnt += 1
        elif element_name == 5:
            element['name'] = "texcoord"    #(ParameterTexture)
            element['channel'] = texcnt
            texcnt += 1
        elif element_name == 6:
            element['name'] = "texcoord"    #(NormalTexture)
            element['channel'] = texcnt
            texcnt += 1
        elif element_name == 7:
            element['name'] = "texcoord"    #(HighlightTexture_or_CubeTexture)
            element['channel'] = texcnt
            texcnt += 1
        elif element_name == 8:
            element['name'] = "texcoord"    #(CubeTexture)
            element['channel'] = texcnt
            texcnt += 1
        elif element_name == 12:
            element['name'] = "BLENDINDICES"
        elif element_name == 13:
            element['name'] = "BLENDWEIGHT"
        else:              #2,9,10,11
            element['name'] = "Unknown"

        next = f.tell()
        assert next - base == 12

        f.seek(next)
        layout.append(element)
    return layout


def parse_indices(f, count, offset):
    indices = []
    f.seek(offset)
    for i in range(count):
        indices.append(read_ushort(f))
    return indices


def parse_vertices(f, count, offset, layout, vertex_size):
    vertices = []
    f.seek(offset)
    # For each vertices
    for i in range(count):
        vertex = {}
        # For each layout type
        for j in range(len(layout)):
            elem = layout[j]
            array = []
            # Figure out vertex type, and set array with content
            type = elem['type']
            if type == 3: #float4
                array = unpack("ffff", f.read(16))
            elif type == 2: #float3
                array = unpack("fff", f.read(12))
            elif type == 1: #float2
                array = unpack("ff", f.read(8))
            elif type == 5: #ubyte4
                array = unpack("BBBB", f.read(4))
            else:
                print("Unknown vertex layout type: " + str(type))
                if j < len(layout) - 1:
                    f.seek(layout[j+1]['offset'] - elem['offset'])
                else:
                    f.seek(vertex_size - elem['offset'])
            vertex[elem['name'] + str(elem['channel'])] = array
        vertices.append(vertex)
    return vertices


def parse_meshes(f, count, offset):
    meshes = []
    f.seek(offset)
    for i in range(count):
        mesh = {}
        base = f.tell()

        mesh['index'] = i

        mesh_name = read_int(f)
        unk1 = read_int(f)
        layout_count = read_int(f)
        layout_offset = read_int(f)
        vertex_count = read_int(f)
        vertex_offset = read_int(f)
        mesh['vertex_size'] = read_int(f)
        unk2 = read_int(f)
        indice_count = read_int(f)
        indice_offset = read_int(f)
        unk3 = read_int(f)

        next = f.tell()
        assert next - base == 44

        f.seek(base + mesh_name)
        mesh['name'] = read_wstr(f) # Mesh name
        mesh['layout'] = parse_vertex_layout(f, layout_count, base+layout_offset)
        mesh['indices'] = parse_indices(f, indice_count, base+indice_offset)
        mesh['vertices'] = parse_vertices(f, vertex_count, base+vertex_offset, mesh['layout'], mesh['vertex_size'])
        
        mesh['material'] = 0

        f.seek(next)
        meshes.append(mesh)
    
    return meshes


def parse_objects(f, count, offset):
    objects = []
    f.seek(offset)
    for i in range(count):
        object = {}
        base = f.tell()

        object['index'] = i

        object_name = read_int(f)
        mesh_count = read_int(f)
        mesh_offset = read_int(f)
        unk1 = read_int(f)
        unk2 = read_int(f)
        unk3 = read_int(f)
        unk4 = read_int(f)
        unk5 = read_int(f)
        unk6 = read_int(f)
        unk7 = read_int(f)

        next = f.tell()
        assert next - base == 40

        f.seek(base + object_name)
        object['name'] = read_wstr(f) # Object name
        object['meshes'] = parse_meshes(f, mesh_count, base+mesh_offset)

        f.seek(next)
        objects.append(object)
    return objects


def parse_mdb(f):
    mdb = {}
    f.seek(0)
    magic = f.read(4)
    version = f.read(4)
    material_count = read_int(f)
    material_offset = read_int(f)
    bone_count = read_int(f)
    bone_offset = read_int(f)
    object_count = read_int(f)
    object_offset = read_int(f)
    
    #The number of bones in the header appears to be different from the actual number of bones.
    bone_count_act = int((object_offset - bone_offset) / 0x5c)

    assert magic == b'DXM\x00'
    assert version == b'\x01\x01\x00\x00'

    mdb['materials'] = parse_materials(f, material_count, material_offset)

    mdb['bones'] = parse_bones(f, bone_count_act, bone_offset)

    mdb['objects'] = parse_objects(f, object_count, object_offset)

    return mdb


def warnparam(input, material, param):
    if input is None:
        print('Warning: Material ' + material['name'] + ' references missing parameter ' + material['shader'] + '.' + param['name'])
    return input


# Main function
def load(operator, context, filepath='', **kwargs):

    # Parse MDB
    with open(filepath, 'rb') as f:
        mdb = parse_mdb(f)

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    # Texture cache
    textures = {}

    # Create unswizzle node group　#Used for normal map calculations
    if bpy.data.node_groups.get('Normal Unswizzle') is None:
        nspace = 160
        unswizzle = bpy.data.node_groups.new('Normal Unswizzle', 'ShaderNodeTree')

        group_inputs = unswizzle.nodes.new('NodeGroupInput')
        group_inputs.location[0] = nspace * 0
        new_socket(unswizzle, 'Color', 'INPUT', 'NodeSocketColor')
        new_socket(unswizzle, 'Alpha', 'INPUT', 'NodeSocketFloat')

        splitRGB = unswizzle.nodes.new('ShaderNodeSeparateRGB')
        splitRGB.location[0] = nspace * 1
        unswizzle.links.new(splitRGB.inputs['Image'], group_inputs.outputs['Color'])

        mulR = unswizzle.nodes.new('ShaderNodeMath')
        mulR.location[0] = nspace * 2
        mulR.operation = 'MULTIPLY_ADD'
        unswizzle.links.new(mulR.inputs[0], group_inputs.outputs['Alpha'])
        mulR.inputs[1].default_value = 2.0
        mulR.inputs[2].default_value = -1.0

        mulG = unswizzle.nodes.new('ShaderNodeMath')
        mulG.location[0] = nspace * 2
        mulG.location[1] = mulG.location[1] - 170
        mulG.operation = 'MULTIPLY_ADD'
        unswizzle.links.new(mulG.inputs[0], splitRGB.outputs['G'])
        mulG.inputs[1].default_value = 2.0
        mulG.inputs[2].default_value = -1.0
        
        mulB = unswizzle.nodes.new('ShaderNodeVectorMath')
        mulB.location[0] = nspace * 3
        mulB.location[1] = mulG.location[1] - 170
        mulB.operation = 'DOT_PRODUCT'
        unswizzle.links.new(mulB.inputs[0], mulR.outputs['Value'])
        unswizzle.links.new(mulB.inputs[1], mulG.outputs['Value'])

        subB = unswizzle.nodes.new('ShaderNodeMath')
        subB.location[0] = nspace * 4
        subB.location[1] = mulG.location[1] - 170
        subB.operation = 'SUBTRACT'
        subB.inputs[0].default_value = 1.0
        unswizzle.links.new(subB.inputs[1], mulB.outputs['Value'])

        sqrtB = unswizzle.nodes.new('ShaderNodeMath')
        sqrtB.location[0] = nspace * 5
        sqrtB.location[1] = mulG.location[1] - 170
        sqrtB.operation = 'SQRT'
        unswizzle.links.new(sqrtB.inputs[0], subB.outputs['Value'])

        combineRGB = unswizzle.nodes.new('ShaderNodeCombineRGB')
        combineRGB.location[0] = nspace * 6
        unswizzle.links.new(combineRGB.inputs['R'], mulR.outputs['Value'])
        unswizzle.links.new(combineRGB.inputs['G'], mulG.outputs['Value'])
        unswizzle.links.new(combineRGB.inputs['B'], sqrtB.outputs['Value'])

        group_outputs = unswizzle.nodes.new('NodeGroupOutput')
        group_outputs.location[0] = nspace * 7
        new_socket(unswizzle, 'Color', 'OUTPUT', 'NodeSocketColor')
        unswizzle.links.new(group_outputs.inputs['Color'], combineRGB.outputs['Image'])

    # Create materials
    materials = []
    for mdb_material in mdb['materials']:
        lshader = mdb_material['shader'].lower()
        material = bpy.data.materials.new(mdb_material['name']) #Load material and shader names
        
        #Each mesh selects a material by material name     
        for object in mdb['objects']:
            for mdb_mesh in object['meshes']:
                if mdb_mesh['name'] == mdb_material['name'] :
                    mdb_mesh['material'] = mdb_material['index']

        material.use_nodes = True
        mat_nodes = material.node_tree
        # Remove default node if it exists
        for node in mat_nodes.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                mat_nodes.nodes.remove(node)
                break
        unhandled = 0

        shader = get_shader(mdb_material['shader'])

        mat_out = None
        for node in mat_nodes.nodes:
            if node.type == 'OUTPUT_MATERIAL':
                mat_out = node
                break
        shader_node = material.node_tree.nodes.new('ShaderNodeGroup')
        shader_node.node_tree = shader.shader_tree
        shader_node.show_options = False
        shader_node.width = 240
        shader_node.location[1] = mat_out.location[1]
        mat_nodes.links.new(mat_out.inputs['Surface'], shader_node.outputs['Surface'])

        # Set up material parameters    #Reads RGBA when 0 and value when 1
        for param in mdb_material['params']:
            if 'Texture' not in param['name']:
                if 'm_MaterialDiffuse' in param['name']:
                    name = 'm_MaterialDiffuse'
                elif 'm_MaterialSpecularColor' in param['name']:
                    name = 'm_MaterialSpecularColor'
                elif 'm_MaterialSpecularPower' in param['name']:
                    name = 'm_MaterialSpecularPower'
                elif 'm_MaterialBumpHeight' in param['name']:
                    name = 'm_MaterialBumpHeight'
                elif 'm_RefrectionRate' in param['name']:
                    name = 'm_RefrectionRate'
                elif 'g_Highlight' in param['name']:
                    name = 'g_Highlight'
                elif 'g_Time' in param['name']:
                    name = 'g_Time' 
                else :
                    name = '' 
            
                if param['type'] == 0:
                    input_node = warnparam(shader_node.inputs.get(name), mdb_material, param)
                    if input_node is not None:
                        input_node.default_value = param['val0']
                elif param['type'] == 1:
                    input_col = warnparam(shader_node.inputs.get(name), mdb_material, param)
                    if input_col is not None:
                        input_alpha = shader_node.inputs.get(name + '_alpha')
                        input_col.default_value = (param['val0'], param['val1'], param['val2'], 1)
                        # It's okay for alpha to be missing, there are no parameters of size 3
                        if input_alpha is not None:
                            input_alpha.default_value = param['val3']

        # Add all material textures
        for texture in mdb_material['params']:
            txr_map = texture['name']    #(MDB)albedo,normal,damage(DXM)m_MaterialDiffuse,m_MaterialSpecularColor,m_MaterialBumpHeight
            if 'Texture' in txr_map:
                if 'Diffuse' in texture['name']:
                    name = 'm_MaterialDiffuse'
                elif 'Parameter' in texture['name']:
                    name = 'parameter'
                elif 'Normal' in texture['name']:
                    name = 'normal'
                elif 'Highlight' in texture['name']:
                    name = 'highlight'
                elif 'Cube' in texture['name']:
                    name = 'cube'
                texImage = mat_nodes.nodes.new('ShaderNodeTexImage')
                filename = texture['tex_name']
            #filename = mdb['textures'][texture['texture']]['filename']
                if filename in textures:
                    texImage.image = textures[filename]
                else:
                #Try and load texture from folder
                    image = None
                    try:
                        image = bpy.data.images.load(os.path.join(os.path.dirname(filepath), '..', 'TEXTURE', filename))
                    except RuntimeError as e: # Ignore texture import error
                        print("Failed to find texture.")
                        print(e)
                    if image is not None:
                        texImage.image = image
                        textures[filename] = image
                    # Why is Straight being treated as Premultiplied by cycles?
                        image.alpha_mode = 'CHANNEL_PACKED'
                        if 'Diffuse' not in txr_map:
                            image.colorspace_settings.name = 'Non-Color'

                texImage.location[0] = shader_node.location[0] - 700 + unhandled * 40
                texImage.location[1] = shader_node.location[1] - unhandled * 40
                unhandled += 1
                input_col = shader_node.inputs.get(name)
                if input_col is not None:
                    if name == 'normal':
                    # Unswizzle normal map
                        unswizzle = material.node_tree.nodes.new('ShaderNodeGroup')
                        unswizzle.location[0] = shader_node.location[0] - 350
                        unswizzle.node_tree = bpy.data.node_groups.get('Normal Unswizzle')
                        unswizzle.show_options = False
                        material.node_tree.links.new(unswizzle.inputs['Color'], texImage.outputs['Color'])
                        material.node_tree.links.new(unswizzle.inputs['Alpha'], texImage.outputs['Alpha'])

                    # Connect fixed normal map
                        normalMap = mat_nodes.nodes.new('ShaderNodeNormalMap')
                        normalMap.location[0] = shader_node.location[0] - 200
                        mat_nodes.links.new(normalMap.inputs['Color'], unswizzle.outputs['Color'])
                        mat_nodes.links.new(input_col, normalMap.outputs['Normal'])
                    else:
                        input_alpha = shader_node.inputs.get(name + '_alpha')
                        mat_nodes.links.new(input_col, texImage.outputs['Color'])
                        if input_alpha is not None:
                            mat_nodes.links.new(input_alpha, texImage.outputs['Alpha'])
                    param = shader.param_map[name]
                    if len(param) >= 3:
                        uvmap = mat_nodes.nodes.new('ShaderNodeUVMap')
                        uvmap.location[0] = texImage.location[0] - 200
                        uvmap.location[1] = texImage.location[1] - 200
                        uvmap.uv_map = 'UVMap'
                        mat_nodes.links.new(texImage.inputs['Vector'], uvmap.outputs['UV'])

        # Deselect all nodes
        for node in mat_nodes.nodes:
            node.select = False

        materials.append(material)

    # Add armature and bones
    armature = bpy.data.armatures.new('Armature')
    armature_obj = bpy.data.objects.new(os.path.splitext(os.path.basename(filename))[0], armature)
    context.scene.collection.objects.link(armature_obj)
    context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    edit_bones = armature.edit_bones
    bones = []
    
    for mdb_bone in mdb['bones']:
        # Create bone with name
        bone = edit_bones.new(mdb_bone['name'])
        # No length would mean they get removed for some reason, so we give it a fixed non zero length
        bone.length = 0.25 

        if mdb_bone['parent'] >= 0:
            bone.parent = bones[mdb_bone['parent']]
            #bone.matrix = mdb_bone['matrix_parent'] @ mdb_bone['matrix_local']    #EDF3 bones are absolute coordinates that do not depend on the parent and do not require calculation.
            bone.matrix = bone_up_Y @ mdb_bone['matrix_local']
        else:
            bone.matrix = bone_up_Y @ mdb_bone['matrix_local']

        bone['unknown_floats'] = [float(mdb_bone['unk3']), float(mdb_bone['unk4']),
                                  float(mdb_bone['unk5']), float(mdb_bone['unk6']),
                                  float(mdb_bone['unk7']), float(mdb_bone['unk8']),
                                  float(mdb_bone['unk48'])]
        # Add bone to bone list
        bones.append(bone)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add meshes
    for object in mdb['objects']:
        name = object['name']
        empty = bpy.data.objects.new(name, None)
        context.scene.collection.objects.link(empty)
        for mdb_mesh in object['meshes']:
            vertices = mdb_mesh['vertices']

            # Read indices
            faces = []
            indices = mdb_mesh['indices']
            for i in range(0, len(indices), 3):
                faces.append((indices[i+0], indices[i+1], indices[i+2]))

            # Read vertices
            vertex = []
            for vert in vertices:
                x = vert['position0'][0]
                y = vert['position0'][1]
                z = vert['position0'][2]
                vertex.append((x, -z, y))

            # Add basic mesh
            mesh = bpy.data.meshes.new('%s_Data' % name)
            mesh_obj = bpy.data.objects.new(name, mesh)
            mesh_obj.data = mesh

            mesh.from_pydata(vertex, [], faces)
            mesh.polygons.foreach_set('use_smooth', (True,)*len(faces))

            # Read normals
            if 'normal0' in vertices[0]:
                normals = []
                for vert in vertices:
                    normal = vert['normal0']
                    normal /= np.sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2])
                    #normal = vert['normal0'].astype(float)
                    #normal /= np.sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2])
                    x = normal[0]
                    y = normal[1]
                    z = normal[2]
                    normals.append((x, -z, y))
                mesh.normals_split_custom_set_from_vertices(normals)
                if bpy.app.version < (4, 1, 0):
                    mesh.use_auto_smooth = True  # Enable custom normals

            # Add UV maps
            for i in range(4):
                coordstr = 'texcoord' + str(i)
                if coordstr in vertices[0]:
                    uvmap = mesh.uv_layers.new(name='UVMap' + ('' if i == 0 else str(i+1)))
                    for face in mesh.polygons:
                        for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
                            texcoord = vertices[vert_idx][coordstr]
                            uvmap.data[loop_idx].uv[0] = texcoord[0]
                            uvmap.data[loop_idx].uv[1] = 1.0 - texcoord[1]

            # Add vertex groups
            if 'BLENDWEIGHT0' in vertices[0]:
                groups = []
                for n in range(len(mdb['bones'])):      #Add vertex groups so that the IDs for BLENDINDICE (different from normal IDs) of the bones are in ascending order.
                    cnt = n
                    while mdb['bones'][cnt]['index'] != n:
                        cnt += 1
                        if cnt >= len(mdb['bones']):
                            cnt = 0
                    else:
                        groups.append(mesh_obj.vertex_groups.new(name=mdb['bones'][cnt]['name']))
                                                                         
                    #groups.append(mesh_obj.vertex_groups.new(name=mdb['bones'][n]['name']['index']))
                for i, vert in enumerate(vertices):
                    for n in range(4):
                        if vert['BLENDWEIGHT0'][n] != 0:
                            groups[vert['BLENDINDICES0'][n]].add([i], vert['BLENDWEIGHT0'][n], 'ADD')

            mod = mesh_obj.modifiers.new("Armature", 'ARMATURE')
            mod.object = armature_obj

            # Assign material
            if mdb_mesh['material'] != -1:
                mesh.materials.append(materials[mdb_mesh['material']])

            mesh.update()

            context.scene.collection.objects.link(mesh_obj)
            mesh_obj.parent = empty
    return {'FINISHED'}
