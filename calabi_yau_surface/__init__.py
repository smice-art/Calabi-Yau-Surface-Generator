bl_info = {
    "name": "Calabi Surface Generator",
    "author": "Claudio",
    "version": (1, 3),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > Calabi Tab",
    "description": "Generate Calabi-like complex surfaces with seam welding and custom mathematical shaders.",
    "category": "Add Mesh",
}

import bpy
import bmesh
import cmath
import numpy as np
from math import sin, cos, sinh, cosh, pi
from bpy.props import IntProperty, FloatProperty, StringProperty, BoolProperty

# -------------------- Shader Material Builder --------------------

def get_or_create_calabi_material(mat_name="Calabi_Iridescent_Mat"):
    """Generates a procedural holographic material with contour lines."""
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # 1. Output Node
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (800, 0)

    # 2. Main Principled BSDF
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = (400, 0)

    # Set parameters across Blender 3.x and 4.x versions safely
    def set_socket(name, val):
        if name in principled.inputs:
            principled.inputs[name].default_value = val

    set_socket("Roughness", 0.12)
    set_socket("Metallic", 0.85)
    set_socket("Coat Weight", 0.8)   # Blender 4.0+
    set_socket("Clearcoat", 0.8)     # Pre-Blender 4.0

    # 3. Texture Coordinates & Separate XYZ
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-1000, 0)

    sep_xyz = nodes.new(type="ShaderNodeSeparateXYZ")
    sep_xyz.location = (-800, -150)
    links.new(tex_coord.outputs["Object"], sep_xyz.inputs["Vector"])

    # 4. Layer Weight (Facing Angle for Iridescence)
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    layer_weight.location = (-800, 150)
    layer_weight.inputs["Blend"].default_value = 0.4

    # Combine Facing + Z position for dynamic color shifting
    add_driver = nodes.new(type="ShaderNodeMath")
    add_driver.operation = "ADD"
    add_driver.location = (-600, 100)
    links.new(layer_weight.outputs["Facing"], add_driver.inputs[0])
    links.new(sep_xyz.outputs["Z"], add_driver.inputs[1])

    # 5. Spectrum ColorRamp (Purple -> Cyan -> Magenta -> Gold)
    ramp_spectrum = nodes.new(type="ShaderNodeValToRGB")
    ramp_spectrum.location = (-350, 100)
    elements = ramp_spectrum.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (0.05, 0.01, 0.25, 1.0)  # Deep Violet

    e1 = elements.new(0.35)
    e1.color = (0.0, 0.85, 0.85, 1.0)             # Neon Cyan

    e2 = elements.new(0.7)
    e2.color = (1.0, 0.15, 0.6, 1.0)             # Vivid Magenta

    elements[1].position = 1.0
    elements[1].color = (1.0, 0.8, 0.15, 1.0)    # Warm Gold

    links.new(add_driver.outputs["Value"], ramp_spectrum.inputs["Fac"])
    links.new(ramp_spectrum.outputs["Color"], principled.inputs["Base Color"])

    # 6. Procedural Spherical Wave (Equipotential Contour Lines)
    wave = nodes.new(type="ShaderNodeTexWave")
    wave.location = (-600, -250)
    wave.wave_type = "RINGS"
    wave.rings_direction = "SPHERICAL"
    wave.inputs["Scale"].default_value = 4.0
    wave.inputs["Distortion"].default_value = 1.5
    wave.inputs["Detail"].default_value = 1.0
    links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])

    # Sharpen the wave into thin glowing contour lines
    ramp_lines = nodes.new(type="ShaderNodeValToRGB")
    ramp_lines.location = (-350, -250)
    ramp_lines.color_ramp.interpolation = "CONSTANT"
    r_elems = ramp_lines.color_ramp.elements
    r_elems[0].position = 0.0
    r_elems[0].color = (0.0, 0.0, 0.0, 1.0)
    r_elems[1].position = 0.88
    r_elems[1].color = (2.0, 2.0, 2.0, 1.0)  # Boosted intensity
    links.new(wave.outputs["Color"], ramp_lines.inputs["Fac"])

    # Connect contour emission
    if "Emission Color" in principled.inputs:
        links.new(ramp_spectrum.outputs["Color"], principled.inputs["Emission Color"])
        if "Emission Strength" in principled.inputs:
            links.new(ramp_lines.outputs["Color"], principled.inputs["Emission Strength"])
    elif "Emission" in principled.inputs:
        links.new(ramp_spectrum.outputs["Color"], principled.inputs["Emission"])

    # 7. Surface Output Link
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    return mat


# -------------------- Math / Mesh Builder --------------------

def calcZ1(x, y, k, n):
    return cmath.exp(1j * (2 * cmath.pi * k / n)) * (cmath.cosh(x + y * 1j)) ** (2 / n)

def calcZ2(x, y, k, n):
    return cmath.exp(1j * (2 * cmath.pi * k / n)) * (1 / 1j) * (cmath.sinh(x + y * 1j)) ** (2 / n)

def calcZ1Real(x, y, k, n):
    return (calcZ1(x, y, k, n)).real

def calcZ2Real(x, y, k, n):
    return (calcZ2(x, y, k, n)).real

def calcZ(x, y, k1_, k2_, n1_, n2_, a_):
    z1 = calcZ1(x, y, k1_, n1_)
    z2 = calcZ2(x, y, k2_, n2_)
    return z1.imag * cos(a_) + z2.imag * sin(a_)


def build_calabi_mesh(
    n1_dimension=2,
    n2_dimension=2,
    a_radian=0.4,
    x_dim=20,
    y_dim=20,
    mesh_name="CalabiMesh",
    object_name="CalabiObject",
    apply_material=True,
    remove_existing=True,
):
    """Builds geometry, welds branch seams, applies material, and returns object."""
    x_vals = np.linspace(0, pi / 2, x_dim)
    y_vals = np.linspace(-pi / 2, pi / 2, y_dim)
    Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="xy")

    all_verts = []
    all_faces = []
    offset = 0

    for k1 in range(n1_dimension):
        for k2 in range(n2_dimension):
            v_calc_X = np.vectorize(lambda xx, yy: float(calcZ1Real(xx, yy, k1, n1_dimension)))
            v_calc_Y = np.vectorize(lambda xx, yy: float(calcZ2Real(xx, yy, k2, n2_dimension)))
            v_calc_Z = np.vectorize(
                lambda xx, yy: float(calcZ(xx, yy, k1, k2, n1_dimension, n2_dimension, a_radian))
            )

            X = v_calc_X(Xg, Yg)
            Y = v_calc_Y(Xg, Yg)
            Z = v_calc_Z(Xg, Yg)

            branch_verts = []
            for row in range(y_dim):
                for col in range(x_dim):
                    branch_verts.append((float(X[row, col]), float(Y[row, col]), float(Z[row, col])))

            branch_faces = []
            for row in range(y_dim - 1):
                for col in range(x_dim - 1):
                    v0 = row * x_dim + col
                    v1 = v0 + 1
                    v2 = v0 + x_dim + 1
                    v3 = v0 + x_dim
                    # Counter-clockwise winding
                    branch_faces.append((v0 + offset, v1 + offset, v2 + offset, v3 + offset))

            all_verts.extend(branch_verts)
            all_faces.extend(branch_faces)
            offset += len(branch_verts)

    # Remove existing object/mesh if requested
    if remove_existing and object_name in bpy.data.objects:
        old = bpy.data.objects[object_name]
        bpy.data.objects.remove(old, do_unlink=True)
    if remove_existing and mesh_name in bpy.data.meshes:
        oldm = bpy.data.meshes[mesh_name]
        bpy.data.meshes.remove(oldm, do_unlink=True)

    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(all_verts, [], all_faces)

    # BMesh: Weld touching vertices across branches
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    mesh.validate(verbose=False)
    mesh.update()

    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)

    # Active & selected
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Smooth shading
    for p in mesh.polygons:
        p.use_smooth = True

    # Material Assignment
    if apply_material:
        mat = get_or_create_calabi_material()
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    return obj


# -------------------- Live Update Scheduling --------------------

_calabi_update_tag = 0

def schedule_calabi_build(delay=0.25):
    global _calabi_update_tag
    _calabi_update_tag += 1
    mytag = _calabi_update_tag

    def _delayed():
        if mytag != _calabi_update_tag:
            return None
        sc = bpy.context.scene
        try:
            build_calabi_mesh(
                n1_dimension=sc.calabi_n1,
                n2_dimension=sc.calabi_n2,
                a_radian=sc.calabi_a,
                x_dim=sc.calabi_xdim,
                y_dim=sc.calabi_ydim,
                mesh_name=sc.calabi_mesh_name,
                object_name=sc.calabi_object_name,
                apply_material=sc.calabi_use_mat,
                remove_existing=True,
            )
        except Exception as e:
            print("Calabi build failed during scheduled update:", e)
        return None

    bpy.app.timers.register(_delayed, first_interval=delay)


def _on_prop_update(self, context):
    if getattr(context.scene, "calabi_auto_update", False):
        schedule_calabi_build(delay=0.25)


# -------------------- Blender UI (N-panel) --------------------

class CALABI_OT_generate(bpy.types.Operator):
    bl_idname = "calabi.generate"
    bl_label = "Generate Calabi Surface"
    bl_description = "Generate the Calabi-like surface with custom shader"

    def execute(self, context):
        sc = context.scene
        build_calabi_mesh(
            n1_dimension=sc.calabi_n1,
            n2_dimension=sc.calabi_n2,
            a_radian=sc.calabi_a,
            x_dim=sc.calabi_xdim,
            y_dim=sc.calabi_ydim,
            mesh_name=sc.calabi_mesh_name,
            object_name=sc.calabi_object_name,
            apply_material=sc.calabi_use_mat,
            remove_existing=True,
        )
        return {"FINISHED"}


class CALABI_PT_panel(bpy.types.Panel):
    bl_label = "Calabi Surface"
    bl_idname = "CALABI_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Calabi"

    def draw(self, context):
        layout = self.layout
        sc = context.scene

        row = layout.row()
        row.prop(sc, "calabi_auto_update")
        row.operator(CALABI_OT_generate.bl_idname, text="Generate Now", icon="MESH_GRID")

        layout.prop(sc, "calabi_n1")
        layout.prop(sc, "calabi_n2")
        layout.prop(sc, "calabi_a")
        layout.prop(sc, "calabi_xdim")
        layout.prop(sc, "calabi_ydim")
        layout.prop(sc, "calabi_use_mat")
        layout.prop(sc, "calabi_mesh_name")
        layout.prop(sc, "calabi_object_name")


# -------------------- Registration --------------------

classes = (CALABI_OT_generate, CALABI_PT_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.calabi_n1 = IntProperty(
        name="n1", default=2, min=1, max=16, update=_on_prop_update
    )
    bpy.types.Scene.calabi_n2 = IntProperty(
        name="n2", default=2, min=1, max=16, update=_on_prop_update
    )
    bpy.types.Scene.calabi_a = FloatProperty(
        name="a (radians)", default=0.4, min=0.0, max=6.283185307, update=_on_prop_update
    )
    bpy.types.Scene.calabi_xdim = IntProperty(
        name="x dim", default=20, min=2, max=512, update=_on_prop_update
    )
    bpy.types.Scene.calabi_ydim = IntProperty(
        name="y dim", default=20, min=2, max=512, update=_on_prop_update
    )
    bpy.types.Scene.calabi_use_mat = BoolProperty(
        name="Auto Material", default=True, update=_on_prop_update
    )
    bpy.types.Scene.calabi_mesh_name = StringProperty(
        name="Mesh Name", default="CalabiMesh"
    )
    bpy.types.Scene.calabi_object_name = StringProperty(
        name="Object Name", default="CalabiObject"
    )
    bpy.types.Scene.calabi_auto_update = BoolProperty(
        name="Auto Update", default=False
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    for prop in (
        "calabi_n1",
        "calabi_n2",
        "calabi_a",
        "calabi_xdim",
        "calabi_ydim",
        "calabi_use_mat",
        "calabi_mesh_name",
        "calabi_object_name",
        "calabi_auto_update",
    ):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

