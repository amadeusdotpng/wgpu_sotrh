use crate::texture;
use std::ops::Range;

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBS: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBS,
        }
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

pub trait DrawModel<'a> {
    fn draw_mesh(&mut self, mesh: &'a Mesh, material: &'a Material);
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
    );
    fn draw_model(&mut self, model: &'a Model);
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(&mut self, mesh: &'b Mesh, material: &'a Material) {
        self.draw_mesh_instanced(mesh, material, 0..1);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'a Material,
        instances: Range<u32>,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances)
    }
    fn draw_model(&mut self, model: &'b Model) {
    self.draw_model_instanced(model, 0..1);
    }
    fn draw_model_instanced(&mut self, model: &'b Model, instances: Range<u32>) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone())
        }
    }
}
