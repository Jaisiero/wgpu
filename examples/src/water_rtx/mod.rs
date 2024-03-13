mod point_gen;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use nanorand::{Rng, WyRand};
use std::time::Instant;
use std::{borrow::Cow, iter, mem};
use wgpu::hal::AccelerationStructureBuildFlags;
use wgpu::ray_tracing::{
    AccelerationStructureUpdateMode, CommandEncoderRayTracing,
    DeviceRayTracing,
};
use wgpu::{ray_tracing as rt, util::DeviceExt, Features, Limits};

///
/// Radius of the terrain.
///
/// Changing this value will change the size of the
/// water and terrain. Note however, that changes to
/// this value will require modification of the time
/// scale in the `render` method below.
///
const SIZE: f32 = 29.0;

///
/// Location of the camera.
/// Location of light is in terrain/water shaders.
///
const CAMERA: Vec3 = glam::Vec3::new(-200.0, 70.0, 200.0);

struct Matrices {
    view: glam::Mat4,
    projection: glam::Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
struct TerrainUniforms {
    view_projection: [f32; 16],
    clipping_plane: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
struct WaterUniforms {
    view: [f32; 16],
    projection: [f32; 16],
    time_size_width: [f32; 4],
    height: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RTUniforms {
    view_inverse: [[f32; 4]; 4],
    proj_inverse: [[f32; 4]; 4],
}

struct Example {

    depth_buffer: wgpu::TextureView,

    current_frame: usize,

    ///
    /// Used to prevent issues when rendering after
    /// minimizing the window.
    ///
    active: Option<usize>,

    uniform_buf: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform_bind_group_layout: wgpu::BindGroupLayout,

    vertex_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl Example {
    ///
    /// Creates the view matrices, and the corrected projection matrix.
    ///
    fn generate_matrices(aspect_ratio: f32) -> Matrices {
        let projection =
            glam::Mat4::perspective_rh(59.0_f32.to_radians(), aspect_ratio, 0.001, 1000.0);
        let view = glam::Mat4::look_at_rh(CAMERA, glam::Vec3::ZERO, glam::Vec3::Y);

        let scale = glam::Mat4::from_scale(glam::Vec3::new(8.0, 1.5, 8.0));

        let reg_view = view * scale;

        Matrices {
            view: reg_view,
            projection,
        }
    }

    fn generate_uniforms(width: u32, height: u32) -> RTUniforms {
        let Matrices {
            view,
            projection,
        } = Self::generate_matrices(width as f32 / height as f32);

        RTUniforms {
            view_inverse: view.inverse().to_cols_array_2d(),
            proj_inverse: projection.inverse().to_cols_array_2d(),
        }
    }

    ///
    /// Initializes Uniforms and textures.
    ///
    fn initialize_resources(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &wgpu::Buffer,
        //terrain_normal_uniforms: &wgpu::Buffer,
        //terrain_flipped_uniforms: &wgpu::Buffer,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (wgpu::TextureView, wgpu::BindGroup) {
        // Matrices for our projection and view.
        // flipped_view is the view from under the water.
        let rt_uniforms = Self::generate_uniforms(config.width, config.height);
        println!("{:?}", rt_uniforms);
        // Put the uniforms into buffers on the GPU
        queue.write_buffer(uniforms, 0, bytemuck::cast_slice(&[rt_uniforms]));

        let texture_extent = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let draw_depth_buffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Buffer"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Depth Sampler"),
            ..Default::default()
        });

        let depth_view = draw_depth_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&depth_sampler),
                },
            ],
            label: Some("Water Bind Group"),
        });

        (depth_view, uniform_bind_group)
    }
}

const MAX_SIZE: usize = (4096) * 1024;

impl crate::framework::Example for Example {
    fn required_limits() -> Limits {
        Limits::default()
    }
    fn required_features() -> Features {
        wgpu::Features::RAY_QUERY | wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
    }
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        device.limits();
        let start = Instant::now();

        let water_vertices = point_gen::HexWaterMesh::generate(SIZE).generate_points();

        // Noise generation
        let terrain_noise = noise::OpenSimplex::default();

        // Random colouration
        let mut terrain_random = WyRand::new_seed(42);

        // Generate terrain. The closure determines what each hexagon will look like.
        let terrain =
            point_gen::HexTerrainMesh::generate(SIZE, |point| -> point_gen::TerrainVertex {
                use noise::NoiseFn;
                let noise = terrain_noise.get([point[0] as f64 / 5.0, point[1] as f64 / 5.0]) + 0.1;

                let y = noise as f32 * 22.0;

                // Multiplies a colour by some random amount.
                fn mul_arr(mut arr: [u8; 4], by: f32) -> [u8; 4] {
                    arr[0] = (arr[0] as f32 * by).min(255.0) as u8;
                    arr[1] = (arr[1] as f32 * by).min(255.0) as u8;
                    arr[2] = (arr[2] as f32 * by).min(255.0) as u8;
                    arr
                }

                // Under water
                const DARK_SAND: [u8; 4] = [235, 175, 71, 255];
                // Coast
                const SAND: [u8; 4] = [217, 191, 76, 255];
                // Normal
                const GRASS: [u8; 4] = [122, 170, 19, 255];
                // Mountain
                const SNOW: [u8; 4] = [175, 224, 237, 255];

                // Random colouration.
                let random = terrain_random.generate::<f32>() * 0.2 + 0.9;

                // Choose colour.
                let colour = if y <= 0.0 {
                    DARK_SAND
                } else if y <= 0.8 {
                    SAND
                } else if y <= 10.0 {
                    GRASS
                } else {
                    SNOW
                };
                point_gen::TerrainVertex {
                    position: Vec3::new(point[0], y, point[1]),
                    colour: mul_arr(colour, random),
                }
            });

        // Generate the buffer data.
        let mut terrain_vertices = terrain.make_buffer_data();

        println!(
            "size {}",
            terrain_vertices.len() * mem::size_of::<point_gen::TerrainVertexAttributes>()
        );
        if (terrain_vertices.len() * mem::size_of::<point_gen::TerrainVertexAttributes>())
            > MAX_SIZE
        {
            let new_len = MAX_SIZE / mem::size_of::<point_gen::TerrainVertexAttributes>();
            println!("new_len {new_len}");
            let new_len = new_len - (new_len % 3);
            terrain_vertices.truncate(new_len);
        }

        // Create the buffers on the GPU to hold the data.
        let water_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water vertices"),
            contents: bytemuck::cast_slice(&water_vertices),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
        });

        let blas_desc = rt::CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        };

        let water_geo_size = rt::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: water_vertices.len() as u32,
            index_format: None,
            index_count: None,
            flags: rt::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let water_blas = device.create_blas(
            &blas_desc,
            rt::BlasGeometrySizeDescriptors::Triangles {
                desc: vec![water_geo_size.clone()],
            },
        );

        let terrain_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain vertices"),
            contents: bytemuck::cast_slice(&terrain_vertices),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });

        let terrain_geo_size = rt::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: terrain_vertices.len() as u32,
            index_format: None,
            index_count: None,
            flags: rt::AccelerationStructureGeometryFlags::OPAQUE,
        };
        println!("{}, {}", water_vertices.len(), terrain_vertices.len());
        let terrain_blas = device.create_blas(
            &blas_desc,
            rt::BlasGeometrySizeDescriptors::Triangles {
                desc: vec![terrain_geo_size.clone()],
            },
        );

        let tlas = device.create_tlas(&rt::CreateTlasDescriptor {
            label: None,
            max_instances: 2,
            flags: rt::AccelerationStructureFlags::empty(),
            update_mode: AccelerationStructureUpdateMode::Build,
        });

        let vertex_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::AccelerationStructure,
                    count: None,
                }],
            });

        let vertex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &vertex_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            }],
        });
        let mut tlas_package = rt::TlasPackage::new(tlas, 2);

        *tlas_package.get_mut_single(0).unwrap() =
            Some(rt::TlasInstance::new_untransformed(&terrain_blas, 0, 0xff));
        *tlas_package.get_mut_single(1).unwrap() =
            Some(rt::TlasInstance::new_untransformed(&water_blas, 0, 0xff));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures(
            [
                rt::BlasBuildEntry {
                    blas: &water_blas,
                    geometry: rt::BlasGeometries::TriangleGeometries(vec![
                        rt::BlasTriangleGeometry {
                            size: &water_geo_size,
                            vertex_buffer: &water_vertex_buf,
                            first_vertex: 0,
                            vertex_stride: mem::size_of::<point_gen::WaterVertexAttributes>()
                                as wgpu::BufferAddress,
                            index_buffer: None,
                            index_buffer_offset: None,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                },
                rt::BlasBuildEntry {
                    blas: &terrain_blas,
                    geometry: rt::BlasGeometries::TriangleGeometries(vec![
                        rt::BlasTriangleGeometry {
                            size: &terrain_geo_size,
                            vertex_buffer: &terrain_vertex_buf,
                            first_vertex: 0,
                            vertex_stride: mem::size_of::<point_gen::TerrainVertexAttributes>()
                                as wgpu::BufferAddress,
                            index_buffer: None,
                            index_buffer_offset: None,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                },
            ]
            .iter(),
            iter::once(&tlas_package),
        );
        queue.submit(Some(encoder.finish()));

        // Create the bind group layout. This is what our uniforms will look like.
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Uniform variables such as projection/view.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<RTUniforms>() as _
                            ),
                        },
                        count: None,
                    },
                    // Depth texture for terrain.
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Sampler to be able to sample the textures.
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Uniforms"),
            size: mem::size_of::<RTUniforms>() as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group.
        // This puts values behind what was laid out in the bind group layout.
        println!("elapsed before init {}ms", start.elapsed().as_millis());
        let (depth_buffer, uniform_bind_group) = Self::initialize_resources(
            config,
            device,
            queue,
            &uniform_buf,
            &uniform_bind_group_layout,
        );
        println!("elapsed after init {}ms", start.elapsed().as_millis());
        // Upload/compile them to GPU code.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniform_bind_group_layout, &vertex_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the render pipelines. These describe how the data will flow through the GPU, and what
        // constraints and modifiers it will have.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        //panic!("success");
        // Done
        Example {
            uniform_bind_group_layout,

            depth_buffer,

            current_frame: 0,

            active: Some(0),

            uniform_buf,
            uniform_bind_group,

            pipeline,
            vertex_bind_group,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if config.width == 0 && config.height == 0 {
            // Stop rendering altogether.
            self.active = None;
            self.active = None;
            return;
        }
        self.active = Some(self.current_frame);

        // Regenerate all of the buffers and textures.

        let (depth_buffer, uniform_bind_group) = Self::initialize_resources(
            config,
            device,
            queue,
            &self.uniform_buf,
            &self.uniform_bind_group_layout,
        );
        self.uniform_bind_group = uniform_bind_group;

        self.depth_buffer = depth_buffer;
    }

    #[allow(clippy::eq_op)]
    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Increment frame count regardless of if we draw.
        self.current_frame += 1;
        // Write the sin/cos values to the uniform buffer for the water.
        /*let (water_sin, water_cos) = ((self.current_frame as f32) / 600.0).sin_cos();
        queue.write_buffer(
            &self.water_uniform_buf,
            mem::size_of::<[f32; 16]>() as wgpu::BufferAddress * 2,
            bytemuck::cast_slice(&[water_sin, water_cos]),
        );*/

        // Only render valid frames. See resize method.
        if let Some(active) = self.active {
            if active >= self.current_frame {
                return;
            }
        } else {
            return;
        }

        // The encoder provides a way to turn our instructions here, into
        // a command buffer the GPU can understand.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Main Command Encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.uniform_bind_group, &[]);
            rpass.set_bind_group(1, &self.vertex_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(iter::once(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("water_rtx");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "water",
    image_path: "/examples/src/water/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default()
        .downlevel_flags(wgpu::DownlevelFlags::READ_ONLY_DEPTH_STENCIL),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.01)],
    _phantom: std::marker::PhantomData::<Example>,
};
