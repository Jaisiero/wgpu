use crate::{
    command::CommandEncoderError,
    device::DeviceError,
    id::{BlasId, BufferId, TlasId},
    resource::CreateBufferError,
};
use std::sync::Arc;
/// Ray tracing
/// Major missing optimizations (no api surface changes needed):
/// - use custom tracker to track build state
/// - no forced rebuilt (build mode deduction)
/// - lazy instance buffer allocation
/// - maybe share scratch and instance staging buffer allocation
/// - partial instance buffer uploads (api surface already designed with this in mind)
/// - ([non performance] extract function in build (rust function extraction with guards is a pain))
use std::{num::NonZeroU64, slice};

use crate::resource::{Blas, ResourceErrorIdent, Tlas};
use thiserror::Error;
use wgt::BufferAddress;

#[derive(Clone, Debug, Error)]
pub enum CreateBlasError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    CreateBufferError(#[from] CreateBufferError),
    #[error(
        "Only one of 'index_count' and 'index_format' was provided (either provide both or none)"
    )]
    MissingIndexData,
    #[error("To use flag ALLOW_RAY_HIT_VERTEX_RETURN device feature RAY_HIT_VERTEX_RETURN must be used too")]
    MissingVertexReturnFeature,
}

#[derive(Clone, Debug, Error)]
pub enum CreateTlasError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    CreateBufferError(#[from] CreateBufferError),
    #[error("To use flag ALLOW_RAY_HIT_VERTEX_RETURN device feature RAY_HIT_VERTEX_RETURN must be used too")]
    MissingVertexReturnFeature,
    #[error("Unimplemented Tlas error: this error is not yet implemented")]
    Unimplemented,
}

/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
pub enum BuildAccelerationStructureError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),

    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error("BufferId is invalid or destroyed")]
    InvalidBufferId,

    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(ResourceErrorIdent),

    #[error("Buffer {0:?} is missing `BLAS_INPUT` usage flag")]
    MissingBlasInputUsageFlag(ResourceErrorIdent),

    #[error(
        "Buffer {0:?} size is insufficient for provided size information (size: {1}, required: {2}"
    )]
    InsufficientBufferSize(ResourceErrorIdent, u64, u64),

    #[error("Buffer {0:?} associated offset doesn't align with the index type")]
    UnalignedIndexBufferOffset(ResourceErrorIdent),

    #[error("Buffer {0:?} associated offset is unaligned")]
    UnalignedTransformBufferOffset(ResourceErrorIdent),

    #[error("Buffer {0:?} associated index count not divisible by 3 (count: {1}")]
    InvalidIndexCount(ResourceErrorIdent, u32),

    #[error("Buffer {0:?} associated data contains None")]
    MissingAssociatedData(ResourceErrorIdent),

    #[error(
        "Blas {0:?} build sizes to may be greater than the descriptor at build time specified"
    )]
    IncompatibleBlasBuildSizes(ResourceErrorIdent),

    #[error("Blas {0:?} build sizes require index buffer but none was provided")]
    MissingIndexBuffer(ResourceErrorIdent),

    #[error("BlasId is invalid")]
    InvalidBlasId,

    #[error("Blas {0:?} is destroyed")]
    InvalidBlas(ResourceErrorIdent),

    #[error(
        "Tlas {0:?} an associated instances contains an invalid custom index (more than 24bits)"
    )]
    TlasInvalidCustomIndex(ResourceErrorIdent),

    #[error(
        "Tlas {0:?} has {1} active instances but only {2} are allowed as specified by the descriptor at creation"
    )]
    TlasInstanceCountExceeded(ResourceErrorIdent, u32, u32),

    #[error("BlasId is invalid or destroyed (for instance)")]
    InvalidBlasIdForInstance,

    #[error("Blas {0:?} is invalid or destroyed (for instance)")]
    InvalidBlasForInstance(ResourceErrorIdent),

    #[error("TlasId is invalid or destroyed")]
    InvalidTlasId,

    #[error("Tlas {0:?} is invalid or destroyed")]
    InvalidTlas(ResourceErrorIdent),

    #[error("Buffer {0:?} is missing `TLAS_INPUT` usage flag")]
    MissingTlasInputUsageFlag(ResourceErrorIdent),

    #[error("Blas {0:?} was missing flag ALLOW_RAY_HIT_VERTEX_RETURN while tlas {1:?} had flag")]
    MissingBlasVertexReturn(BlasId, TlasId),
}

#[derive(Clone, Debug, Error)]
pub enum ValidateBlasActionsError {
    #[error("BlasId is invalid or destroyed")]
    InvalidBlas,

    #[error("Blas {0:?} is used before it is build")]
    UsedUnbuilt(ResourceErrorIdent),
}

#[derive(Clone, Debug, Error)]
pub enum ValidateTlasActionsError {
    #[error("Tlas {0:?} is invalid or destroyed")]
    InvalidTlas(ResourceErrorIdent),

    #[error("Tlas {0:?} is used before it is build")]
    UsedUnbuilt(ResourceErrorIdent),

    #[error("Blas {0:?} is used before it is build (in Tlas {1:?})")]
    UsedUnbuiltBlas(ResourceErrorIdent, ResourceErrorIdent),

    #[error("BlasId is invalid or destroyed (in Tlas {0:?})")]
    InvalidBlasId(ResourceErrorIdent),

    #[error("Blas {0:?} is newer than the containing Tlas {1:?}")]
    BlasNewerThenTlas(ResourceErrorIdent, ResourceErrorIdent),
}

#[derive(Debug)]
pub struct BlasTriangleGeometry<'a> {
    pub size: &'a wgt::BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: BufferId,
    pub index_buffer: Option<BufferId>,
    pub transform_buffer: Option<BufferId>,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub index_buffer_offset: Option<BufferAddress>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

#[derive(Debug)]
pub struct BlasProceduralGeometry<'a> {
    pub size: &'a wgt::BlasProceduralGeometrySizeDescriptor,
    pub bounding_box_buffer: BufferId,
    pub bounding_box_buffer_offset: BufferAddress,
    pub bounding_box_stride: BufferAddress,
}

/// Enum to encapsulate different geometry types in a BLAS (Bottom-Level Acceleration Structure).
#[derive(Debug)]
pub enum BlasGeometry<'a> {
    /// Variant for triangle geometry.
    Triangle(BlasTriangleGeometry<'a>),
    /// Variant for procedural (e.g., AABB) geometry.
    Procedural(BlasProceduralGeometry<'a>),
}

pub enum BlasGeometries<'a> {
    TriangleGeometries(Box<dyn Iterator<Item = BlasTriangleGeometry<'a>> + 'a>),
    ProceduralGeometries(Box<dyn Iterator<Item = BlasProceduralGeometry<'a>> + 'a>),
}

pub struct BlasBuildEntry<'a> {
    pub blas_id: BlasId,
    pub geometries: BlasGeometries<'a>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TlasBuildEntry {
    pub tlas_id: TlasId,
    pub instance_buffer_id: BufferId,
    pub instance_count: u32,
}

#[derive(Debug)]
pub struct TlasInstance<'a> {
    pub blas_id: BlasId,
    pub transform: &'a [f32; 12],
    pub custom_index: u32,
    pub mask: u8,
}

pub struct TlasPackage<'a> {
    pub tlas_id: TlasId,
    pub instances: Box<dyn Iterator<Item = Option<TlasInstance<'a>>> + 'a>,
    pub lowest_unmodified: u32,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum BlasActionKind {
    Build(NonZeroU64),
    Use,
}

#[derive(Debug, Clone)]
pub(crate) enum TlasActionKind {
    Build {
        build_index: NonZeroU64,
        dependencies: Vec<Arc<Blas>>,
    },
    Use,
}

#[derive(Debug, Clone)]
pub(crate) struct BlasAction {
    pub blas: Arc<Blas>,
    pub kind: BlasActionKind,
}

#[derive(Debug, Clone)]
pub(crate) struct TlasAction {
    pub tlas: Arc<Tlas>,
    pub kind: TlasActionKind,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceBlasTriangleGeometry {
    pub size: wgt::BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: BufferId,
    pub index_buffer: Option<BufferId>,
    pub transform_buffer: Option<BufferId>,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub index_buffer_offset: Option<BufferAddress>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceBlasProceduralGeometry {
    pub size: wgt::BlasProceduralGeometrySizeDescriptor,
    pub bounding_box_buffer: BufferId,
    pub bounding_box_buffer_offset: BufferAddress,
    pub bounding_box_stride: BufferAddress,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TraceBlasGeometries {
    TriangleGeometries(Vec<TraceBlasTriangleGeometry>),
    ProceduralGeometries(Vec<TraceBlasProceduralGeometry>),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceBlasBuildEntry {
    pub blas_id: BlasId,
    pub geometries: TraceBlasGeometries,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceTlasInstance {
    pub blas_id: BlasId,
    pub transform: [f32; 12],
    pub custom_index: u32,
    pub mask: u8,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceTlasPackage {
    pub tlas_id: TlasId,
    pub instances: Vec<Option<TraceTlasInstance>>,
    pub lowest_unmodified: u32,
}

pub(crate) fn get_raw_tlas_instance_size() -> usize {
    // TODO: this should be provided by the backend
    64
}

#[derive(Clone)]
#[repr(C)]
struct RawTlasInstance {
    transform: [f32; 12],
    custom_index_and_mask: u32,
    shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

pub(crate) fn tlas_instance_into_bytes(instance: &TlasInstance, blas_address: u64) -> Vec<u8> {
    // TODO: get the device to do this
    const MAX_U24: u32 = (1u32 << 24u32) - 1u32;
    let temp = RawTlasInstance {
        transform: *instance.transform,
        custom_index_and_mask: (instance.custom_index & MAX_U24) | (u32::from(instance.mask) << 24),
        shader_binding_table_record_offset_and_flags: 0,
        acceleration_structure_reference: blas_address,
    };
    let temp: *const _ = &temp;
    unsafe {
        slice::from_raw_parts::<u8>(temp.cast::<u8>(), std::mem::size_of::<RawTlasInstance>())
            .to_vec()
    }
}
