from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, jaxtyped

from ..shader import ID, PerFragment, PerVertex, Shader
from ..geometry import Camera, to_homogeneous
from ..types import Colour, Vec2f, Vec3f, Vec4f

jax.config.update('jax_array', True)


class GouraudExtraVertexInput(NamedTuple):
    position: Vec3f  # in world space
    colour: Colour


class GouraudExtraFragmentData(NamedTuple):
    colour: Colour = jnp.array([0.0, 0.0, 0.0, 1.0])


class GouraudExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""
    canvas: Colour


class GouraudShader(Shader[GouraudExtraVertexInput, GouraudExtraFragmentData,
                           GouraudExtraMixerOutput]):
    """Gouraud Shading without lighting."""

    @staticmethod
    @jaxtyped
    @jax.jit
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: GouraudExtraVertexInput,
    ) -> tuple[PerVertex, GouraudExtraFragmentData]:
        position: Vec4f = to_homogeneous(extra.position)
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            GouraudExtraFragmentData(colour=extra.colour),
        )

    @staticmethod
    @jaxtyped
    @jax.jit
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: Bool[Array, ""],
        gl_PointCoord: Vec2f,
        extra: GouraudExtraFragmentData,
    ) -> tuple[PerFragment, GouraudExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            extra,
        )[0]
        assert isinstance(built_in, PerFragment)

        return (
            PerFragment(
                gl_FragDepth=built_in.gl_FragDepth,
                keeps=jnp.logical_and(built_in.keeps, gl_FrontFacing),
            ),
            extra,
        )