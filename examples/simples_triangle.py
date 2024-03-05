
import jax.numpy as jnp
import jax
import jax.lax as lax
from renderer import (
    CameraParameters,
    LightParameters,
    ShadowParameters,
    Camera,
    Renderer,
    Model,
    ModelObject,
    List,
    transpose_for_display,
    Texture,
    build_texture_from_PyTinyrenderer
)
from renderer.render_types import FaceIndices


width = 640
height = 480
eye = (2.0, 4.0, 1.0)
target = (0.0, 0.0, 0.0)

light: LightParameters = LightParameters()
camera: CameraParameters = CameraParameters(
    viewWidth=width,
    viewHeight=height,
    position=eye,
    target=target,
)

texture: Texture = (
    build_texture_from_PyTinyrenderer(
        (
            255,
            255,
            255,  # White
            255,
            0,
            0,  # Red
            0,
            255,
            0,  # Green
            0,
            0,
            255,  # Blue
        ),
        2,
        2,
    )
    / 255.0
)

vertices = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [100.000000, -100.000000, 0.000000],
        [100.000000, 100.000000, 0.000000],
        [-100.000000, 100.000000, 0.000000],
        [-100.000000, -100.000000, 0.000000],
    ]
)
vertices = vertices * 0.01
normals = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
    ]
)

uvs = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [1.000000, 0.000000],
        [1.000000, 1.000000],
        [0.000000, 1.000000],
        [0.000000, 0.000000],
    ]
)

indices = jnp.array([[0, 1, 2], [0, 2, 3]])  # pyright: ignore[reportUnknownMemberType]
model: Model = Model.create(
    verts=vertices,
    norms=normals,
    uvs=uvs,
    faces=indices,
    diffuse_map=texture,
)

plane_obj = ModelObject(model)

images = []

#1.
img = Renderer.get_camera_image(objects=[plane_obj], light=light, camera=camera, width=width, height=height, shadow_param=ShadowParameters())
rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)  # pyright: ignore
images.append(rgb_array)


#2.
img = Renderer.get_camera_image(objects=[plane_obj], light=light, camera=camera, width=width, height=height, shadow_param=ShadowParameters())
rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)  # pyright: ignore
images.append(rgb_array)


with jax.profiler.trace("./jax-trace", create_perfetto_link=False):
#3.
    img = Renderer.get_camera_image(objects=[plane_obj], light=light, camera=camera, width=width, height=height, shadow_param=ShadowParameters())
    rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)  # pyright: ignore
    images.append(rgb_array)



from typing import cast

import matplotlib.animation as animation
import matplotlib.figure as figure
import matplotlib.image as mimage
import matplotlib.pyplot as plt

fig: figure.Figure
fig, ax = plt.subplots()  # pyright: ignore

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims: List[List[mimage.AxesImage]] = []
for i, img in enumerate(images):
    im = cast(
        mimage.AxesImage,
        ax.imshow(  # pyright: ignore[reportUnknownMemberType]
            transpose_for_display(img),
            animated=True,
        ),
    )
    if i == 0:
        # show an initial one first
        ax.imshow(  # pyright: ignore[reportUnknownMemberType]
            transpose_for_display(img),
        )

    ims.append([im])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=500,
    blit=True,
    repeat_delay=0,
)
plt.savefig('output.png')  # pyright: ignore[reportUnknownMemberType]

##plt.show()  # pyright: ignore[reportUnknownMemberType]