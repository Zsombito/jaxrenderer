
import functools
from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jp
import numpy as onp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image

import brax
from brax import base, envs, math
from brax.envs import humanoid
from brax.io import model
from brax.training.acme import running_statistics, specs
import brax.training.agents.ppo.networks as ppo_networks
import brax.training.agents.sac.networks as sac_networks

import trimesh

from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import Model as RendererMesh
from renderer import ModelObject as Instance
from renderer import ShadowParameters as Shadow
from renderer import Renderer, UpAxis, create_capsule, create_cube, transpose_for_display
import time



canvas_width: int = 520 #@param {type:"integer"}
canvas_height: int = 360 #@param {type:"integer"}

def grid(grid_size: int, color) -> jp.ndarray:
  grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.single)
  grid[:, :] = onp.array(color) / 255.0
  grid[0] = onp.zeros((grid_size, 3), dtype=onp.single)
  # to reverse texture along y direction
  grid[:, -1] = onp.zeros((grid_size, 3), dtype=onp.single)
  return jp.asarray(grid)

_GROUND: jp.ndarray = grid(100, [200, 200, 200])

class Obj(NamedTuple):
  """An object to be rendered in the scene.

  Assume the system is unchanged throughout the rendering.

  col is accessed from the batched geoms `sys.geoms`, representing one geom.
  """
  instance: Instance
  """An instance to be rendered in the scene, defined by jaxrenderer."""
  link_idx: int
  """col.link_idx if col.link_idx is not None else -1"""
  off: jp.ndarray
  """col.transform.rot"""
  rot: jp.ndarray
  """col.transform.rot"""

def _build_objects(sys: brax.System) -> list[Obj]:
    """Converts a brax System to a list of Obj."""
    objs: list[Obj] = []

    for geom_idx in range(sys.ngeom):
      tex = sys.geom_rgba[geom_idx][:3].reshape((1, 1, 3))
      # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
      specular_map = jax.lax.full(tex.shape[:2], 2.0)
      geom_type = sys.geom_type[geom_idx]
      if geom_type == 6: #Box
          model = create_cube(sys.geom_size[geom_idx][0], tex, specular_map)
      elif geom_type == 2: #Sphere
          model = create_capsule(sys.geom_size[geom_idx][0], 0, 2, tex, specular_map)
      elif geom_type == 3: #Capsule
          if sys.geom_size[geom_idx].shape[0] == 1:
              model = create_capsule(sys.geom_size[geom_idx][0], 1 * sys.geom_size[geom_idx][0], 2, tex, specular_map)
          else:
              model = create_capsule(sys.geom_size[geom_idx][0], sys.geom_size[geom_idx][1], 2, tex, specular_map)
      else:
          continue

      instance = Instance(model=model)
      obj = Obj(instance=instance, link_idx=sys.geom_bodyid[geom_idx] - 1, off=sys.geom_pos[geom_idx], rot=sys.geom_quat[geom_idx])
      objs.append(obj)

    return objs


def _with_state(objs: Iterable[Obj], x: brax.Transform) -> list[Instance]:
  """x must has at least 1 element. This can be ensured by calling
    `x.concatenate(base.Transform.zero((1,)))`. x is `state.x`.

    This function does not modify any inputs, rather, it produces a new list of
    `Instance`s.
  """
  instances: list[Instance] = []
  for obj in objs:
    i = obj.link_idx
    pos = x.pos[i] + math.rotate(obj.off, x.rot[i])
    rot = math.quat_mul(x.rot[i], obj.rot)
    instance = obj.instance
    instance = instance.replace_with_position(pos)
    instance = instance.replace_with_orientation(rot)
    instances.append(instance)

  return instances

def get_camera(
    width: int = canvas_width,
    height: int = canvas_height,
) -> Camera:
  """Gets camera object."""
  eye, up = jp.array([10, 10, 10]), jp.array([0,0,1])
  hfov = 58.0
  vfov = hfov * height / width
  target = jp.zeros(3, float)
  camera = Camera(
      viewWidth=width,
      viewHeight=height,
      position=eye,
      target=target,
      up=up,
      hfov=hfov,
      vfov=vfov,
  )

  return camera

@jax.default_matmul_precision("float32")
def render_instances(
  instances: list[Instance],
  width: int,
  height: int,
  camera: Camera,
  light: Optional[Light] = None,
  shadow: Optional[Shadow] = None,
  camera_target: Optional[jp.ndarray] = None,
  enable_shadow: bool = False,
) -> jp.ndarray:
  """Renders an RGB array of sequence of instances.

  Rendered result is not transposed with `transpose_for_display`; it is in
  floating numbers in [0, 1], not `uint8` in [0, 255].
  """
  if light is None:
    direction = jp.array([0, 0, 1])
    light = Light(
        direction=direction,
        ambient=0.8,
        diffuse=0.8,
        specular=0.6,
    )
  if shadow is None and enable_shadow:
    assert camera_target is not None, 'camera_target is None'
    shadow = Shadow(centre=camera_target)
  elif not enable_shadow:
    shadow = None

  img = Renderer.get_camera_image(
    objects=instances,
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow,
    loop_unroll=100
  )
  arr = jax.lax.clamp(0., img, 1.)

  return arr

human = humanoid.Humanoid()


obj = _build_objects(human.sys)
print("Building done")


import pickle

with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)

frames = []
camera = get_camera(canvas_width, canvas_height)
avarage = 0

for i in range(120):
    instances = _with_state(obj, states[i].pipeline_state.x)
    start = time.time_ns()
    pixels = jax.block_until_ready(render_instances(instances, canvas_width, canvas_height, camera))
    end = time.time_ns()
    frames.append(pixels.astype("uint8"))
    print(f"Frame{i} took: {(end - start) / 1000 / 1000} ms")
    if i != 0:
        avarage += end - start

print(f"On avarage frames took: {avarage / 119 / 1000 / 1000} ms")

print("Making giff")
import imageio
imageio.mimsave('output.gif', frames)
