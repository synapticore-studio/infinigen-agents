# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: text & art generators
# - Stamatis Alexandropoulos: image postprocessing effects
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=hpamCaVrbTk by Joey Carlino

import colorsys
import inspect
import io
import logging
import os

import bpy
import numpy as np
from numpy.random import rand, uniform
from PIL import Image

from infinigen import repo_root
from infinigen.assets.utils.decorate import decimate
from infinigen.assets.utils.misc import generate_text
from infinigen.assets.utils.object import new_plane
from infinigen.assets.utils.uv import compute_uv_direction
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.math import clip_gaussian
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg

logger = logging.getLogger(__name__)

font_dir = repo_root() / "infinigen/assets/fonts"
font_names = [_.replace("_", " ") for _ in os.listdir(font_dir)]

def _init_fonts():
    import matplotlib.font_manager
    for f in matplotlib.font_manager.findSystemFonts([font_dir]):
        matplotlib.font_manager.fontManager.addfont(f)
    all_fonts = matplotlib.font_manager.get_font_names()
    assert [f in all_fonts for f in font_names]


class Text:
    default_font_name = "DejaVu Sans"
    patch_fns = (
        "weighted_choice",
        (2, "Circle"),
        (4, "Rectangle"),
        (1, "Wedge"),
        (1, "RegularPolygon"),
        (1, "Ellipse"),
        (2, "Arrow"),
        (2, "FancyBboxPatch"),
    )
    hatches = {"/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"}
    font_weights = ["normal", "bold", "heavy"]
    font_styles = ["normal", "italic", "oblique"]

    def __init__(self, has_barcode=True, emission=0):
        self.size = 4
        self.dpi = 100
        self.colormap = (
            self.build_sequential_colormap()
            if uniform() < 0.5
            else self.build_diverging_colormap()
        )
        self.white_chance = 0.03
        self.black_chance = 0.05

        self.n_patches = np.random.randint(5, 8)
        self.force_horizontal = uniform() < 0.75

        self.n_texts = np.random.randint(2, 4)

        self.n_barcodes = 1 if has_barcode and uniform() < 0.5 else 0
        self.barcode_scale = uniform(0.3, 0.6)
        self.barcode_length = np.random.randint(25, 40)
        self.barcode_aspect = log_uniform(1.5, 3)

        self.emission = emission

    @staticmethod
    def build_diverging_colormap():
        count = 20
        hue = (uniform() + np.linspace(0, 0.5, count)) % 1
        mid = uniform(0.6, 0.8)
        lightness = np.concatenate(
            [
                np.linspace(uniform(0.1, 0.3), mid, count // 2),
                np.linspace(mid, uniform(0.1, 0.3), count // 2),
            ]
        )
        saturation = np.concatenate(
            [np.linspace(1, 0.5, count // 2), np.linspace(0.5, 1, count // 2)]
        )

        # TODO hack
        saturation *= uniform(0, 1)
        lightness *= uniform(0.5, 1)

        return np.array(
            [
                colorsys.hls_to_rgb(h, l, s)
                for h, l, s in zip(hue, lightness, saturation)
            ]
        )

    @staticmethod
    def build_sequential_colormap():
        count = 20
        hue = (uniform() + np.linspace(0, 0.5, count)) % 1
        lightness = np.linspace(uniform(0.0), uniform(0.6, 0.8), count)
        saturation = np.concatenate(
            [np.linspace(1, 0.5, count // 2), np.linspace(0.5, 1, count // 2)]
        )

        # TODO hack
        saturation *= uniform(0, 1)
        lightness *= uniform(0.5, 1)

        return np.array(
            [
                colorsys.hls_to_rgb(h, l, s)
                for h, l, s in zip(hue, lightness, saturation)
            ]
        )

    @property
    def random_color(self):
        r = uniform()
        if r < self.white_chance:
            return np.array([1, 1, 1])
        elif r < self.white_chance + self.black_chance:
            return np.array([0, 0, 0])
        else:
            return self.colormap[np.random.randint(len(self.colormap))]

    @property
    def random_colors(self):
        while True:
            c, d = self.random_color, self.random_color
            if np.abs(c - d).sum() > 0.2:
                return c, d

    def build_image(self, bbox):
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.kaleido.scope.default_format = "png"
        
        # Initialize fonts
        _init_fonts()
        
        # Create figure with Plotly
        fig = go.Figure()
        
        # Set background color
        bg_color = self.random_color
        fig.update_layout(
            width=self.size * self.dpi,
            height=self.size * self.dpi,
            plot_bgcolor=f'rgb({int(bg_color[0]*255)}, {int(bg_color[1]*255)}, {int(bg_color[2]*255)})',
            paper_bgcolor=f'rgb({int(bg_color[0]*255)}, {int(bg_color[1]*255)}, {int(bg_color[2]*255)})',
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1])
        )
        
        # Get locations for elements
        locs = self.get_locs(bbox, self.n_patches + self.n_texts + self.n_barcodes)
        
        # Add elements (simplified for now)
        self.add_plotly_elements(fig, locs, bbox)
        
        # Convert to image
        img_bytes = fig.to_image(format="png")
        
        # Create Blender image
        import io

        import numpy as np
        from PIL import Image
        
        size = self.size * self.dpi
        image = bpy.data.images.new("text_texture", width=size, height=size, alpha=True)
        data = np.asarray(Image.open(io.BytesIO(img_bytes)), dtype=np.float32)[::-1, :] / 255.0
        image.pixels.foreach_set(data.ravel())
        image.pack()
        
        return image

    def add_plotly_elements(self, fig, locs, bbox):
        """Add elements to Plotly figure instead of matplotlib"""
        import plotly.graph_objects as go
        
        # Add simple shapes for now (can be expanded later)
        for i, (x, y) in enumerate(locs):
            if i < self.n_patches:
                # Add simple circles as placeholders
                fig.add_shape(
                    type="circle",
                    x0=x-0.05, y0=y-0.05, x1=x+0.05, y1=y+0.05,
                    fillcolor=f'rgb({int(self.random_color[0]*255)}, {int(self.random_color[1]*255)}, {int(self.random_color[2]*255)})',
                    line=dict(color="black", width=1)
                )
            elif i < self.n_patches + self.n_texts:
                # Add text annotations
                fig.add_annotation(
                    x=x, y=y,
                    text=f"Text{i}",
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            else:
                # Add barcode placeholders
                fig.add_shape(
                    type="rect",
                    x0=x-0.02, y0=y-0.01, x1=x+0.02, y1=y+0.01,
                    fillcolor="black",
                    line=dict(color="black", width=1)
                )

    @staticmethod
    def loc_uniform(min_, max_, size=None):
        ratio = 0.1
        return uniform(
            min_ + ratio * (max_ - min_), min_ + (1 - ratio) * (max_ - min_), size
        )

    @staticmethod
    def scale_uniform(min_, max_):
        return (max_ - min_) * log_uniform(0.2, 0.8)

    def get_locs(self, bbox, n):
        m = 8 * n
        x, y = (
            self.loc_uniform(bbox[0], bbox[1], m),
            self.loc_uniform(bbox[2], bbox[3], m),
        )
        return decimate(np.stack([x, y], -1), n)

    def make_shader_func(self, bbox):
        assert bbox[1] - bbox[0] > 0.001 and bbox[3] - bbox[2] > 0.001
        image = self.build_image(bbox)

        def shader_text(nw: NodeWrangler, **kwargs):
            uv_map = nw.new_node(Nodes.UVMap)

            reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": uv_map})

            voronoi_texture = nw.new_node(
                Nodes.VoronoiTexture, input_kwargs={"Vector": reroute, "Scale": 60.0000}
            )

            voronoi_texture_1 = nw.new_node(
                Nodes.VoronoiTexture, input_kwargs={"Vector": reroute, "Scale": 60.0000}
            )

            mix = nw.new_node(
                Nodes.Mix,
                input_kwargs={
                    6: voronoi_texture.outputs["Position"],
                    7: voronoi_texture_1.outputs["Position"],
                },
                attrs={"data_type": "RGBA"},
            )

            musgrave_texture = nw.new_node(
                Nodes.MusgraveTexture,
                input_kwargs={"Vector": reroute, "Detail": 5.6000, "Dimension": 1.4000},
            )

            noise_texture_1 = nw.new_node(
                Nodes.NoiseTexture,
                input_kwargs={
                    "Vector": reroute,
                    "Scale": 35.4000,
                    "Detail": 3.3000,
                    "Roughness": 1.0000,
                },
            )

            mix_3 = nw.new_node(
                Nodes.Mix,
                input_kwargs={
                    0: uniform(0.2, 1.0),
                    6: musgrave_texture,
                    7: noise_texture_1.outputs["Color"],
                },
                attrs={"data_type": "RGBA"},
            )

            mix_1 = nw.new_node(
                Nodes.Mix,
                input_kwargs={0: 0.0417, 6: mix.outputs[2], 7: mix_3.outputs[2]},
                attrs={"data_type": "RGBA"},
            )

            if rand() < 0.5:
                mix_2 = nw.new_node(
                    Nodes.Mix,
                    input_kwargs={0: uniform(0, 0.4), 6: mix_1.outputs[2], 7: uv_map},
                    attrs={"data_type": "RGBA"},
                )
            else:
                mix_2 = nw.new_node(
                    Nodes.Mix,
                    input_kwargs={0: 1.0, 6: mix_1.outputs[2], 7: uv_map},
                    attrs={"data_type": "RGBA"},
                )
            
            color = nw.new_node(
                Nodes.ShaderImageTexture, [mix_2], attrs={"image": image}
            ).outputs[0]
            roughness = nw.new_node(Nodes.NoiseTexture)
            if self.emission > 0:
                emission = color
                color = (0.05, 0.05, 0.05, 1)
                roughness = 0.05
            else:
                emission = None
            principled_bsdf = nw.new_node(
                Nodes.PrincipledBSDF,
                input_kwargs={
                    "Base Color": color,
                    "Roughness": roughness,
                    "Metallic": uniform(0, 0.5),
                    "Specular IOR Level": uniform(0, 0.2),
                    "Emission Color": emission,
                    "Emission Strength": self.emission,
                },
            )
            nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf})

        return shader_text

    # def apply(self, obj, selection=None, bbox=(0, 1, 0, 1), **kwargs):
    #     common.apply(obj, self.make_shader_func(bbox), selection, **kwargs)
    def generate(self, selection=None, bbox=(0, 1, 0, 1), **kwargs):
        return surface.shaderfunc_to_material(self.make_shader_func(bbox))

    __call__ = generate


class TextGeneral:
    def generate(
        self, selection=None, bbox=(0, 1, 0, 1), has_barcode=True, emission=0, **kwargs
    ):
        return Text(has_barcode, emission).generate(selection, bbox, **kwargs)

    __call__ = generate


def make_sphere():
    obj = new_plane()
    obj.rotation_euler[0] = np.pi / 2
    butil.apply_transform(obj)
    compute_uv_direction(obj, "x", "z")
    return obj


class TextNoBarcode:
    def generate(self, selection=None, bbox=(0, 1, 0, 1), emission=0, **kwargs):
        return Text(False, emission).generate(selection, bbox, **kwargs)

    __call__ = generate
