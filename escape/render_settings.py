"""Cycles settings shared by previews and the final render."""
import bpy


def apply(scene, width=1280, height=536, samples=24, motion_blur=True, preview=False):
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.device = "CPU"
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.05 if not preview else 0.1
    c.adaptive_min_samples = min(8, samples)
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    try:
        c.denoising_prefilter = "FAST" if preview else "ACCURATE"
        c.denoising_quality = "BALANCED" if preview else "HIGH"
    except (TypeError, AttributeError):
        pass
    c.max_bounces = 4
    c.diffuse_bounces = 1
    c.glossy_bounces = 1
    c.transmission_bounces = 1
    c.volume_bounces = 0
    c.transparent_max_bounces = 12
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.sample_clamp_indirect = 6.0
    c.use_light_tree = False
    scene.render.film_transparent = False
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = motion_blur
    scene.render.motion_blur_shutter = 0.5
    scene.render.use_persistent_data = True
    scene.cycles_curves.shape = "RIBBONS"
    scene.cycles_curves.subdivisions = 2
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.fps = 24
