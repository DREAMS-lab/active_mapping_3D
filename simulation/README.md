# Simulation Setup

Copy these files into your PX4-Autopilot installation before running.

## Airframe
```bash
cp simulation/airframes/4022_gz_px4_gsplat \
   ~/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/
```

## Gazebo Models
```bash
cp -r simulation/models/gimbal_rgbd ~/PX4-Autopilot/Tools/simulation/gz/models/
cp -r simulation/models/px4_gsplat ~/PX4-Autopilot/Tools/simulation/gz/models/
cp -r simulation/models/rock ~/PX4-Autopilot/Tools/simulation/gz/models/

# Lunar rock models (also in src/gsplat/models/, need to be in PX4 too)
cp -r src/gsplat/models/lunar_sample_15016 ~/PX4-Autopilot/Tools/simulation/gz/models/
cp -r src/gsplat/models/lunar_sample_65035 ~/PX4-Autopilot/Tools/simulation/gz/models/
```

## World SDF
```bash
cp simulation/worlds/sample_15016.sdf \
   ~/PX4-Autopilot/Tools/simulation/gz/worlds/
```

**Important:** Edit `sample_15016.sdf` and update the texture paths (lines 37-39) to match your workspace location:
```xml
<albedo_map>file:///YOUR/PATH/simulation/textures/moon_surface/textures/Diffuse.jpeg</albedo_map>
<normal_map>file:///YOUR/PATH/simulation/textures/moon_surface/textures/NormalMap.jpeg</normal_map>
<roughness_map>file:///YOUR/PATH/simulation/textures/moon_surface/textures/Roughness.jpeg</roughness_map>
```

## Moon Surface Textures
Keep the `simulation/textures/` folder in place. The world SDF references these textures by absolute path.
