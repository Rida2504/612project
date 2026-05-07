# Stage 4: VR Rendering on Meta Quest

## Unity Project Setup

### Prerequisites
- Unity 2022.3 LTS (or newer)
- Meta XR SDK (from Unity Package Manager or Meta developer hub)
- UnityGaussianSplatting plugin

### Step-by-step Setup

#### 1. Create Unity Project
- Open Unity Hub → New Project → 3D (URP) template
- Name: `TextWorldVR`

#### 2. Install Meta XR SDK
- Window → Package Manager → Add package by name:
  - `com.meta.xr.sdk.all`
- Edit → Project Settings → XR Plug-in Management → Enable Oculus

#### 3. Install UnityGaussianSplatting
- Clone: https://github.com/aras-p/UnityGaussianSplatting
- Copy the `package/` folder into your Unity project's `Packages/` directory
- OR use Package Manager → Add package from git URL

#### 4. Import Splat Files
- Copy your `.ply` files from `outputs/splats/` into `Assets/Splats/`
- Create a new GameObject → Add `GaussianSplatRenderer` component
- Assign your `.ply` asset to the renderer

#### 5. VR Scene Setup
- Add XR Origin (Action-based) to scene
- Position it at the center of your splat
- Add locomotion (teleport or continuous movement)

#### 6. Build for Quest
- File → Build Settings → Android
- Switch Platform
- Player Settings:
  - Minimum API Level: 29
  - Target: ARM64
  - Graphics API: Vulkan (primary), OpenGLES3 (fallback)
- Build and Run (connect Quest via USB or use wireless ADB)

### Performance Targets
- 72+ FPS on Quest 2/3
- If FPS is low:
  - Reduce Gaussian count in OpenSplat (use fewer iterations or `--split-screen-size 0.1`)
  - Use lower SH degree (`--sh-degree 1`)
  - Downscale input images

### Alternative: ALVR Streaming
If standalone Quest performance is insufficient:
1. Install ALVR on PC (Windows/Linux with NVIDIA GPU)
2. Install ALVR client on Quest
3. Run Unity in Play mode on PC → streams to Quest
