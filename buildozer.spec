[app]

title = Skyline Geolocation
package.name = skylinegeolocation
package.domain = org.skyline

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,onnx,tflite,json,txt,pt

requirements = python3,kivy==2.3.0,plyer,pillow,numpy<2.0.0

orientation = portrait

permissions = CAMERA, INTERNET, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

android.api = 33
android.minapi = 24
android.ndk = 25b
android.build_tools_version = 33.0.2

android.accept_ndk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.private_storage = True

# Pin python-for-android to official release tag v2024.01.21 (forces Python 3.11.5 on device)
p4a.fork = kivy
p4a.branch = v2024.01.21

# Disable pip build isolation so Ninja/Meson is NEVER invoked
p4a.extra_args = --no-build-isolation

version = 0.1

[buildozer]

log_level = 2
warn_on_root = 1
build_dir = .buildozer
bin_dir = bin