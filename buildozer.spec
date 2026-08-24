[app]

title = Skyline Geolocation
package.name = skylinegeolocation
package.domain = org.skyline

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,onnx,tflite,json,txt,pt

# Requirements using stable NumPy 1.x
requirements = python3,kivy==2.3.0,plyer,pillow,numpy==1.26.4

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

# Disable pip build isolation so Ninja/Meson is NEVER invoked
p4a.extra_args = --no-build-isolation

version = 0.1

[buildozer]

log_level = 2
warn_on_root = 1
build_dir = .buildozer
bin_dir = bin