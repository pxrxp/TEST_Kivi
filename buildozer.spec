[app]

title = Skyline Geolocation
package.name = skylinegeolocation
package.domain = org.skyline

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,onnx,npz,tflite,json,txt,pt

requirements = python3,kivy==2.3.0,plyer,pillow,numpy<2.0.0,opencv

orientation = sensorLandscape

permissions = CAMERA, INTERNET
android.permissions = CAMERA, INTERNET

android.features = android.hardware.camera, android.hardware.camera.autofocus

android.api = 34
android.minapi = 24
android.sdk = 34
android.build_tools_version = 34.0.0
android.ndk = 25b

android.accept_ndk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.private_storage = True

# Pin python-for-android to tag v2024.01.21 (Python 3.11.5)
p4a.fork = kivy
p4a.branch = v2024.01.21

version = 0.1

[buildozer]

log_level = 2
warn_on_root = 1
build_dir = .buildozer
bin_dir = bin