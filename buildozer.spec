[app]

title = Skyline Geolocation

package.name = skylinegeolocation

package.domain = org.test

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx

requirements = python3.12,kivy,opencv,numpy,scipy,plyer,onnxruntime

orientation = portrait

permissions = CAMERA

android.api = 33
android.minapi = 24
android.ndk = 28c
android.ndk_api = 24
android.private_storage = True
android.accept_ndk_license = False
android.enable_androidx = True
android.archs = arm64-v8a
android.copy_libs = 1
android.add_args = --kivy-ios.use_frameworks=1

version = 0.1

[buildozer]

log_level = 2
warn_on_root = 1
build_dir = .buildozer
bin_dir = bin
