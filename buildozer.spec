[app]

title = Skyline Geolocation

package.name = skylinegeolocation

package.domain = org.test

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx

requirements = kivy==2.3.0,plyer,onnxruntime

orientation = portrait

permissions = CAMERA

android.api = 33
android.minapi = 24
android.ndk = 28c
android.ndk_api = 24
android.private_storage = True
android.accept_ndk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.copy_libs = 1
android.gradle_options = org.gradle.jvmargs=-Xmx4096m

version = 0.1

[buildozer]

log_level = 2
warn_on_root = 1
build_dir = .buildozer
bin_dir = bin

