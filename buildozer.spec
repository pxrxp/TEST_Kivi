[app]

title = Skyline Geolocation
package.name = skylinegeolocation
package.domain = org.test

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx

# 1. Changed kivy>=2.4.0 to kivy==2.3.0
# 2. Removed cython from requirements (it is handled on the host side)
requirements = python3,kivy==2.3.0,plyer,opencv

orientation = portrait

permissions = CAMERA

android.api = 33
android.minapi = 24
android.build_tools_version = 33.0.2
android.ndk = 25b
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
