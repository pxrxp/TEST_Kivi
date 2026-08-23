[app]

# (str) Title of your application
title = Skyline Geolocation

# (str) Package name
package.name = skylinegeolocation

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py live
source.dir = .
source.include_exts = py,png,jpg,kv,atlas

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,opencv,numpy,scipy,plyer,tf-lite-gpu

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (list) Application permissions
# e.g. permissions = INTERNET, ACCESS_FINE_LOCATION
permissions = CAMERA

# (int) Target Android API, use 0 to use the latest available
android.api = 33

# (int) Minimum API your APK will support
android.minapi = 24

# (int) Android SDK version to use
android.sdk = 33

# (int) Android NDK version to use
android.ndk = 28c

# (int) Android NDK API to use. This is the minimum API your app will support, it should usually match android.minapi.
android.ndk_api = 24

# (bool) Use --private data storage (True) or --dir public storage (False)
android.private_storage = True

# (str) Android NDK directory (if empty, it will be automatically downloaded.)
#android.ndk_path =

# (str) Android SDK directory (if empty, it will be automatically downloaded.)
#android.sdk_path =

# (str) ANT directory (if empty, it will be automatically downloaded.)
#android.ant_path =

# (bool) If True, then skip trying to update the android sdk
# This can be useful to avoid excess Internet downloads or save time
# when an update is due and you just want to test/build your package
android.skip_updates = False

# (bool) If True, accept the 31.0.0 NDK license agreement (set to
# False if you don't have a license or if you don't want to accept the license).
android.accept_ndk_license = True

# (str) Android entry point, default is ok for Kivy-based app
#android.entrypoint = org.renpy.android.PythonActivity

# (list) List of java .jar files to add to the libs so that pyjnius can access
# their classes. Don't add jars that you do not need, since extra jars can slow
# down the build process. Wildcards are allowed, looks for *.jar in a directory
#android.add_jars = foo.jar,bar.jar,path/to/more/*.jar

# (list) List of Java files to add to the android project (can be java or a
# directory containing Java files)
#android.add_src =

# (list) Android AARs to add (currently works only with sdl2 window)
#android.add_aars =

# (list) Gradle dependencies to add to the build
#android.gradle_dependencies =

# (bool) Enable AndroidX support. Use when building with Android Gradle plugin 3.2+
android.enable_androidx = True

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.arch = arm64-v8a

# (int) Overrides automatic versionCode computation (used in build.gradle)
# This is not the same as app version_str
#android.version_code = 

# (str) Overrides automatic versionName computation (used in build.gradle)
# This is not the same as app version_str
#android.version_name = 

# (list) python-for-android specific arguments
# e.g. android.add_args = --verbose
android.add_args = --kivy-ios.use_frameworks=1

# ---- Added by CI fix ----
# (str) App version (required by buildozer)
version = 0.1.0
