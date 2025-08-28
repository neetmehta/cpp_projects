# How to build with cmake

## Build OpenCV with cmake

### Release

```bash
cd libs/opencv
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_LIST="core,highgui,imgproc,imgcodecs"
cmake --build build --target INSTALL --config Release
```

### Debug

```bash
cd libs/opencv
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_LIST="core,highgui,imgproc,imgcodecs,features2d,flann,calib3d"
cmake --build build --target INSTALL --config Debug
```

- cmake --build build --target INSTALL

    - This tells CMake:

    - “Go into build/ and build the special target INSTALL.”

    - It runs the build tool (MSBuild, Ninja, Make, etc.), and when it finishes, it triggers the install step (copy headers, libs, binaries into CMAKE_INSTALL_PREFIX).

    - It’s equivalent to doing:

```bash
cmake --build build
cmake --build build --target install
```

👉 Rule of thumb:

During development, use cmake --build build --target INSTALL (so build + install happens in one step).

For packaging or after building manually, use cmake --install build --config Release (only copies files to CMAKE_INSTALL_PREFIX).
