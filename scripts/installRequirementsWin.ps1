$ErrorActionPreference = "Stop"
Write-Host (Get-Location)

New-Item -ItemType directory -Path .\external\
Set-Location .\external\

# OpenCV install from https://learnopencv.com/install-opencv-4-on-windows/

# Invoke-WebRequest https://github.com/opencv/opencv/archive/refs/tags/3.4.16.zip -OutFile 3.4.16.zip
# Expand-Archive -LiteralPath .\3.4.16.zip -DestinationPath .\
# Set-Location .\opencv-3.4.16\
# New-Item -ItemType directory -Path .\build\
# Set-Location .\build\
# cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="..\..\" -D BUILD_LIST=core,improc,imgcodecs,dnn ..
# make -j8
# make install
git clone https://github.com/vbadenas/learnopencv.git
Set-Location .\learnopencv\InstallScripts\Windows-3
python main.py
installOpenCV_modified.bat
python modifyBatchScript.py
finalScript.bat


# Onnx install
