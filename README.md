# Blender DXM Addon
Original source code : https://github.com/Smileynator/blender-mdb-addon<br><br>
Blender model importer for Earth Defense Force 2017 (.dxm) files.<br>
Most of the EDF2017 portable models can also be imported.
## Download
https://github.com/stafern/blender-dxm-addon/archive/refs/heads/main.zip
## Features
* Import of (.dxm) files(mesh and bone only)<br>
There are currently no plans to support the Export.
## Usage
Model files (.dxm) and textures (.dds) must be prepared in advance.<br>
These files are stored in the game in a compressed format called (.dxb) or (.dxm).<br>
(.dxb) or (.dxm) can be extracted with EDF2017parser.7z. <br>

## Known Issues
* The size of Player02.Dxm seems too large.
* UfoCarrier_02.Dxm doesn't seem to load properly.
* booster_B(G,R,Y).Dxm can be loaded, but the shape appears wrong.
* In models consisting of multiple meshes, positional relationships are not reflected (e.g., Player01.Dxm visor).
* Textures are not automatically reflected.<br>
I think it will load, so please reflect this manually.
![tex](https://github.com/user-attachments/assets/895d3fdd-4e74-4544-a139-b0759efc11db)

## Other
* Documentation of EDF2017 file format<br>
https://github.com/KCreator/Earth-Defence-Force-Documentation/wiki/EDF3(EDF2017)-DXB-and-DXM-Format<br>
https://github.com/KCreator/Earth-Defence-Force-Documentation/wiki/EDF3(EDF2017)-DXM(model)-Format<br>
