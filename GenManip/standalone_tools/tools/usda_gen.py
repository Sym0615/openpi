import argparse
import os

parser = argparse.ArgumentParser(description="Generate USDZ files from USD files.")
parser.add_argument(
    "-f",
    "--directory_path",
    type=str,
    help="Path to the directory containing USD files",
)
parser.add_argument(
    "-r",
    "--recursive",
    default=False,
    action="store_true",
    help="Whether to generate usda in a recursive way",
)
args = parser.parse_args()

usda_template = """#usda 1.0
(
    customLayerData = {{
        dictionary omni_layer = {{
            dictionary locked = {{
            }}
            dictionary muteness = {{
            }}
        }}
        dictionary renderSettings = {{
            float3 "rtx:debugView:pixelDebug:textColor" = (0, 1e18, 0)
            float3 "rtx:fog:fogColor" = (0.75, 0.75, 0.75)
            float3 "rtx:index:backgroundColor" = (0, 0, 0)
            float3 "rtx:index:regionOfInterestMax" = (0, 0, 0)
            float3 "rtx:index:regionOfInterestMin" = (0, 0, 0)
            float3 "rtx:post:backgroundZeroAlpha:backgroundDefaultColor" = (0, 0, 0)
            float3 "rtx:post:colorcorr:contrast" = (1, 1, 1)
            float3 "rtx:post:colorcorr:gain" = (1, 1, 1)
            float3 "rtx:post:colorcorr:gamma" = (1, 1, 1)
            float3 "rtx:post:colorcorr:offset" = (0, 0, 0)
            float3 "rtx:post:colorcorr:saturation" = (1, 1, 1)
            float3 "rtx:post:colorgrad:blackpoint" = (0, 0, 0)
            float3 "rtx:post:colorgrad:contrast" = (1, 1, 1)
            float3 "rtx:post:colorgrad:gain" = (1, 1, 1)
            float3 "rtx:post:colorgrad:gamma" = (1, 1, 1)
            float3 "rtx:post:colorgrad:lift" = (0, 0, 0)
            float3 "rtx:post:colorgrad:multiply" = (1, 1, 1)
            float3 "rtx:post:colorgrad:offset" = (0, 0, 0)
            float3 "rtx:post:colorgrad:whitepoint" = (1, 1, 1)
            float3 "rtx:post:lensDistortion:lensFocalLengthArray" = (10, 30, 50)
            float3 "rtx:post:lensFlares:anisoFlareFalloffX" = (450, 475, 500)
            float3 "rtx:post:lensFlares:anisoFlareFalloffY" = (10, 10, 10)
            float3 "rtx:post:lensFlares:cutoffPoint" = (2, 2, 2)
            float3 "rtx:post:lensFlares:haloFlareFalloff" = (10, 10, 10)
            float3 "rtx:post:lensFlares:haloFlareRadius" = (75, 75, 75)
            float3 "rtx:post:lensFlares:isotropicFlareFalloff" = (50, 50, 50)
            float3 "rtx:post:tonemap:whitepoint" = (1, 1, 1)
            float3 "rtx:raytracing:indexdirect:svoBrickSize" = (32, 32, 32)
            float3 "rtx:raytracing:inscattering:singleScatteringAlbedo" = (0.9, 0.9, 0.9)
            float3 "rtx:raytracing:inscattering:transmittanceColor" = (0.5, 0.5, 0.5)
            float3 "rtx:sceneDb:ambientLightColor" = (0.1, 0.1, 0.1)
            double "rtx:translucency:worldEps" = 0.005
        }}
    }}
    defaultPrim = "World"
    endTimeCode = 1000000
    metersPerUnit = 1.0
    startTimeCode = 0
    timeCodesPerSecond = 60
    upAxis = "Z"
)

over "Render" (
    hide_in_stage_window = true
)
{{
}}

def Xform "World"
{{
    def "_{uid}" (
        prepend payload = @./{absolute_usd_path}@
    )
    {{
        float3 xformOp:rotateXYZ = (0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }}
}}

def Xform "Environment"
{{
    double3 xformOp:rotateXYZ = (0, 0, 0)
    double3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
}}
"""

directory_path = args.directory_path

if args.recursive:
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".usd"):
                uid = file[:-4]
                usda_content = usda_template.format(uid=uid, absolute_usd_path=file)
                usda_filename = f"{uid}.usda"
                usda_filepath = os.path.join(root, usda_filename)
                with open(usda_filepath, "w", encoding="utf-8") as usda_file:
                    usda_file.write(usda_content)
                print(f"Generated {usda_filename}")
else:
    for filename in os.listdir(directory_path):
        if filename.endswith(".usd"):
            uid = filename[:-4]
            usda_content = usda_template.format(uid=uid, absolute_usd_path=filename)
            usda_filename = f"{uid}.usda"
            usda_filepath = os.path.join(directory_path, usda_filename)
            with open(usda_filepath, "w", encoding="utf-8") as usda_file:
                usda_file.write(usda_content)
            print(f"Generated {usda_filename}")
print("All .usda files have been generated.")
