import argparse
import asyncio
import os
from isaacsim import SimulationApp  # type: ignore[import-untyped]
from pathlib import Path


async def convert(in_file, out_file, load_materials=False):
    import omni.kit.asset_converter  # type: ignore

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    converter_context.use_meter_as_world_unit = True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        in_file, out_file, progress_callback, converter_context
    )
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def asset_convert(args):
    supported_file_formats = ["usd"]
    for folder in args.folders:
        local_asset_output = folder + f"/../{args.dist_folder}"
        result = omni.client.create_folder(f"{local_asset_output}")  # type: ignore[attr-defined]

    for folder in args.folders:
        print(f"\nConverting folder {folder}...")
        (result, models) = omni.client.list(folder)  # type: ignore[attr-defined]
        for i, entry in enumerate(models):
            if i >= args.max_models:
                print(f"max models ({args.max_models}) reached, exiting conversion")
                break
            model = str(entry.relative_path)
            model_name = os.path.splitext(model)[0]
            model_format = (os.path.splitext(model)[1])[1:]
            if model_format in supported_file_formats:
                input_model_path = folder + "/" + model
                converted_model_path = (
                    folder + f"/../{args.dist_folder}/" + model_name + ".glb"
                )
                if not os.path.exists(converted_model_path):
                    status = asyncio.get_event_loop().run_until_complete(
                        convert(input_model_path, converted_model_path, True)
                    )
                    if not status:
                        print(f"ERROR Status is {status}")
                    print(f"---Added {converted_model_path}")


if __name__ == "__main__":
    kit = SimulationApp()

    import omni
    from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

    enable_extension("omni.kit.asset_converter")
    parser = argparse.ArgumentParser("Convert GLB assets to USD")
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=None,
        help="List of folders to convert (space seperated).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=50,
        help="If specified, convert up to `max-models` per folder.",
    )
    parser.add_argument(
        "--load-materials",
        action="store_true",
        help="If specified, materials will be loaded from meshes",
    )
    parser.add_argument(
        "--dist-folder",
        type=str,
        default="obj",
        help="If specified, converted assets will be placed in this folder.",
    )
    args, unknown_args = parser.parse_known_args()
    dist_folder = Path(args.dist_folder)
    dist_folder.mkdir(parents=True, exist_ok=True)
    if args.folders is not None:
        asset_convert(args)
    else:
        print(f"No folders specified via --folders argument, exiting")
    kit.close()
