#!/usr/bin/env python3
"""
Infinigen MCP Server

This MCP server provides tools for interacting with Infinigen, a procedural generation
framework for creating realistic 3D scenes and assets.

Author: AI Assistant
"""

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
app = FastMCP("Infinigen MCP Server")


# Helper functions
def get_infinigen_root() -> Path:
    """Get the Infinigen repository root directory."""
    return Path(__file__).parent.parent


def get_objects_path() -> Path:
    """Get the path to infinigen objects."""
    return get_infinigen_root() / "infinigen" / "assets" / "objects"


def get_available_assets() -> List[str]:
    """Get list of available asset factories."""
    objects_path = get_objects_path()
    assets = []

    if objects_path.exists():
        for subdir in sorted(objects_path.iterdir()):
            if subdir.is_dir():
                clsname = subdir.name.split(".")[0].strip()
                try:
                    module = importlib.import_module(
                        f"infinigen.assets.objects.{clsname}"
                    )
                    # Look for Factory classes
                    for attr_name in dir(module):
                        if attr_name.endswith("Factory"):
                            assets.append(f"{clsname}.{attr_name}")
                except ImportError:
                    continue

    return assets


def get_scene_types() -> List[str]:
    """Get available scene types from configs."""
    configs_path = Path(__file__).parent / "configs_nature" / "scene_types"
    scene_types = []

    if configs_path.exists():
        for gin_file in configs_path.glob("*.gin"):
            scene_types.append(gin_file.stem)

    return scene_types


def run_infinigen_command(command: List[str], cwd: Optional[Path] = None) -> str:
    """Run an Infinigen command and return output."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd or get_infinigen_root(),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 5 minutes"
    except Exception as e:
        return f"Error: {str(e)}"


# MCP Tools
@app.tool()
def list_available_assets() -> str:
    """List all available asset factories in Infinigen."""
    assets = get_available_assets()
    if not assets:
        return "No assets found. Make sure you're in the correct Infinigen directory."

    return "Available assets:\n" + "\n".join(f"- {asset}" for asset in assets)


@app.tool()
def list_scene_types() -> str:
    """List all available scene types for nature generation."""
    scene_types = get_scene_types()
    if not scene_types:
        return "No scene types found."

    return "Available scene types:\n" + "\n".join(
        f"- {scene_type}" for scene_type in scene_types
    )


@app.tool()
def generate_individual_asset(
    asset_name: str,
    output_folder: str,
    seed: Optional[int] = None,
    num_assets: int = 1,
    configs: Optional[List[str]] = None,
) -> str:
    """Generate individual assets using Infinigen.

    Args:
        asset_name: Name of the asset factory (e.g., "trees.TreeFactory")
        output_folder: Output directory path
        seed: Random seed for generation
        num_assets: Number of assets to generate
        configs: List of config files to use
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_individual_assets.py"),
        "--output_folder",
        output_folder,
        "--asset",
        asset_name,
        "--num_assets",
        str(num_assets),
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if configs:
        cmd.extend(["--configs"] + configs)

    result = run_infinigen_command(cmd)
    return f"Asset generation result:\n{result}"


@app.tool()
def generate_nature_scene(
    output_folder: str,
    scene_type: str = "forest",
    seed: Optional[int] = None,
    configs: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
) -> str:
    """Generate a nature scene using Infinigen.

    Args:
        output_folder: Output directory path
        scene_type: Type of scene to generate (forest, mountain, desert, etc.)
        seed: Random seed for generation
        configs: Additional config files
        tasks: Pipeline tasks to run (coarse, populate, fine_terrain, etc.)
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_nature.py"),
        "--output_folder",
        output_folder,
        "--scene_type",
        scene_type,
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if configs:
        cmd.extend(["--configs"] + configs)

    if tasks:
        cmd.extend(["--tasks"] + tasks)

    result = run_infinigen_command(cmd)
    return f"Nature scene generation result:\n{result}"


@app.tool()
def generate_asset_demo(
    output_folder: str,
    asset_name: str,
    seed: Optional[int] = None,
    configs: Optional[List[str]] = None,
) -> str:
    """Generate an asset demo scene showing the asset in a natural environment.

    Args:
        output_folder: Output directory path
        asset_name: Name of the asset factory
        seed: Random seed for generation
        configs: Additional config files
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_asset_demo.py"),
        "--output_folder",
        output_folder,
        "--asset",
        asset_name,
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if configs:
        cmd.extend(["--configs"] + configs)

    result = run_infinigen_command(cmd)
    return f"Asset demo generation result:\n{result}"


@app.tool()
def generate_material_balls(output_folder: str, seed: Optional[int] = None) -> str:
    """Generate material ball samples for testing materials.

    Args:
        output_folder: Output directory path
        seed: Random seed for generation
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_material_balls.py"),
        "--output_folder",
        output_folder,
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    result = run_infinigen_command(cmd)
    return f"Material balls generation result:\n{result}"


@app.tool()
def get_asset_parameters() -> str:
    """Get available asset parameters and their configurations."""
    try:
        from infinigen_examples.asset_parameters import parameters

        return json.dumps(parameters, indent=2)
    except ImportError:
        return "Could not import asset parameters. Make sure you're in the correct directory."


@app.tool()
def list_config_files() -> str:
    """List all available configuration files."""
    configs = []

    # Nature configs
    nature_configs_path = Path(__file__).parent / "configs_nature"
    if nature_configs_path.exists():
        for gin_file in nature_configs_path.rglob("*.gin"):
            configs.append(f"nature/{gin_file.relative_to(nature_configs_path)}")

    # Indoor configs
    indoor_configs_path = Path(__file__).parent / "configs_indoor"
    if indoor_configs_path.exists():
        for gin_file in indoor_configs_path.rglob("*.gin"):
            configs.append(f"indoor/{gin_file.relative_to(indoor_configs_path)}")

    if not configs:
        return "No config files found."

    return "Available config files:\n" + "\n".join(f"- {config}" for config in configs)


@app.tool()
def run_custom_command(
    command: str,
    args: Optional[List[str]] = None,
    working_directory: Optional[str] = None,
) -> str:
    """Run a custom Infinigen command.

    Args:
        command: The Python script to run (e.g., 'generate_nature.py')
        args: List of command line arguments
        working_directory: Working directory (defaults to infinigen root)
    """
    cmd = [sys.executable, str(Path(__file__).parent / command)]

    if args:
        cmd.extend(args)

    cwd = Path(working_directory) if working_directory else get_infinigen_root()

    result = run_infinigen_command(cmd, cwd)
    return f"Command execution result:\n{result}"


@app.tool()
def get_infinigen_status() -> str:
    """Get the current status of Infinigen installation and available features."""
    status = {
        "infinigen_root": str(get_infinigen_root()),
        "examples_path": str(Path(__file__).parent),
        "available_assets": len(get_available_assets()),
        "scene_types": get_scene_types(),
        "python_version": sys.version,
        "working_directory": os.getcwd(),
    }

    return json.dumps(status, indent=2)


# Resources
@app.resource("infinigen://assets")
def get_assets_resource() -> str:
    """Get information about available assets."""
    assets = get_available_assets()
    return json.dumps({"total_assets": len(assets), "assets": assets}, indent=2)


@app.resource("infinigen://configs")
def get_configs_resource() -> str:
    """Get information about available configurations."""
    configs = list_config_files()
    return configs


@app.resource("infinigen://scene_types")
def get_scene_types_resource() -> str:
    """Get information about available scene types."""
    scene_types = get_scene_types()
    return json.dumps(
        {"total_scene_types": len(scene_types), "scene_types": scene_types}, indent=2
    )


# Prompts
@app.prompt()
def generate_scene_prompt(scene_type: str = "forest", style: str = "realistic") -> str:
    """Generate a prompt for creating a specific type of scene.

    Args:
        scene_type: Type of scene (forest, mountain, desert, etc.)
        style: Style of generation (realistic, artistic, etc.)
    """
    return f"""Please generate a {style} {scene_type} scene using Infinigen with the following parameters:

Scene Type: {scene_type}
Style: {style}
Output: High-quality 3D scene with realistic terrain, vegetation, and lighting

Use the generate_nature_scene tool with appropriate seed and configuration."""


@app.prompt()
def asset_creation_prompt(asset_type: str = "tree", complexity: str = "medium") -> str:
    """Generate a prompt for creating a specific type of asset.

    Args:
        asset_type: Type of asset (tree, rock, building, etc.)
        complexity: Complexity level (simple, medium, complex)
    """
    return f"""Please create a {complexity} complexity {asset_type} asset using Infinigen:

Asset Type: {asset_type}
Complexity: {complexity}
Requirements: Procedurally generated, realistic appearance, optimized geometry

Use the generate_individual_asset tool with appropriate parameters and seed."""


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Infinigen MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to use for MCP communication",
    )
    parser.add_argument("--host", default="localhost", help="Host for HTTP transports")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transports"
    )

    args = parser.parse_args()

    logger.info(f"Starting Infinigen MCP Server with {args.transport} transport")

    if args.transport == "stdio":
        import asyncio

        asyncio.run(app.run_stdio_async())
    elif args.transport == "sse":
        import uvicorn

        uvicorn.run(app.sse_app(), host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        # Mount Streamable HTTP server at /mcp and simple health at /health
        import uvicorn
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse, PlainTextResponse
        from starlette.routing import Mount, Route

        async def health(_request):
            return JSONResponse({"status": "ok"})

        # Serve MCP at /mcp to avoid 406 on / and provide a landing text
        async def root(_request):
            return PlainTextResponse(
                "Infinigen MCP Server - use /mcp for MCP transport"
            )

        star_app = Starlette(
            routes=[
                Route("/", root),
                Route("/health", health),
                Mount("/mcp", app.streamable_http_app()),
            ]
        )

        uvicorn.run(star_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
