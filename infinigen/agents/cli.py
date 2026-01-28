#!/usr/bin/env python3
"""
Infinigen Agent CLI - Natural language interface for procedural generation

Usage:
    infinigen-agent "Generate a mountain scene with pine trees"
    infinigen-agent --terrain mountain --seed 42
    infinigen-agent --interactive
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="infinigen-agent",
        description="Infinigen Agent System - AI-powered procedural generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate terrain with specific type
    infinigen-agent --terrain mountain --seed 42

    # Interactive mode
    infinigen-agent --interactive

    # Natural language description
    infinigen-agent "Create a forest scene at sunset"
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )
    mode_group.add_argument(
        "--terrain",
        type=str,
        choices=["mountain", "hills", "valley", "plateau", "desert", "ocean"],
        help="Generate specific terrain type",
    )

    # Common options
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Quality preset (default: medium)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # Positional argument for natural language
    parser.add_argument(
        "description",
        nargs="?",
        type=str,
        help="Natural language scene description",
    )

    return parser


async def run_terrain_generation(
    terrain_type: str,
    seed: int,
    output: Path,
    quality: str,
) -> dict:
    """Run terrain generation with the TerrainEngineerAgent"""
    try:
        from infinigen.agent_deps.core_deps import SeedManager, ValidationManager
        from infinigen.agent_tools.file_tools import FileManager, Logger
        from infinigen.agents.terrain_engineer import TerrainEngineerAgent

        # Initialize dependencies
        file_manager = FileManager()
        logger_tool = Logger()
        seed_manager = SeedManager()
        validation_manager = ValidationManager()

        # Create agent
        agent = TerrainEngineerAgent()

        # Generate terrain
        result = agent.generate_terrain(
            output_folder=output,
            scene_seed=seed,
            file_manager=file_manager,
            logger_tool=logger_tool,
            seed_manager=seed_manager,
            validation_manager=validation_manager,
            terrain_type=terrain_type,
            detail_level=quality,
        )

        return result

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure all dependencies are installed: uv sync")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"success": False, "error": str(e)}


async def run_interactive() -> None:
    """Run interactive mode"""
    print("\n🏔️  Infinigen Agent System - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  terrain <type> [seed]  - Generate terrain")
    print("  scene <description>    - Generate from description")
    print("  help                   - Show help")
    print("  quit                   - Exit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "help":
                print(
                    "\nAvailable terrain types: mountain, hills, valley, plateau, desert, ocean"
                )
                print("Example: terrain mountain 42")
                continue

            parts = user_input.split()
            command = parts[0].lower()

            if command == "terrain":
                terrain_type = parts[1] if len(parts) > 1 else "mountain"
                seed = int(parts[2]) if len(parts) > 2 else 42
                print(f"\n🏔️  Generating {terrain_type} terrain (seed: {seed})...")
                result = await run_terrain_generation(
                    terrain_type=terrain_type,
                    seed=seed,
                    output=Path("output"),
                    quality="medium",
                )
                if result.get("success"):
                    print(
                        f"✅ Terrain generated: {result.get('terrain_file', 'output/')}"
                    )
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")

            elif command == "scene":
                description = " ".join(parts[1:])
                print(f"\n🎬 Processing: {description}")
                print("(Natural language scene generation not yet implemented)")

            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        asyncio.run(run_interactive())
        return 0

    if args.terrain:
        print(f"🏔️  Generating {args.terrain} terrain (seed: {args.seed})...")
        result = asyncio.run(
            run_terrain_generation(
                terrain_type=args.terrain,
                seed=args.seed,
                output=args.output,
                quality=args.quality,
            )
        )
        if result.get("success"):
            print(f"✅ Terrain generated successfully!")
            print(f"   Output: {result.get('terrain_file', args.output)}")
            return 0
        else:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            return 1

    if args.description:
        print(f"🎬 Processing: {args.description}")
        print("(Natural language scene generation coming soon)")
        return 0

    # No arguments - show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
