import argparse
import asyncio
import json

from .loop_controller import run_image_generation_loop


async def main() -> None:
    """Run the image generation loop from the command line."""
    parser = argparse.ArgumentParser(description="Agentic image generation CLI")
    parser.add_argument("prompt", help="Initial text prompt")
    parser.add_argument(
        "--refs",
        nargs="*",
        default=None,
        help="Optional reference image paths or URLs",
    )
    parser.add_argument(
        "--quality",
        default="auto",
        choices=["auto", "low", "medium", "high"],
        help="Quality of the generated image. Defaults to 'high'."
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        choices=["auto", "1024x1024", "1024x1536", "1536x1024"],
        help="Dimensions of the generated image. Defaults to '1024x1024'."
    )
    parser.add_argument(
        "--background",
        default="auto",
        choices=["auto", "opaque", "transparent"],
        help="Background of the generated image (PNG/WEBP only for transparent). Defaults to 'transparent'."
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "jpeg", "webp"],
        help="Output format of the generated image. Defaults to 'png'."
    )

    args = parser.parse_args()

    result = await run_image_generation_loop(
        args.prompt, 
        args.refs,
        args.quality,
        args.size,
        args.background,
        args.format
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
