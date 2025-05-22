import argparse
import asyncio

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
    args = parser.parse_args()

    result = await run_image_generation_loop(args.prompt, args.refs)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
