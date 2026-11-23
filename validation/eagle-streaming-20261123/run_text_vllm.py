"""Run the text-only producer with a CUDA-13 Torch and no optional vision wheel."""
import sys
from pathlib import Path

def main():
    source = Path('/home/songbinbin/vllm/.venv/lib/python3.12/site-packages')
    overlay = Path(__file__).resolve().parent / 'producer-site'
    overlay.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.name.startswith(('torchvision', 'torchaudio', 'torchcodec', 'torch-', 'torch.', 'functorch', 'torchgen')):
            continue
        target = overlay / path.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(path)
    sys.path[:] = [str(overlay) if s == str(source) else s for s in sys.path]
    import os
    os.environ["PYTHONPATH"] += ":" + str(overlay)
    from vllm.entrypoints.cli.main import main as cli
    cli()

if __name__ == '__main__':
    main()
